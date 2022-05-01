import argparse
import glob
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as dset
from torch.utils.tensorboard import SummaryWriter

from src import utils
from src.combination import LinearCombination
from architect import Architect
from model_search import Network

parser = argparse.ArgumentParser("cifar")
# general
parser.add_argument("--seed", type=int, default=2, help="random seed")
parser.add_argument("--data", type=str, default="../data", help="location of the data corpus")
parser.add_argument("--set", type=str, default="cifar10", help="cifar 10 or 100")
parser.add_argument("--train_portion", type=float, default=0.5, help="portion of training data")
parser.add_argument("--save", type=str, default="outputs/darts-lfm/search/debug", help="experiment name")
parser.add_argument("--epochs", type=int, default=50, help="num of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
# model
parser.add_argument("--unrolled", action="store_true", default=False, help="use one-step unrolled validation loss")
parser.add_argument("--init_channels", type=int, default=16, help="num of init channels")
parser.add_argument("--layers", type=int, default=8, help="total number of layers")
parser.add_argument("--drop_path_prob", type=float, default=0.3, help="drop path probability")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument("--model_beta", type=float, help="beta start value", default=0.5)
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
# optimization related
parser.add_argument("--learning_rate", type=float, default=0.025, help="init learning rate") # lr of w1, w2
parser.add_argument("--learning_rate_min", type=float, default=0.001, help="min learning rate")
parser.add_argument("--learning_rate_beta", type=float, default=2e-4) # lr of beta
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--arch_learning_rate", type=float, default=3e-4, help="learning rate for arch encoding") # lr of A
parser.add_argument("--arch_weight_decay", type=float , default=1e-3, help="weight decay for arch encoding")


args = parser.parse_args()
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
with open(Path(args.save, "args.json"), "w") as f:
    json.dump(vars(args), f)

# logging
log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
fh = logging.FileHandler(Path(args.save, "log.txt"), "w")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
# tensorboard writer
writer = SummaryWriter(args.save)

CIFAR_CLASSES = 10
if args.set == "cifar100":
    CIFAR_CLASSES = 100


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)
    init_seeds(args.seed, False)
    device = "cuda"
    
    logging.info("args = %s", args)
    
    # criterion
    criterion = nn.CrossEntropyLoss().to(device)
    # model
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion).to(device)
    model_rw = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion,
                               shared_a=model.arch_parameters()).to(device)
    model_beta = LinearCombination(args.model_beta).to(device)
    logging.info("param size in MB: [model] {:.2f} [model_rw] {:.2f}".format(
                 utils.count_parameters_in_MB(model), utils.count_parameters_in_MB(model_rw)))
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer_rw = torch.optim.SGD(model_rw.parameters(),
                                   args.learning_rate,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)

    # data
    train_transform, valid_transform = utils._data_transforms_cifar(args.set, args.cutout, args.cutout_length)
    if args.set == "cifar100":
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=4)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=4)
    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_rw = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_rw, float(args.epochs), eta_min=args.learning_rate_min)


    architect = Architect(model, model_rw, model_beta, args)

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        lr_rw = optimizer_rw.param_groups[0]["lr"]
        logging.info("epoch %d lr %e lr_rw %e", epoch, lr, lr_rw)

        genotype = model.genotype()
        logging.info("genotype = %s", genotype)

        writer.add_scalar("LR/lr", lr, epoch)
        writer.add_scalar("LR/lr_rw", lr_rw, epoch)
        writer.add_scalar("beta", model_beta.beta, epoch)
        writer.add_text("genotype", str(genotype), epoch)

        # training
        train_acc, train_obj, train_obj_rw = train(
            train_queue, valid_queue, model, model_rw, architect, criterion,
            optimizer, optimizer_rw, lr, lr_rw, model_beta, epoch, device)
        logging.info("[train] loss {:.4f} loss_rw {:.4f} top1 {:.4f}".format(train_obj, train_obj_rw, train_acc))
        scheduler.step()
        scheduler_rw.step()

        # validation
        if args.epochs - epoch <= 5:
            with torch.no_grad():
                valid_acc, valid_obj = infer(valid_queue, model, model_rw, model_beta, criterion, epoch, device)
                logging.info("[valid] loss {:.4f} top1 {:.4f}".format(valid_obj, valid_acc))

        utils.save(model, os.path.join(args.save, "weights.pt"))


def train(train_queue, valid_queue,
          model, model_rw, architect, criterion,
          optimizer, optimizer_rw, lr, lr_rw, model_beta,
          epoch, device):
    objs = utils.AvgrageMeter()
    objr = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    model.train()
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # get a random minibatch from the search queue with replacement
        # why not create iter(valid_queue) right before the for loop?
        # because iter(valid_queue) may run out and raise StopIteration error
        try:
            input_valid, target_valid = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_valid, target_valid = next(valid_queue_iter)
        input_valid = input_valid.to(device, non_blocking=True)
        target_valid = target_valid.to(device, non_blocking=True)

        architect.step(input, target, input_valid, target_valid, lr, lr_rw,
                       optimizer, optimizer_rw, unrolled=args.unrolled)

        # Update the model W1 parameters using initial loss function
        optimizer.zero_grad()
        logits = model(input)
        loss_unraveled = F.cross_entropy(logits, target, reduction="none")
        loss = torch.mean(loss_unraveled)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Update the second model W2 parameters using reweighted loss function
        with torch.no_grad():
            weights = loss_unraveled / torch.sum(loss_unraveled)
            
        optimizer_rw.zero_grad()
        logits_rw = model_rw(input)
        loss_rw_unraveled = F.cross_entropy(logits_rw, target, reduction="none")
        loss_rw = torch.dot(loss_rw_unraveled, weights)
        loss_rw.backward()
        nn.utils.clip_grad_norm_(model_rw.parameters(), args.grad_clip)
        optimizer_rw.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        objr.update(loss_rw.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info("train %03d/%03d loss %e loss_rw %e top1 %f top5 %f",
                         step, len(train_queue), objs.avg, objr.avg, top1.avg, top5.avg)
            writer.add_scalar("LossBatch/train", objs.avg, epoch * len(train_queue) + step)
            writer.add_scalar("LossRWBatch/train", objr.avg, epoch * len(train_queue) + step)
            writer.add_scalar("AccuBatch/train", top1.avg, epoch * len(train_queue) + step)

        writer.add_scalar("LossEpoch/train", objs.avg, epoch)
        writer.add_scalar("LossRWEpoch/train", objr.avg, epoch)
        writer.add_scalar("AccuEpoch/train", top1.avg, epoch)

    return top1.avg, objs.avg, objr.avg


def infer(valid_queue, model, model_rw, model_beta, criterion, epoch, device):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(input)
        logits_rw = model_rw(input)
        output = model_beta(logits, logits_rw)
        loss = criterion(output, target)

        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info("valid %03d loss %e top1 %f top5 %f", step, objs.avg, top1.avg, top5.avg)
            writer.add_scalar("LossBatch/valid", objs.avg, epoch * len(valid_queue) + step)
            writer.add_scalar("AccuBatch/valid", top1.avg, epoch * len(valid_queue) + step)

        writer.add_scalar("LossEpoch/valid", objs.avg, epoch)
        writer.add_scalar("AccuEpoch/valid", top1.avg, epoch)

    return top1.avg, objs.avg


if __name__ == "__main__":
    main()
