import argparse
import glob
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from torch.utils.tensorboard import SummaryWriter

from src import genotypes, utils
from src.model import NetworkCIFAR as Network

parser = argparse.ArgumentParser("cifar")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--method", type=str, default="darts-lfm", help="darts-lfm, etc")
parser.add_argument("--data", type=str, default="../../data", help="location of the data corpus")
parser.add_argument("--set", type=str, default="cifar100", help="cifar10 or cifar100")
parser.add_argument("--save", type=str, default="outputs/tmp/DEBUG", help="experiment name")
parser.add_argument("--epochs", type=int, default=600, help="num of training epochs")
parser.add_argument("--batch_size", type=int, default=96, help="batch size")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--resume_dir", type=str)

parser.add_argument("--arch", type=str, default="DARTS", help="which architecture to use")
parser.add_argument("--init_channels", type=int, default=36, help="num of init channels")
parser.add_argument("--layers", type=int, default=20, help="total number of layers")
parser.add_argument("--auxiliary", action="store_true", default=False, help="use auxiliary tower")
parser.add_argument("--auxiliary_weight", type=float, default=0.4, help="weight for auxiliary loss")
parser.add_argument("--drop_path_prob", type=float, default=0.2, help="drop path probability")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")

parser.add_argument("--learning_rate", type=float, default=0.025, help="init learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")

args = parser.parse_args()
################################### not very clean way
if args.method not in ["darts", "darts-lfm"]:
    args.drop_path_prob = 0.3
###################################
# create output directory
utils.create_exp_dir(args.save, scripts_to_save=None)
with open(Path(args.save, "args.json"), "w") as f:
    json.dump(vars(args), f)

# logging
log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
fh = logging.FileHandler(Path(args.save, "log.txt"), "w+" if args.resume else "w")
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
    
    logging.info("args = %s", args)
    genotype = eval("genotypes.%s" % args.arch)
    logging.info("genotype = %s", genotype)
    
    if args.set == "cifar10":
        model = Network(args.init_channels, 10, args.layers, args.auxiliary, genotype).cuda()
    elif args.set == "cifar100":
        model = Network(args.init_channels, 100, args.layers, args.auxiliary, genotype).cuda()
    else:
        raise Exception("invalid set name")
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    train_transform, valid_transform = utils._data_transforms_cifar(args.set, args.cutout, args.cutout_length)
    if args.set == "cifar100":
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=4)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_acc_top1 = 0.0
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.save, "checkpoint.pth.tar"))
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    for epoch in range(start_epoch, args.epochs):
        logging.info("epoch %d lr %e", epoch, optimizer.param_groups[0]["lr"])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc_top1, train_acc_top5, train_obj = train(train_queue, model, criterion, optimizer, epoch)
        logging.info("[train] loss %f top1 %f top5 %f", train_obj, train_acc_top1, train_acc_top5)
        with torch.no_grad():
            valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion, epoch)
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        logging.info("[valid] loss %f top1 %f top5 %f top1_best %f",
                     valid_obj, valid_acc_top1, valid_acc_top5, best_acc_top1)
        scheduler.step()
        
        # save checkpoint
        utils.save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_acc_top1": best_acc_top1,
            "scheduler": scheduler.state_dict(),
            "optimizer": optimizer.state_dict()}, is_best, args.save)


def train(train_queue, model, criterion, optimizer, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info("train %03d loss %e top1 %f top5 %f", step, objs.avg, top1.avg, top5.avg)
            writer.add_scalar("LossBatch/train", objs.avg, epoch*len(train_queue) + step)
            writer.add_scalar("AccuBatch/train", top1.avg, epoch*len(train_queue) + step)
    writer.add_scalar("LossEpoch/train", objs.avg, epoch)
    writer.add_scalar("AccuEpoch/train", top1.avg, epoch)

    return top1.avg, top5.avg, objs.avg


def infer(valid_queue, model, criterion, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
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

    return top1.avg, top5.avg, objs.avg


if __name__ == "__main__":
    main()
