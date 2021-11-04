import os
import sys
sys.path.append("..")
import time
import glob
import logging
import argparse
import copy

import numpy as np
import torch
import torch.utils
import torch.nn as nn
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import utils
from model_search import Network
from architect import Architect
from combination import LinearCombination

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# New hyperparameters
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--learning_rate_beta', type=float, default=2e-3)
parser.add_argument('--model_beta', type=float, help='beta initial value, -1 means unif[0.45, 0.55]', default=-1)

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
# tensorboard writer
writer = SummaryWriter(args.save)

CIFAR_CLASSES = 10
if args.set == 'cifar100':
    CIFAR_CLASSES = 100


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    ngpu = torch.cuda.device_count()
    logging.info('ngpu = %d', ngpu)
    gpus = list(range(ngpu))

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu devices = %s' % gpus)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    reweighted_model = copy.deepcopy(model)
    model_beta = LinearCombination(args.model_beta).cuda()
    if ngpu > 1:
        model = nn.parallel.DataParallel(model, device_ids=gpus, output_device=gpus[0])
        reweighted_model = nn.parallel.DataParallel(reweighted_model, device_ids=gpus, output_device=gpus[0])
        model_beta = nn.parallel.DataParallel(model_beta, device_ids=gpus, output_device=gpus[0])
        model = model.module
        reweighted_model = reweighted_model.module
        model_beta = model_beta.module
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer_rw = torch.optim.SGD(reweighted_model.parameters(), args.learning_rate, momentum=args.momentum,
                                   weight_decay=args.weight_decay)

    # data
    train_transform, valid_transform = utils._data_transforms_cifar(args)
    if args.set == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.num_workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.num_workers)
    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_rw = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_rw, float(args.epochs), eta_min=args.learning_rate_min)


    architect = Architect(model, reweighted_model, model_beta, args)

    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        lr_rw = scheduler_rw.get_last_lr()[0]
        logging.info('epoch %d lr %e lr_rw %e', epoch, lr, lr_rw)

        genotype = model.genotype()
        genotype_rw = reweighted_model.genotype()
        logging.info('genotype = %s', genotype)
        logging.info('genotype_rw = %s', genotype_rw)
        logging.info('beta = %f', model_beta.beta)

        writer.add_scalar('LR/lr', lr, epoch)
        writer.add_scalar('LR/lr_rw', lr_rw, epoch)
        writer.add_text('genotype', str(genotype), epoch)
        writer.add_scalar('beta', model_beta.beta, epoch)


        # print("Model")
        # print(F.softmax(model.alphas_normal, dim=-1))
        # print(F.softmax(model.alphas_reduce, dim=-1))
        # print("Reweighted Model")
        # print(F.softmax(reweighted_model.alphas_normal, dim=-1))
        # print(F.softmax(reweighted_model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj, train_obj_rw = train(
            train_queue, valid_queue, model, reweighted_model, architect, criterion,
            optimizer, optimizer_rw, lr, lr_rw, model_beta, epoch)
        logging.info('train_acc %f train_loss %e %e', train_acc, train_obj, train_obj_rw)
        scheduler.step()
        scheduler_rw.step()

        # validation
        if args.epochs - epoch <= 5:
            with torch.no_grad():
                valid_acc, valid_obj = infer(valid_queue, model, reweighted_model, model_beta, criterion, epoch)
                logging.info('valid_acc %f valid_loss %e', valid_acc, valid_obj)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, reweighted_model, architect, criterion,
          optimizer, optimizer_rw, lr, lr_rw, model_beta, epoch):
    objs = utils.AvgrageMeter()
    objr = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_valid, target_valid = next(iter(valid_queue))
        input_valid = input_valid.cuda()
        target_valid = target_valid.cuda(non_blocking=True)

        architect.step(input, target, input_valid, target_valid, lr, lr_rw,
                       optimizer, optimizer_rw, unrolled=args.unrolled)
        # copy arch parameters of W1 to W2
        for v, v_rw in zip(model.arch_parameters(), reweighted_model.arch_parameters()):
            v_rw.data.copy_(v.data)

        # Update the model W1 parameters using initial loss function
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Update the second model W2 parameters using reweighted loss function
        optimizer_rw.zero_grad()
        with torch.no_grad():
            logits = model(input)
            weights = F.cross_entropy(logits, target, reduction='none')
            weights = weights / weights.sum()
            # normalizing the weights seems legit because otherwise the weights can be very small
        logits_rw = reweighted_model(input)
        loss_rw = F.cross_entropy(logits_rw, target, reduction='none')
        loss_rw = torch.dot(loss_rw, weights)
        loss_rw.backward()
        nn.utils.clip_grad_norm_(reweighted_model.parameters(), args.grad_clip)
        optimizer_rw.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        objr.update(loss_rw.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d loss %e loss_rw %e top1 %f top5 %f', step, objs.avg, objr.avg, top1.avg, top5.avg)
            writer.add_scalar('LossBatch/train', objs.avg, epoch * len(train_queue) + step)
            writer.add_scalar('LossRWBatch/train', objr.avg, epoch * len(train_queue) + step)
            writer.add_scalar('AccuTop1/train', top1.avg, epoch * len(train_queue) + step)
            writer.add_scalar('AccuTop5/train', top5.avg, epoch * len(train_queue) + step)

        writer.add_scalar('LossEpoch/train', objs.avg, epoch)
        writer.add_scalar('LossRWEpoch/train', objr.avg, epoch)
        writer.add_scalar('AccuEpoch/train', top1.avg, epoch)

    return top1.avg, objs.avg, objr.avg


def infer(valid_queue, model, reweighted_model, model_beta, criterion, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        logits = model(input)
        logits_rw = reweighted_model(input)
        output = model_beta(logits, logits_rw)
        loss = criterion(output, target)

        prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d loss %e top1 %f top5 %f', step, objs.avg, top1.avg, top5.avg)
            writer.add_scalar('LossBatch/valid', objs.avg, epoch * len(valid_queue) + step)
            writer.add_scalar('AccuTop1/valid', top1.avg, epoch * len(valid_queue) + step)
            writer.add_scalar('AccuTop5/valid', top5.avg, epoch * len(valid_queue) + step)

        writer.add_scalar('LossEpoch/valid', objs.avg, epoch)
        writer.add_scalar('AccuEpoch/valid', top1.avg, epoch)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
