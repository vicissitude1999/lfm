import os
import sys
import time
import glob
import logging
import argparse
import random
import builtins

import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from model import NetworkImageNet as Network
import utils
import genotypes

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--method', type=str, default='darts')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')

parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--save', type=str, default='eval-ImNet', help='experiment name')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--resume_dir', type=str)
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
parser.add_argument('--num_workers', type=int, default=32, help='number of workers')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')

args = parser.parse_args()

def setup_args():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    ngpus = torch.cuda.device_count()
    args.ngpus = ngpus

    # learning rate is 0.5 for pcdarts-based, 0.1 everyone else
    if args.method in ['pcdarts', 'pcdarts-lfm']:
        args.learning_rate = 0.5
    if not args.resume:
        run = 'runs_trash' if args.debug else 'runs'
        args.save = os.path.join(run, args.method, '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S")))
    else:
        args.save = os.path.join('runs', args.method, args.resume_dir)

def setup_logging():
    print(args)
    if is_master_proc(args):
        dirs = ['runs', 'runs_trash']
        for d in dirs:
            os.makedirs(os.path.join(d, args.method), exist_ok=True)
        if not args.resume:
            utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        # tensorboard writer
        writer = SummaryWriter(args.save)

        return writer
    else:
        def print_none(*args, **kwargs):
            pass
        # 将内置print函数变为一个空函数，从而使非主进程的进程不会输出
        builtins.print = print_none

        return None


def is_master_proc(args):
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() % args.ngpus == 0
    else:
        return True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs - epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(args):
    setup_args()
    writer = setup_logging()
    logging.info("args = %s", args)
    set_seed(args.seed)

    CLASSES = 1000
    cudnn.benchmark = True
    cudnn.enabled = True
    cur_device = torch.cuda.current_device()

    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')

    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cur_device], output_device=cur_device)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    train_transform, valid_transform = utils._data_transforms_imagenet(args)
    train_data = dset.ImageFolder(os.path.join(args.data, 'imagenet/train'), train_transform)
    valid_data = dset.ImageFolder(os.path.join(args.data, 'imagenet/val'), valid_transform)

    # DPP part
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)


    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers,
        sampler=train_sampler)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    #    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    start_epoch = -1
    if args.resume:
        checkpoint = torch.load(os.path.join(args.save, 'checkpoint.pth.tar'))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    best_acc_top1 = 0.0
    best_acc_top5 = 0.0
    is_best = False
    lr = args.learning_rate

    for epoch in range(start_epoch+1, args.epochs):
        start_time = time.time()

        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)

        logging.info('epoch %d lr %e', epoch, current_lr)
        if epoch < 5:
            for param_group in optimizer.param_groups:
                if args.method in ['pdarts', 'pdarts-lfm']:
                    param_group['lr'] = current_lr * (epoch + 1) / 5.0
                else:
                    param_group['lr'] = lr * (epoch + 1) / 5.0
            if args.method in ['pdarts', 'pdarts-lfm']:
                logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
            else:
                logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        # train function start
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.train()

        for step, (input, target) in enumerate(train_queue):
            input = input.to(cur_device, non_blocking=True)
            target = target.to(cur_device, non_blocking=True)
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
                logging.info('train %03d loss %e top1 %f top5 %f', step, objs.avg, top1.avg, top5.avg)
            writer.add_scalar('LossBatch/train', objs.avg, epoch * len(train_queue) + step)
            writer.add_scalar('AccuBatch/train', top1.avg, epoch * len(train_queue) + step)
            writer.add_scalar('AccuBatchTop5/train', top5.avg, epoch * len(train_queue) + step)
            if args.debug:
                break

        writer.add_scalar('LossEpoch/train', objs.avg, epoch)
        writer.add_scalar('AccuEpoch/train', top1.avg, epoch)
        writer.add_scalar('AccuEpochTop5/train', top5.avg, epoch)

        train_acc, train_obj = top1.avg, objs.avg
        logging.info('train_acc %f train_loss %e', train_acc, train_obj)
        # train function end

        # val function start
        with torch.no_grad():
            objs = utils.AvgrageMeter()
            top1 = utils.AvgrageMeter()
            top5 = utils.AvgrageMeter()
            model.eval()

            for step, (input, target) in enumerate(valid_queue):
                input = input.to(cur_device, non_blocking=True)
                target = target.to(cur_device, non_blocking=True)

                logits, _ = model(input)
                loss = criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % args.report_freq == 0:
                    logging.info('valid %03d loss %e top1 %f top5 %f', step, objs.avg, top1.avg, top5.avg)
                if args.debug:
                    break

                writer.add_scalar('LossEpoch/valid', objs.avg, epoch)
                writer.add_scalar('AccuEpoch/valid', top1.avg, epoch)
                writer.add_scalar('AccuEpochTop5/valid', top5.avg, epoch)

            valid_acc_top1, valid_acc_top5, valid_obj = top1.avg, top5.avg, objs.avg
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        else:
            is_best = False
        logging.info('valid_acc %f best_acc %f valid_loss %e', valid_acc_top1,  best_acc_top1, valid_obj)
        logging.info('valid_acc_top5 %f best_acc_top5 %f', valid_acc_top5, best_acc_top5)

        # save checkpoint
        if is_master_proc(args):
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()}, is_best, args.save)
        if args.debug and epoch - start_epoch > 3:
            break

        end_time = time.time()
        duration = end_time - start_time
        logging.info('Epoch time: %ds.' % duration)


def run(local_rank, func, args):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:9999',
        world_size=args.ngpus,
        rank=local_rank,
    )

    torch.cuda.set_device(local_rank)
    func(args)


def main(args, func, daemon=False):
    setup_args()
    torch.multiprocessing.spawn(run, nprocs=args.ngpus, args=(func, args), daemon=daemon)


if __name__ == '__main__':
    main(args, train)