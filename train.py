import os
import sys
import time
import glob
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from model import NetworkCIFAR as Network
import utils
import genotypes

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--method', type=str, default='darts')
parser.add_argument('--set', type=str, default='cifar100', help='which dataset')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')

parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--resume_dir', type=str)
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
args = parser.parse_args()


dirs = ['runs', 'runs_trash']
for d in dirs:
    os.makedirs(os.path.join(d, args.method), exist_ok=True)
if not args.resume:
    run = dirs[1] if args.debug else dirs[0]
    args.save = os.path.join(run, args.method, 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S")))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
else:
    args.save = os.path.join('runs', args.method, args.resume_dir)

# logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
# tensorboard writer
writer = SummaryWriter(args.save)

if args.method not in ['darts', 'darts-lfm']:
    args.drop_path_prob = 0.3
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
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)

    if ngpu > 1:
        model = nn.DataParallel(model, device_ids=gpus)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    train_transform, valid_transform = utils._data_transforms_cifar(args)
    if args.set == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    start_epoch = -1
    if args.resume:
        checkpoint = torch.load(os.path.join(args.save, 'checkpoint.pth.tar'))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    best_acc = 0.0
    is_best = False
    for epoch in range(start_epoch + 1, args.epochs):
        start_time = time.time()

        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        if ngpu > 1:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch)
        logging.info('train_acc %f train_loss %e', train_acc, train_obj)
        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch)
        if valid_acc > best_acc:
            best_acc = valid_acc
            is_best = True
        else:
            is_best = False
        best_acc = max(best_acc, valid_acc)
        logging.info('valid_acc %f best_acc %f valid_loss %e', valid_acc, best_acc, valid_obj)

        scheduler.step()

        # save model at the end of each epoch
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        # save checkpoint
        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict()}, is_best, args.save)
        if args.debug and epoch - start_epoch > 3:
            break

        end_time = time.time()
        duration = end_time - start_time
        print('Epoch time: %ds.' % duration)


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
            logging.info('train %03d loss %e top1 %f top5 %f', step, objs.avg, top1.avg, top5.avg)
            writer.add_scalar('LossBatch/train', objs.avg, epoch*len(train_queue) + step)
            writer.add_scalar('AccuBatch/train', top1.avg, epoch*len(train_queue) + step)
        if args.debug:
            break

    writer.add_scalar('LossEpoch/train', objs.avg, epoch)
    writer.add_scalar('AccuEpoch/train', top1.avg, epoch)

    return top1.avg, objs.avg


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
            logging.info('valid %03d loss %e top1 %f top5 %f', step, objs.avg, top1.avg, top5.avg)
            writer.add_scalar('LossBatch/valid', objs.avg, epoch * len(valid_queue) + step)
            writer.add_scalar('AccuBatch/valid', top1.avg, epoch * len(valid_queue) + step)
        if args.debug:
            break

        writer.add_scalar('LossEpoch/valid', objs.avg, epoch)
        writer.add_scalar('AccuEpoch/valid', top1.avg, epoch)

    return top1.avg, objs.avg


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Eval time: %ds.', duration)