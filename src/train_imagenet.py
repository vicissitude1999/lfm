import argparse
import glob
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from lfm import genotypes, utils
from lfm.model import NetworkImageNet as Network

parser = argparse.ArgumentParser("training imagenet")
# general
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--data', type=str, default="../data", help='dataset directory')
parser.add_argument('--save', type=str, default='outputs/tmp/DEBUG', help='experiment output directory')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--report_freq', type=int, default=100, help='report frequency')
# model
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
# optimizer
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
# scheduler
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
# loss function
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
args, unparsed = parser.parse_known_args()

# create output directory
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
# logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
# tensorboard writer

CLASSES = 1000


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
    
    
def init_seeds(seed=0, cuda_deterministic=True):
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
        logging.info('no gpu device available')
        sys.exit(1)

    local_rank = args.local_rank
    init_seeds(args.seed, False)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    
    logging.info(f"args = {args}")
    genotype = eval("genotypes.%s" % args.arch)
    logging.info(f"genotypes = {genotype}")
        
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().to(local_rank)
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth).to(local_rank)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_data = dset.ImageFolder(
        os.path.join(args.data, 'imagenet/train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(
        os.path.join(args.data, 'imagenet/val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_queue = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    valid_queue = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    
    best_acc_top1 = 0
    for epoch in range(args.epochs):
        logging.info('epoch %d lr %e', epoch, optimizer.param_groups[0]['lr'])
        model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc_top1, train_acc_top5, train_obj = train(train_queue, model, criterion_smooth, optimizer, local_rank)
        logging.info('[train] loss %f top1 %f top5 %f', train_obj, train_acc_top1, train_acc_top5)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        logging.info('[valid] loss %f top1 %f top5 %f top1_best %f',
                     valid_obj, valid_acc_top1, valid_acc_top5, best_acc_top1)
        scheduler.step()
        
        if dist.get_rank() == 0:
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer' : optimizer.state_dict(),
                }, is_best, args.save)


def train(train_queue, model, criterion, optimizer, local_rank):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.to(local_rank)
        target = target.to(local_rank)
        n = input.size(0)
        
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d loss %e top1 %f top5 %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


def infer(valid_queue, model, criterion, local_rank):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    
    for step, (input, target) in enumerate(valid_queue):
        input = input.to(local_rank)
        target = target.to(local_rank)
        n = input.size(0)
        
        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d loss %e top1 %f top5 %f', step, objs.avg, top1.avg, top5.avg)
            
    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main() 
