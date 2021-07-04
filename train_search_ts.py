import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect_ts import Architect
from teacher import *
from teacher_update import *

from genotypes import PRIMITIVES
from genotypes import Genotype


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../datasets',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate_1', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_2', type=float,
                    default=0.5, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.001, help='min learning rate')
parser.add_argument('--encoder', type=str, default='18')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,
                    default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true',
                    default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float,
                    default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=1e-3, help='weight decay for arch encoding')

# new hyperparams.
parser.add_argument('--weight_gamma', type=float, default=1.0)
parser.add_argument('--weight_lambda', type=float, default=1.0)
parser.add_argument('--model_v_learning_rate', type=float, default=3e-4)
parser.add_argument('--model_v_weight_decay', type=float, default=1e-3)
parser.add_argument('--learning_rate_v', type=float, default=0.025)
parser.add_argument('--learning_rate_r', type=float, default=0.025)
parser.add_argument('--weight_decay_w', type=float, default=3e-4)
parser.add_argument('--weight_decay_h', type=float, default=3e-4)
parser.add_argument('--is_parallel', type=int, default=0)
parser.add_argument('--teacher_arch', type=str, default='18')
parser.add_argument('--is_cifar100', type=int, default=0)
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
CIFAR100_CLASSES = 100

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # device = 'cpu'

  np.random.seed(args.seed)
  if not args.is_parallel:
    torch.cuda.set_device(int(args.gpu))
    logging.info('gpu device = %d' % int(args.gpu))
  else:
    logging.info('gpu device = %s' % args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss().to(device)
  criterion2 = nn.CrossEntropyLoss(reduction='none').to(device)

  # define alphas, model1 and model2 share the same alphas so we can't define them separetely in Network class.
  steps = 4
  k = sum(1 for i in range(steps) for n in range(2 + i))
  num_ops = len(PRIMITIVES)
  alphas_normal = 1e-3 * torch.randn(k, num_ops, requires_grad=True).to(device)
  alphas_reduce = 1e-3 * torch.randn(k, num_ops, requires_grad=True).to(device)
  _arch_parameters = [
    alphas_normal,
    alphas_reduce,
  ]


  if args.is_cifar100:
    model1 = Network(args.init_channels, CIFAR100_CLASSES, args.layers, criterion, criterion2, _arch_parameters).to(device)
    model2 = Network(args.init_channels, CIFAR100_CLASSES, args.layers, criterion, criterion2, _arch_parameters).to(device)
  else:
    model1 = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, criterion2, _arch_parameters).to(device)
    model2 = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, criterion2, _arch_parameters).to(device)

  if args.encoder == '18':
    v = resnet18().cuda()
  elif args.encoder == '34':
    v = resnet34().cuda()
  elif args.encoder == '50':
    v = resnet50().cuda()
  elif args.encoder == '101':
    v = resnet101().cuda()

  r = torch.randn(args.batch_size, requires_grad=True, device=device)

  if args.is_parallel:
    gpus = [int(i) for i in args.gpu.split(',')]
    model1 = nn.parallel.DataParallel(
        model1, device_ids=gpus, output_device=gpus[0])
    model2 = nn.parallel.DataParallel(
        model2, device_ids=gpus, output_device=gpus[0])
    v = nn.parallel.DataParallel(
        v, device_ids=gpus, output_device=gpus[0])
    model1 = model1.module
    model2 = model2.module
    v = v.module

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model1))

  optimizer_1 = torch.optim.SGD(
      model1.parameters(),
      args.learning_rate_1,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  optimizer_2 = torch.optim.SGD(
      model2.parameters(),
      args.learning_rate_2,
      momentum=args.momentum,
      weight_decay=args.weight_decay_w)
  # ??
  optimizer_v = torch.optim.SGD(
      v.parameters(),
      args.learning_rate_v,
      momentum=args.momentum,
      weight_decay=args.weight_decay_h)
  optimizer_r = torch.optim.SGD(
    [r],
    args.learning_rate_r,
    momentum=args.momentum,
    weight_decay=args.weight_decay_h)

  if args.is_cifar100:
    train_transform, valid_transform = utils._data_transforms_cifar100(args)
  else:
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.is_cifar100:
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  else:
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train)) # default is half

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=False, num_workers=4)
  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(
          indices[split:num_train]),
      pin_memory=False, num_workers=4)
  # the dataset for data selection. can be imagenet or so.
  external_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(
          indices[split:num_train]),
      pin_memory=False, num_workers=4)

  scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_1, float(args.epochs), eta_min=args.learning_rate_min)
  scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_2, float(args.epochs), eta_min=args.learning_rate_min)
  scheduler_v = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_v, float(args.epochs), eta_min=args.learning_rate_min)
  scheduler_r = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_r, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model1, model2, v, r, args)

  for epoch in range(args.epochs):
    print("epoch", epoch)
    lr_1 = scheduler_1.get_lr()[0]
    lr_2 = scheduler_2.get_lr()[0]
    lr_v = scheduler_v.get_lr()[0]
    lr_r = scheduler_r.get_lr()[0]
    logging.info('epoch %d lr_1 %e lr_2 %e lr_v %e lr_r %e', epoch, lr_1, lr_2, lr_v, lr_r)

    genotype = model1.genotype()
    logging.info('genotype = %s', genotype)

    # print(F.softmax(model1.alphas_normal, dim=-1))
    # print(F.softmax(model1.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(
        train_queue, valid_queue, external_queue,
        model1, model2, v, r, architect,
        optimizer_1, optimizer_2,
        lr_1, lr_2)
    logging.info('train_acc %f', train_acc)
    scheduler_1.step()
    scheduler_2.step()
    scheduler_v.step()
    scheduler_r.step()
    # validation
    valid_acc, valid_obj = infer(valid_queue, model1, criterion)
    # external_acc, external_obj = infer(external_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)
    # logging.info('external_acc %f', external_acc)

    utils.save(model1, os.path.join(args.save, 'weights_1.pt'))
    utils.save(model2, os.path.join(args.save, 'weights_2.pt'))


def train(train_queue, valid_queue, external_queue,
          model1, model2, v, r, architect,
          optimizer_1, optimizer_2, lr_1, lr_2):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model1.train()
    n = input.size(0) # batch size

    input = input.cuda()
    target = target.cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda(non_blocking=True)

    input_external, target_external = next(iter(external_queue))
    input_external = input_external.cuda()
    target_external = target_external.cuda(non_blocking=True)

    # update alphas
    architect.step(input, target, input_external, target_external,
                   lr_1, lr_2, optimizer_1, optimizer_2,
                   unrolled=args.unrolled)

    # update model 2 parameters
    optimizer_2.zero_grad()
    # compute weights
    encoded_train = v(input)
    encoded_valid = v(input_external)
    x = torch.einsum('ij, kj -> ik', [encoded_train, encoded_valid])
    x = torch.softmax(x, dim=1)  # compute softmax along with the rows

    z = [[1 if target_i == target_j else 0 for target_j in target_external] for target_i in target]
    z = torch.FloatTensor(z).cuda()

    # model1.eval()  # set to eval or not?
    u = -model1._loss(input_external, target_external, reduction='none')
    # model1.train()

    a = torch.sigmoid(torch.matmul(x * z * u, r))

    loss = model2._loss(input, target, reduction='none')
    weighted_loss = torch.dot(a, loss)

    weighted_loss.backward()
    nn.utils.clip_grad_norm_(model2.parameters(), args.grad_clip)
    optimizer_2.step()

    # update model 1 parameters
    optimizer_1.zero_grad()
    loss = model1._loss(input, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model1.parameters(), args.grad_clip)
    optimizer_1.step()

    logits = model1(input)
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    # if step % args.report_freq == 0:
    logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()
