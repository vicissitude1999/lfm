import os
import sys
import time
import glob
import numpy as np
import utils
import logging
import argparse
import json
import copy
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from genotypes import PRIMITIVES
from genotypes import Genotype
from model_search import Network
from combination import *

parser = argparse.ArgumentParser("cifar")
# general
parser.add_argument("--seed", type=int, default=2, help="random seed")
parser.add_argument("--data", type=str, default="../data", help="location of the data corpus")
parser.add_argument("--set", type=str, default="cifar10", help="cifar 10 or 100")
parser.add_argument("--train_portion", type=float, default=0.5, help="portion of training data")
parser.add_argument("--save", type=str, default="outputs/pdarts-lfm/search/debug", help="experiment name")
parser.add_argument("--epochs", type=int, default=25, help="num of training epochs")
parser.add_argument("--batch_size", type=int, default=96, help="batch size")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")

parser.add_argument("--init_channels", type=int, default=16, help="num of init channels")
parser.add_argument("--layers", type=int, default=5, help="total number of layers")
parser.add_argument("--drop_path_prob", type=float, default=0.3, help="drop path probability")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument("--model_beta", type=float, help="beta start value", default=0.5)
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
# optimization related
parser.add_argument("--learning_rate", type=float, default=0.025, help="init learning rate")
parser.add_argument("--learning_rate_min", type=float, default=0.0, help="min learning rate")
parser.add_argument("--learning_rate_beta", type=float, default=2e-3)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--arch_learning_rate", type=float, default=6e-4, help="learning rate for arch encoding")
parser.add_argument("--arch_weight_decay", type=float, default=1e-3, help="weight decay for arch encoding")
# pdarts unique params
parser.add_argument("--dropout_rate", action="append", default=[], help="dropout rate of skip connect")
parser.add_argument("--add_width", action="append", default=["0"], help="add channels")
parser.add_argument("--add_layers", action="append", default=["0"], help="add layers")

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
    
    # models, optimizers, etc
    switches = []
    for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])
    switches_normal = copy.deepcopy(switches)
    switches_reduce = copy.deepcopy(switches)
    # To be moved to args
    num_to_keep = [5, 3, 1]
    num_to_drop = [3, 2, 2]
    if len(args.add_width) == 3:
        add_width = args.add_width
    else:
        add_width = [0, 0, 0]
    if len(args.add_layers) == 3:
        add_layers = args.add_layers
    else:
        add_layers = [0, 6, 12]
    if len(args.dropout_rate) == 3:
        drop_rate = args.dropout_rate
    else:
        drop_rate = [0.0, 0.0, 0.0]
    eps_no_archs = [10, 10, 10]
    for sp in range(len(num_to_keep)):
        model = Network(args.init_channels + int(add_width[sp]),
                        CIFAR_CLASSES,
                        args.layers + int(add_layers[sp]),
                        criterion,
                        switches_normal=switches_normal,
                        switches_reduce=switches_reduce,
                        p=float(drop_rate[sp]),
                        shared_a=None).to(device)
        model_rw = Network(args.init_channels + int(add_width[sp]),
                           CIFAR_CLASSES,
                           args.layers + int(add_layers[sp]),
                           criterion,
                           switches_normal=switches_normal,
                           switches_reduce=switches_reduce,
                           p=float(drop_rate[sp]),
                           shared_a=model.arch_parameters()).to(device)
        model_beta = LinearCombination(args.model_beta).to(device)
        logging.info("param size in MB: [model] {:.2f} [model_rw] {:.2f}".format(
                 utils.count_parameters_in_MB(model), utils.count_parameters_in_MB(model_rw)))
        
        network_params = []
        network_params_rw = []
        for k, v in model.named_parameters():
            if not (k.endswith("alphas_normal") or k.endswith("alphas_reduce")):
                network_params.append(v)
        for k, v in model_rw.named_parameters():
            if not (k.endswith("alphas_normal") or k.endswith("alphas_reduce")):
                network_params_rw.append(v)
        # optimizers
        optimizer = torch.optim.SGD(network_params, args.learning_rate,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_rw = torch.optim.SGD(network_params_rw, args.learning_rate,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(model.arch_parameters(), lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                       weight_decay=args.arch_weight_decay)
        optimizer_beta = torch.optim.Adam(model_beta.parameters(), lr=args.learning_rate_beta, betas=(0.5, 0.999))
        # schedulers
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        scheduler_rw = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_rw, float(args.epochs), eta_min=args.learning_rate_min)

        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2
        
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]["lr"]
            lr_rw = optimizer_rw.param_groups[0]["lr"]
            logging.info("epoch %d lr %e lr_rw %e", epoch, lr, lr_rw)
            
            if epoch < eps_no_arch:
                model.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                model.update_p()
                train_acc, train_obj = train(
                    train_queue, valid_queue, model, model_rw, model_beta, network_params, network_params_rw,
                    criterion, optimizer, optimizer_a, optimizer_rw, optimizer_beta, lr, lr_rw,
                    device, epoch, train_arch=False)
            else:
                model.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor)
                model.update_p()
                train_acc, train_obj, train_obj_rw = train(
                    train_queue, valid_queue, model, model_rw, model_beta, network_params, network_params_rw,
                    criterion, optimizer, optimizer_a, optimizer_rw, optimizer_beta, lr, lr_rw,
                    device, epoch, train_arch=True)
            logging.info("[train] loss {:.4f} loss_rw {:.4f} top1 {:.4f}".format(train_obj, train_obj_rw, train_acc))
            scheduler.step()
            scheduler_rw.step()

            # validation
            if epochs - epoch <= 5:
                with torch.no_grad():
                    valid_acc, valid_obj = infer(valid_queue, model, model_rw, model_beta, criterion)
                    logging.info("[valid] loss {:.4f} top1 {:.4f}".format(valid_obj, valid_acc))
        utils.save(model, os.path.join(args.save, "weights.pt"))
        
        print("------Dropping %d paths------" % num_to_drop[sp])
        # Save switches info for s-c refinement.
        if sp == len(num_to_keep) - 1:
            switches_normal_2 = copy.deepcopy(switches_normal)
            switches_reduce_2 = copy.deepcopy(switches_reduce)
        # drop operations with low architecture weights
        arch_param = model.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_normal[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                # for the last stage, drop all Zero operations
                drop = get_min_k_no_zero(
                    normal_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(normal_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_normal[i][idxs[idx]] = False
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_reduce[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(
                    reduce_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(reduce_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_reduce[i][idxs[idx]] = False
        logging.info("switches_normal = %s", switches_normal)
        logging_switches(switches_normal)
        logging.info("switches_reduce = %s", switches_reduce)
        logging_switches(switches_reduce)

        if sp == len(num_to_keep) - 1:
            arch_param = model.arch_parameters()
            normal_prob = F.softmax(
                arch_param[0], dim=sm_dim).data.cpu().numpy()
            reduce_prob = F.softmax(
                arch_param[1], dim=sm_dim).data.cpu().numpy()
            normal_final = [0 for idx in range(14)]
            reduce_final = [0 for idx in range(14)]
            # remove all Zero operations
            for i in range(14):
                if switches_normal_2[i][0] == True:
                    normal_prob[i][0] = 0
                normal_final[i] = max(normal_prob[i])
                if switches_reduce_2[i][0] == True:
                    reduce_prob[i][0] = 0
                reduce_final[i] = max(reduce_prob[i])
            # Generate Architecture, similar to DARTS
            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            for i in range(3):
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1
            # set switches according the ranking of arch parameters
            for i in range(14):
                if not i in keep_normal:
                    for j in range(len(PRIMITIVES)):
                        switches_normal[i][j] = False
                if not i in keep_reduce:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce[i][j] = False
            # translate switches into genotype
            genotype = parse_network(switches_normal, switches_reduce)
            logging.info(genotype)
            # restrict skipconnect (normal cell only)
            logging.info("Restricting skipconnect...")
            # generating genotypes with different numbers of skip-connect operations
            for sks in range(0, 9):
                max_sk = 8 - sks
                num_sk = check_sk_number(switches_normal)
                if not num_sk > max_sk:
                    continue
                while num_sk > max_sk:
                    normal_prob = delete_min_sk_prob(
                        switches_normal, switches_normal_2, normal_prob)
                    switches_normal = keep_1_on(switches_normal_2, normal_prob)
                    switches_normal = keep_2_branches(
                        switches_normal, normal_prob)
                    num_sk = check_sk_number(switches_normal)
                logging.info("Number of skip-connect: %d", max_sk)
                genotype = parse_network(switches_normal, switches_reduce)
                logging.info(genotype)


def train(train_queue, valid_queue, model, model_rw, model_beta,
          network_params, network_params_rw, criterion,
          optimizer, optimizer_a, optimizer_rw, optimizer_beta,
          lr, lr_rw, device, epoch, train_arch=True):
    objs = utils.AvgrageMeter()
    objr = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    model.train()
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if train_arch:
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search = input_search.to(device, non_blocking=True)
            target_search = target_search.to(device, non_blocking=True)

            # update A, beta
            optimizer_a.zero_grad()
            optimizer_beta.zero_grad()

            logits = model(input_search)
            logits_rw = model_rw(input_search)
            output = model_beta(logits, logits_rw)
            valid_loss = criterion(output, target_search)
            valid_loss.backward()

            nn.utils.clip_grad_norm_(model.arch_parameters(), args.grad_clip)
            optimizer_a.step()
            optimizer_beta.step()

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


def infer(valid_queue, model, model_rw, model_beta, criterion, device, epoch):
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

    return top1.avg, objs.avg


def parse_network(switches_normal, switches_reduce):
    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene

    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)

    concat = range(2, 6)

    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )

    return genotype


def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1

    return index


def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index


def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)


def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:
            count = count + 1

    return count


def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx

    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0

    return probs_out


def keep_1_on(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 2)
        for idx in drop:
            switches[i][idxs[idx]] = False
    return switches


def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False
    return switches


if __name__ == "__main__":
    main()