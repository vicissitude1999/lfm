#!/usr/bin/env bash
nvidia-smi
#conda env create -f environment.yaml
source activate mytorch
python train_search_ts.py --weight_lambda 1 --weight_gamma 1 --unrolled --is_cifar100 0 --gpu 0 --save debugging --batch_size 32