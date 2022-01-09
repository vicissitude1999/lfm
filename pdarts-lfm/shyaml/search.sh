#!/bin/sh

#SBATCH --mail-user=renyi@uchicago.edu
#SBATCH --mail-type=ALL

source activate env_me
nvidia-smi
cd ..
# 11g vram
#python search.py --set cifar100 --batch_size 32 --seed 1 --model_beta 0.8
#python search.py --set cifar100 --batch_size 32 --seed 2 --model_beta 0.8
#python search.py --set cifar100 --batch_size 32 --seed 3 --model_beta 0.8

#python search.py --set cifar10 --batch_size 32 --seed 1 --model_beta 0.8
#python search.py --set cifar10 --batch_size 32 --seed 2 --model_beta 0.8
python search.py --set cifar10 --batch_size 32 --seed 3 --model_beta 0.8