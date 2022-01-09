#!/bin/sh

#SBATCH --mail-user=renyi@uchicago.edu
#SBATCH --mail-type=ALL

#source activate /renyi-volume/env_darts
source activate env_me

nvidia-smi
cd ..
python train_imagenet.py --method pdarts-lfm --epochs 400 --batch_size 64 --auxiliary --arch pdarts_lfm_cifar10_best --debug