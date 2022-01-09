#!/bin/sh

#SBATCH --mail-user=renyi@uchicago.edu
#SBATCH --mail-type=ALL

#source activate /renyi-volume/env_darts
source activate env_me

nvidia-smi
cd ..
#python train.py --data /share/data/willett-group/users/renyi/data --method darts-lfm --epochs 900 --set cifar100 \
#--batch_size 128 --auxiliary --cutout --arch darts_lfm_cifar100_4
#python train.py --data /share/data/willett-group/users/renyi/data --method darts-lfm --epochs 900 --set cifar100 \
#--batch_size 128 --auxiliary --cutout --arch darts_lfm_cifar100_5
#python train.py --data /share/data/willett-group/users/renyi/data --method darts-lfm --epochs 900 --set cifar100 \
#--batch_size 128 --auxiliary --cutout --arch darts_lfm_cifar100_6
#python train.py --data /share/data/willett-group/users/renyi/data --method darts-lfm --epochs 900 --set cifar100 \
# --batch_size 77 --auxiliary --cutout --arch darts_lfm_cifar10_4
#python train.py --data /share/data/willett-group/users/renyi/data --method darts-lfm --epochs 900 --set cifar100 \
# --batch_size 83 --auxiliary --cutout --arch darts_lfm_cifar10_5
#python train.py --data /share/data/willett-group/users/renyi/data --method darts-lfm --epochs 900 --set cifar100 \
# --batch_size 85 --auxiliary --cutout --arch darts_lfm_cifar10_6
#python train.py --data /share/data/willett-group/users/renyi/data --method darts-lfm --epochs 900 --set cifar10 \
# --batch_size 77 --auxiliary --cutout --arch darts_lfm_cifar10_4
#python train.py --data /share/data/willett-group/users/renyi/data --method darts-lfm --epochs 900 --set cifar10 \
# --batch_size 82 --auxiliary --cutout --arch darts_lfm_cifar10_5
 python train.py --data /share/data/willett-group/users/renyi/data --method darts-lfm --epochs 900 --set cifar10 \
 --batch_size 85 --auxiliary --cutout --arch darts_lfm_cifar10_6


########################### pcdarts-lfm
#python train.py --method pcdarts-lfm --epochs 900 --set cifar100 --batch_size 80 --auxiliary --cutout --arch pcdarts_lfm_cifar100_1
#python train.py --method pcdarts-lfm --epochs 900 --set cifar100 --batch_size 77 --auxiliary --cutout --arch pcdarts_lfm_cifar100_2
#python train.py --method pcdarts-lfm --epochs 900 --set cifar100 --batch_size 70 --auxiliary --cutout --arch pcdarts_lfm_cifar100_3
#python train.py --method pcdarts-lfm --epochs 900 --set cifar100 --batch_size 70 --auxiliary --cutout --arch pcdarts_lfm_cifar100_4
#python train.py --method pcdarts-lfm --epochs 900 --set cifar100 --batch_size 70 --auxiliary --cutout --arch pcdarts_lfm_cifar100_5
#python train.py --method pcdarts-lfm --epochs 900 --set cifar100 --batch_size 70 --auxiliary --cutout --arch pcdarts_lfm_cifar100_6
#
#python train.py --method pcdarts-lfm --epochs 900 --set cifar100 --batch_size 75 --auxiliary --cutout --arch pcdarts_lfm_cifar10_4
#python train.py --method pcdarts-lfm --epochs 900 --set cifar100 --batch_size 75 --auxiliary --cutout --arch pcdarts_lfm_cifar10_5
#python train.py --method pcdarts-lfm --epochs 900 --set cifar100 --batch_size 75 --auxiliary --cutout --arch pcdarts_lfm_cifar10_6
# python train.py --method pcdarts-lfm --epochs 900 --set cifar100 --batch_size 75 --auxiliary --cutout --arch pcdarts_lfm_cifar10_7

########################## pdarts-lfm
#python train.py --method pdarts-lfm --epochs 900 --set cifar100 --batch_size 92 --auxiliary --cutout --arch pdarts_lfm_cifar100_1
#python train.py --method pdarts-lfm --epochs 900 --set cifar100 --batch_size 92 --auxiliary --cutout --arch pdarts_lfm_cifar100_2
#python train.py --method pdarts-lfm --epochs 900 --set cifar100 --batch_size 92 --auxiliary --cutout --arch pdarts_lfm_cifar100_3
#python train.py --method pdarts-lfm --epochs 900 --set cifar100 --batch_size 92 --auxiliary --cutout --arch pdarts_lfm_cifar10_1
#python train.py --method pdarts-lfm --epochs 900 --set cifar100 --batch_size 92 --auxiliary --cutout --arch pdarts_lfm_cifar10_2
#python train.py --method pdarts-lfm --epochs 900 --set cifar100 --batch_size 92 --auxiliary --cutout --arch pdarts_lfm_cifar10_3

#python train.py --method pdarts-lfm --epochs 900 --set cifar10 --batch_size 105 --auxiliary --cutout --arch pdarts_lfm_cifar10_2
#python train.py --method pdarts-lfm --epochs 900 --set cifar10 --batch_size 95 --auxiliary --cutout --arch pdarts
#python train.py --method pdarts-lfm --epochs 900 --set cifar100 --batch_size 100 --auxiliary --cutout --arch pdarts