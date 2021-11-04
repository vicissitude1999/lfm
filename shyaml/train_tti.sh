#!/bin/sh

nvidia-smi
source activate env_me

# 11g vram
cd ..
#python train.py --method darts --set cifar100 --batch_size 200 --auxiliary --cutout --arch darts_cifar100_1

########################### darts
# python train.py --set cifar100 --batch_size 132 --auxiliary --cutout --arch darts_cifar100_1
# python train.py --set cifar100 --batch_size 120 --auxiliary --cutout --arch darts_cifar100_2


########################### darts-lfm
# python train.py --set cifar100 --batch_size 126 --auxiliary --cutout --arch darts_lfm_cifar100_3
# evaluating on cifar100 using architectures searched on cifar10
#python train.py --set cifar100 --batch_size 80 --auxiliary --cutout --arch darts_lfm_cifar10_1
#python train.py --set cifar100 --batch_size 80 --auxiliary --cutout --arch darts_lfm_cifar10_2
#python train.py --set cifar100 --batch_size 80 --auxiliary --cutout --arch darts_lfm_cifar10_3

#python train.py --set cifar10 --batch_size 80 --auxiliary --cutout --arch darts_lfm_cifar10_1
#python train.py --set cifar10 --batch_size 80 --auxiliary --cutout --arch darts_lfm_cifar10_2
# python train.py --set cifar10 --batch_size 80 --auxiliary --cutout --arch darts_lfm_cifar10_3


########################### pcdarts-lfm
# python train.py --set cifar100 --batch_size 81 --auxiliary --cutout --arch pcdarts_lfm_cifar100_1
#python train.py --resume True --resume_dir eval-EXP-20211017-230806 --method pcdarts-lfm \
#--set cifar100 --epochs 900 --batch_size 81 --auxiliary --cutout --arch pcdarts_lfm_cifar100_1
#python train.py --set cifar100 --batch_size 77 --auxiliary --cutout --arch pcdarts_lfm_cifar100_2
# python train.py --method pcdarts-lfm --epochs 900 --set cifar100 --batch_size 290 --auxiliary --cutout --arch pcdarts_lfm_cifar100_3

# evaluating on cifar100 using architectures searched on cifar10
# python train.py --set cifar100 --epochs 900 --batch_size 150 --method pcdarts-lfm  --auxiliary --cutout --arch pcdarts_lfm_cifar10_1  # 22g, slurm-7240454
python train.py --set cifar100 --epochs 900 --batch_size 95 --method pcdarts-lfm  --auxiliary --cutout --arch pcdarts_lfm_cifar10_2  # 11g, slurm-
# python train.py --set cifar100 --epochs 900 --batch_size 75 --method pcdarts-lfm  --auxiliary --cutout --arch pcdarts_lfm_cifar10_3

# evaluate on cifar10
#python train.py --set cifar10 --batch_size 150 --auxiliary --cutout --arch pcdarts_lfm_cifar10_1
#python train.py --set cifar10 --batch_size 180 --auxiliary --cutout --arch pcdarts_lfm_cifar10_2
#python train.py --set cifar10 --batch_size 145 --auxiliary --cutout --arch pcdarts_lfm_cifar10_3

# python train.py --set cifar100 --batch_size 100 --auxiliary --cutout --arch pcdarts_lfm_fixed