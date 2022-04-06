#!/bin/sh

# for TTIC cluster
# SBATCH --mail-user=renyi@uchicago.edu
# SBATCH --mail-type=ALL
# source activate env_me

# for PRP cluster
# the prp image specified in yaml already contains a pytorch environment
# may use another conda environment on disk if need to
# source activate /renyi-volume/env_darts

# 11g vram, 1080Ti or 2080Ti

python train.py \
--method pcdarts-lfm \
--set cifar100 \
--save ../outputs/pcdarts-lfm/cifar100_on_cifar100/EVAL \
--batch_size 96 \
--arch PCDARTS_CIFAR100_LFM_F3 \
--auxiliary \
--cutout