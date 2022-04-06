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
--method darts-lfm \
--set cifar100 \
--save ../outputs/darts-lfm/cifar100_on_cifar100/EVAL \
--batch_size 90 \
--arch darts_lfm_cifar100_3 \
--auxiliary \
--cutout