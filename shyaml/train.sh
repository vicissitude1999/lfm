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

nvidia-smi
cd ..

python train.py --method darts-lfm --epochs 900 --set {} --arch {} --batch_size {} --auxiliary --cutout