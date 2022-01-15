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
# run each search with three different random seeds

python search.py --set {} --batch_size 128 --seed {} --model_beta -1