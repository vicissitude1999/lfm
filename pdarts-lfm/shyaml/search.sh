nvidia-smi
conda activate mytorch3

# 11g vram
python -W ignore train_search_lfm_f3.py --set cifar100 --batch_size 32 --is_parallel 0 --model_beta 0.5
