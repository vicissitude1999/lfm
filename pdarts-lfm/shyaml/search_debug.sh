nvidia-smi
conda activate env_darts

# 3.5g vram
python search.py --set cifar10 --batch_size 16 --model_beta -1
python search.py --set cifar100 --batch_size 16 --model_beta -1