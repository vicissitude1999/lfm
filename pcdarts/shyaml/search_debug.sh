nvidia-smi
conda activate env_darts

# 3g vram
cd ..
python search.py --set cifar10 --batch_size 56
# python search.py --set cifar100 --batch_size 56
# python search.py --set cifar10 --batch_size 56 --unrolled
# python search.py --set cifar100 --batch_size 56 --unrolled