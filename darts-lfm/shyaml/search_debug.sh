nvidia-smi
conda activate env_darts

# 3.5g vram
cd ..
python search.py --set cifar10 --batch_size 10
# python search.py --set cifar100 --batch_size 10

# python search.py --set cifar10 --batch_size 10 --unrolled
# python search.py --set cifar100 --batch_size 10 --unrolled