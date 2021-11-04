nvidia-smi
source activate /renyi-volume/env_darts

# 11g vram
cd ..
#python search.py --set cifar10 --batch_size 210
#python search.py --set cifar10 --batch_size 210
#python search.py --set cifar10 --batch_size 210

python search.py --set cifar100 --batch_size 210
python search.py --set cifar100 --batch_size 210
python search.py --set cifar100 --batch_size 210


# python search.py --set cifar10 --batch_size 210 --unrolled
# python search.py --set cifar100 --batch_size 210 --unrolled