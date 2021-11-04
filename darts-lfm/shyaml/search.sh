# 11g vram
nvidia-smi
source activate /renyi-volume/env_darts

cd ..

python search.py --set cifar10 --batch_size 36 --seed 2
python search.py --set cifar10 --batch_size 36 --seed 4
python search.py --set cifar10 --batch_size 36 --seed 6

# python search.py --set cifar100 --batch_size 36 --seed 2
# python search.py --set cifar100 --batch_size 36 --seed 4
# python search.py --set cifar100 --batch_size 36 --seed 6

# python search.py --set cifar10 --batch_size 36 --unrolled --seed 2
# python search.py --set cifar10 --batch_size 36 --unrolled --seed 4
# python search.py --set cifar10 --batch_size 36 --unrolled --seed 6

# python search.py --set cifar100 --batch_size 36 --unrolled --seed 2
# python search.py --set cifar100 --batch_size 36 --unrolled --seed 4
# python search.py --set cifar100 --batch_size 36 --unrolled --seed 6