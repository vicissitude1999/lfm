nvidia-smi
source activate /renyi-volume/env_darts

# 11g vram
cd ..
#python search.py --set cifar10 --batch_size 112 --seed 2
#python search.py --set cifar10 --batch_size 112 --seed 4
#python search.py --set cifar10 --batch_size 112 --seed 6

python search.py --set cifar100 --batch_size 112 --seed 2
python search.py --set cifar100 --batch_size 112 --seed 4
python search.py --set cifar100 --batch_size 112 --seed 6



#python search.py --set cifar10 --batch_size 112 --seed 2 --unrolled
#python search.py --set cifar10 --batch_size 112 --seed 4 --unrolled
#python search.py --set cifar10 --batch_size 112 --seed 6 --unrolled

#python search.py --set cifar100 --batch_size 112 --seed 2 --unrolled
#python search.py --set cifar100 --batch_size 112 --seed 4 --unrolled
#python search.py --set cifar100 --batch_size 112 --seed 6 --unrolled