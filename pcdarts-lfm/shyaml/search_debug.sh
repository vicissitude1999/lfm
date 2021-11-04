nvidia-smi
conda activate env_darts

# 3.5g vram
cd ..
python search.py --set cifar10 --batch_size 30 --model_beta -1 --seed 2
#python search.py --set cifar100 --batch_size 30 --model_beta -1 --seed 2

#python search.py --set cifar10 --batch_size 30 --model_beta -1 --seed 2 --unrolled
#python search.py --set cifar100 --batch_size 30 --model_beta -1 --seed 2 --unrolled