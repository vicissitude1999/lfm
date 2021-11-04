nvidia-smi
conda activate env_me

# 3g vram
cd ..
#python train.py --debug True --resume True --resume_dir eval-EXP-20211011-171452 \
# --method darts --set cifar100 --batch_size 48 --auxiliary --cutout --arch darts_cifar100_1


# python train.py --set cifar100 --batch_size 44 --auxiliary --cutout --arch darts_cifar100_1
# python train.py --set cifar100 --batch_size 40 --auxiliary --cutout --arch darts_cifar100_2
# python train.py --set cifar100 --batch_size 42 --auxiliary --cutout --arch darts_lfm_cifar100_3

# evaluating on cifar100 using architectures searched on cifar10
#python train.py --set cifar100 --batch_size 24 --auxiliary --cutout --arch darts_lfm_cifar10_1
#python train.py --set cifar100 --batch_size 24 --auxiliary --cutout --arch darts_lfm_cifar10_2
#python train.py --set cifar100 --batch_size 24 --auxiliary --cutout --arch darts_lfm_cifar10_3

#python train.py --set cifar10 --batch_size 24 --auxiliary --cutout --arch darts_lfm_cifar10_1
#python train.py --set cifar10 --batch_size 24 --auxiliary --cutout --arch darts_lfm_cifar10_2
#python train.py --set cifar10 --batch_size 24 --auxiliary --cutout --arch darts_lfm_cifar10_3



# python train.py --set cifar100 --batch_size 22 --auxiliary --cutout --arch pcdarts_lfm_cifar100_1
#python train.py --set cifar100 --batch_size 23 --auxiliary --cutout --arch pcdarts_lfm_cifar100_2
#python train.py --set cifar100 --batch_size 23 --auxiliary --cutout --arch pcdarts_lfm_cifar100_3
python train.py --debug True --resume True --resume_dir eval-EXP-20211017-230806 --method pcdarts-lfm \
--set cifar100 --epochs 900 --batch_size 23 --auxiliary --cutout --arch pcdarts_lfm_cifar100_1

# evaluating on cifar100 using architectures searched on cifar10
#python train.py --set cifar100 --batch_size 22 --auxiliary --cutout --arch pcdarts_lfm_cifar10_1
#python train.py --set cifar100 --batch_size 26 --auxiliary --cutout --arch pcdarts_lfm_cifar10_2
#python train.py --set cifar100 --batch_size 21 --auxiliary --cutout --arch pcdarts_lfm_cifar10_3

# evaluate on cifar10
#python train.py --set cifar10 --batch_size 22 --auxiliary --cutout --arch pcdarts_lfm_cifar10_1
#python train.py --set cifar10 --batch_size 26 --auxiliary --cutout --arch pcdarts_lfm_cifar10_2
#python train.py --set cifar10 --batch_size 21 --auxiliary --cutout --arch pcdarts_lfm_cifar10_3