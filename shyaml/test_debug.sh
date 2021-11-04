nvidia-smi
conda activate xietorch

cd ..
python test.py --set cifar100 --batch_size 512 --auxiliary \
--model_path ../LFM_F3_DARTS_CIFAR100/weights.pt --arch DARTS_CIFAR100_LFM_F3_V2