source activate /renyi-volume/env1

python -m torch.distributed.launch --nproc_per_node=2 \
lfm/train_imagenet.py \
--seed 0 \
--save outputs/darts-lfm/cifar10_on_imagenet/DEBUG \
--batch_size 96 \
--arch DARTS_CIFAR10_LFM_F3