torchrun --standalone \
--nnodes=1 \
--nproc_per_node 2 \
train_imagenet.py \
--seed 0 \
--save outputs/darts-lfm/cifar10_on_imagenet/DEBUG \
--arch DARTS_CIFAR10_LFM_F3