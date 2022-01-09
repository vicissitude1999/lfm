from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


#### formulation 1
DARTS_CIFAR10_LFM_RES18 = Genotype(
    normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3),
            ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
DARTS_CIFAR10_LFM_RES18_1 = Genotype(
    normal=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3),
            ('sep_conv_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 3), ('max_pool_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))

### formulation 3
DARTS_CIFAR10_LFM_F3 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 3),
            ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

# Namespace(arch_learning_rate=0.0003, arch_weight_decay=0.001, batch_size=60, cutout=False, cutout_length=16, data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_para
# llel=0, layers=8, learning_rate=0.025, learning_rate_beta=0.002, learning_rate_min=0.001, model_beta=-1.0, model_path='saved_models', momentum=0.9, num_workers=4, report_freq=50, save='search-EXP-20211010-222912', seed=2, set='cifa
# r100', train_portion=0.5, unrolled=False, weight_decay=0.0003)
# 10/10 10:29:27 PM param size = 1.953748MB

darts_cifar100_1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0),
            ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0),
            ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

# Namespace(arch_learning_rate
#  =0.0003, arch_weight_decay=0.001, batch_size=38, cutout=False, cutout_length=16, data='../../data',
#  drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_cifar100=0, is_parallel=0, layers=8,
#  learning_rate=0.025, learning_rate_beta=0.002, learning_rate_min=0.001, model_path='saved_models', momentum=0.9,
#  report_freq=50, save='search-EXP-20210910-164045', seed=2, train_portion=0.5, unrolled=False, weight_decay=0.0003)
darts_lfm_cifar10_1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 2), ('max_pool_3x3', 2), ('sep_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_5x5', 3),
            ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
# 17.22, 2.85

# Namespace(arch_learning_rate=0.0003, arch_weight_decay=0.001, batch_size=38, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_cifar100=0,
#           is_parallel=1, layers=8, learning_rate=0.025, learning_rate_beta=0.002, learning_rate_min=0.001,
#           model_path='saved_models', momentum=0.9, report_freq=50, save='search-EXP-20210910-164342', seed=2,
#           train_portion=0.5, unrolled=False, weight_decay=0.0003
# 16.1, 2.66
darts_lfm_cifar10_2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 3),
            ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

# Namespace(arch_learning_rate=0.0003, arch_weight_decay=0.001, batch_size=36, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_parallel=0,
#           layers=8, learning_rate=0.025, learning_rate_beta=0.002, learning_rate_min=0.001, model_beta='0.5',
#           model_path='saved_models', momentum=0.9, num_workers=4, report_freq=50, save='search-EXP-20211003-164809',
#           seed=2, set='cifar10', train_portion=0.5, unrolled=False, weight_decay=0.0003)
# 16.72, 2.74
darts_lfm_cifar10_3 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0),
            ('sep_conv_5x5', 3), ('max_pool_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 1), ('dil_conv_5x5', 2), ('skip_connect', 1),
            ('sep_conv_5x5', 0), ('dil_conv_5x5', 4), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

# ritvik's result
# Namespace(arch_learning_rate=0.0003, arch_weight_decay=0.001, batch_size=72, cutout=False, cutout_length=16,
#           data='../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_cifar100=1,
#           is_parallel=0, layers=8, learning_rate=0.025, learning_rate_beta=0.002, learning_rate_min=0.001,
#           model_path='saved_models', momentum=0.9, report_freq=50, save='search-EXP-20210702-170859', seed=2,
#           train_portion=0.5, unrolled=True, weight_decay=0.0003)
# my run 18.12, his run 17.86
darts_lfm_cifar100_1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2),
            ('max_pool_3x3', 0), ('sep_conv_3x3', 4), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

# Namespace(arch_learning_rate=0.0003, arch_weight_decay=0.001, batch_size=38, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_cifar100=1,
#           is_parallel=0, layers=8, learning_rate=0.025, learning_rate_beta=0.002, learning_rate_min=0.001,
#           model_path='saved_models', momentum=0.9, report_freq=50, save='search-EXP-20210910-163939', seed=2,
#           train_portion=0.5, unrolled=False, weight_decay=0.0003)
# 17.75
darts_lfm_cifar100_2 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0),
            ('avg_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

# Namespace(arch_learning_rate=0.0003, arch_weight_decay=0.001, batch_size=36, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_parallel=0,
#           layers=8, learning_rate=0.025, learning_rate_beta=0.002, learning_rate_min=0.001, model_beta='0.5',
#           model_path='saved_models', momentum=0.9, num_workers=4, report_freq=50, save='search-EXP-20211003-120952',
#           seed=2, set='cifar100', train_portion=0.5, unrolled=False, weight_decay=0.0003)
#
# 17.62
darts_lfm_cifar100_3 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 3), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))


darts_lfm_cifar10_4 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

darts_lfm_cifar10_5 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

darts_lfm_cifar10_6 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 3), ('skip_connect', 0), ('skip_connect', 1)], reduce_concat=range(2, 6))

darts_lfm_cifar100_4 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

darts_lfm_cifar100_5 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 3), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

darts_lfm_cifar100_6 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))


################################################################################################
# architectures in original paper
pcdarts = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0),
            ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
pcdarts_imagenet = Genotype(
    normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

# architectures from my experiments
pcdarts_cifar10_1 = Genotype(
    normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('skip_connect', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 2),
            ('dil_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0)], reduce_concat=range(2, 6))
pcdarts_cifar10_2 = Genotype(
    normal=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 1),
            ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0),
            ('dil_conv_3x3', 3), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
pcdarts_cifar10_3 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 1),
            ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
pcdarts_cifar100_1 = Genotype(
    normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2),
            ('skip_connect', 3), ('sep_conv_5x5', 2), ('skip_connect', 0)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3),
            ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
pcdarts_cifar100_2 = Genotype(
    normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 3),
            ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0),
            ('dil_conv_3x3', 3), ('sep_conv_3x3', 3), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
pcdarts_cifar100_3 = Genotype(
    normal=[('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 4), ('max_pool_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))

# 20210923-072315
pcdarts_lfm_cifar10_1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2),
            ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 1),
            ('dil_conv_5x5', 3), ('dil_conv_3x3', 4), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
# 20211017-230851
pcdarts_lfm_cifar10_2 = Genotype(
    normal=[('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 3),
            ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2),
            ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
# 20211018-075429
pcdarts_lfm_cifar10_3 = Genotype(
    normal=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
# 20211112-090338
pcdarts_lfm_cifar10_4 = Genotype(
    normal=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1),
            ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3),
            ('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
# 20211112-090436
pcdarts_lfm_cifar10_5 = Genotype(
    normal=[('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 3),
            ('dil_conv_5x5', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))
# 20211112-090710
pcdarts_lfm_cifar10_6 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2),
            ('dil_conv_5x5', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
# 20211112-090718
pcdarts_lfm_cifar10_7 = Genotype(
    normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('skip_connect', 1), ('dil_conv_5x5', 2),
            ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
# 20210923-072228 model beta = -1, sampled unif 0.4-0.6
pcdarts_lfm_cifar100_1 = Genotype(
    normal=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0),
            ('dil_conv_5x5', 3), ('sep_conv_5x5', 4), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
# 20211017-004609 model beta = -1, sampled unif 0.45-0.55
pcdarts_lfm_cifar100_2 = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1),
            ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 3),
            ('sep_conv_3x3', 2), ('max_pool_3x3', 3), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
# 20211005-184741 model beta = 0.8
pcdarts_lfm_cifar100_3 = Genotype(
    normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 4), ('sep_conv_5x5', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2),
            ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
# 20211112-090155 model beta = 0.8
pcdarts_lfm_cifar100_4 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 3),
            ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
# 20211112-090216 mode beta = 0.8
pcdarts_lfm_cifar100_5 = Genotype(
    normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 2),
            ('avg_pool_3x3', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
# 20211017-004558 model beta = -1, sampled unif 0.45-0.55
pcdarts_lfm_cifar100_6 = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 0), ('sep_conv_3x3', 4), ('sep_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0),
            ('sep_conv_5x5', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

############################################################################################
pdarts = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
# 20211111-223047
pdarts_lfm_cifar100_1 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
# 20211111-223058
pdarts_lfm_cifar100_2 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
# 20211111-223109
pdarts_lfm_cifar100_3 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 3), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
# 20211111-223115
pdarts_lfm_cifar10_1 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0),
            ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
# 20211111-223123
pdarts_lfm_cifar10_2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))
# 20211111-223130
pdarts_lfm_cifar10_3 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 3), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

###############################################################################################################
pcdarts_lfm_cifar100_best = pcdarts_lfm_cifar100_3
pcdarts_lfm_cifar10_best = pcdarts_lfm_cifar10_6
pdarts_lfm_cifar100_best = pdarts_lfm_cifar100_1
pdarts_lfm_cifar10_best = pdarts_lfm_cifar10_2

############################################################ not useful below
# Namespace(arch_learning_rate=0.0006, arch_weight_decay=0.001, batch_size=112, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_parallel=0,
#           layers=8, learning_rate=0.1, learning_rate_beta=0.002, learning_rate_min=0.0, model_beta='0.5 fixed', momentum=0.9,
#           num_workers=2, report_freq=50, save='search-EXP-20211004-155300', seed=2, set='cifar10', train_portion=0.5,
#           unrolled=False, weight_decay=0.0003)
# param size = 0.299578MB beta = 0.500000 valid_acc 87.115999 99.551999 valid_loss 3.823299e-01
Genotype(
    normal=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 3),
            ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 2),
            ('max_pool_3x3', 3), ('sep_conv_3x3', 4), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

# Namespace(arch_learning_rate=0.0006, arch_weight_decay=0.001, batch_size=112, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_parallel=0,
#           layers=8, learning_rate=0.1, learning_rate_beta=0.002, learning_rate_min=0.0, model_beta='0.8 fixed', momentum=0.9,
#           num_workers=2, report_freq=50, save='search-EXP-20211006-002616', seed=2, set='cifar100', train_portion=0.5,
#           unrolled=False, weight_decay=0.0003)
# param size = 0.322708MB beta = 0.800000 valid_acc 59.360000 87.127999 valid_loss 1.452640e+00
Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2),
            ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('skip_connect', 1),
            ('dil_conv_5x5', 3), ('skip_connect', 1), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

# Namespace(arch_learning_rate=0.0006, arch_weight_decay=0.001, batch_size=112, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_parallel=0,
#           layers=8, learning_rate=0.1, learning_rate_beta=0.002, learning_rate_min=0.0, model_beta='0.8 fixed', momentum=0.9,
#           num_workers=2, report_freq=50, save='search-EXP-20211005-175644', seed=2, set='cifar100', train_portion=0.5,
#           unrolled=False, weight_decay=0.0003)
# param size = 0.322708MB beta = 0.800000 valid_acc 59.852000 87.495999 valid_loss 1.424694e+0
Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0),
            ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('skip_connect', 1), ('sep_conv_5x5', 3),
            ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))

# Namespace(arch_learning_rate=0.0006, arch_weight_decay=0.001, batch_size=112, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_parallel=0,
#           layers=8, learning_rate=0.1, learning_rate_beta=0.002, learning_rate_min=0.0, model_beta='1.0 fixed', momentum=0.9,
#           num_workers=2, report_freq=50, save='search-EXP-20211007-130729', seed=2, set='cifar100', train_portion=0.5,
#           unrolled=False, weight_decay=0.0003)
# param size = 0.322708MB beta = 1.000000 valid_acc 57.983999 86.047999 valid_loss 1.519299e+00
Genotype(
    normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 0),
            ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))
