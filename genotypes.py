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

PDARTS_TS_CIFAR10 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('dil_conv_3x3', 2),
            ('skip_connect', 0),
            ('dil_conv_3x3', 3),
            ('skip_connect', 2),
            ('dil_conv_5x5', 4)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('dil_conv_3x3', 1),
            ('avg_pool_3x3', 0),
            ('dil_conv_5x5', 2),
            ('skip_connect', 0),
            ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 0),
            ('dil_conv_3x3', 2)],
    reduce_concat=range(2, 6))

PDARTS_TS_CIFAR100 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_5x5', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 2),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0),
            ('avg_pool_3x3', 1),
            ('dil_conv_3x3', 1),
            ('sep_conv_5x5', 2),
            ('sep_conv_5x5', 1),
            ('sep_conv_5x5', 2),
            ('avg_pool_3x3', 0),
            ('dil_conv_5x5', 2)],
    reduce_concat=range(2, 6))

DARTS_MINUS_TS_CIFAR10_NEW = Genotype(
    normal=[('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 0),
            ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 2),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_MINUS_TS_CIFAR100_NEW = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 1),
            ('dil_conv_5x5', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0),
            ('sep_conv_5x5', 1),
            ('skip_connect', 2),
            ('avg_pool_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_MINUS_CIFAR10 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 1),
            ('skip_connect', 0)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_MINUS_CIFAR100 = Genotype(
    normal=[('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 0),
            ('skip_connect', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 1),
            ('skip_connect', 1),
            ('skip_connect', 0)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0),
            ('avg_pool_3x3', 1),
            ('skip_connect', 2),
            ('avg_pool_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_MINUS_TS_CIFAR10 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 0),
            ('sep_conv_3x3', 1),
            ('dil_conv_3x3', 0),
            ('dil_conv_5x5', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_MINUS_TS_CIFAR100 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('dil_conv_3x3', 1),
            ('skip_connect', 1),
            ('dil_conv_5x5', 0),
            ('skip_connect', 1),
            ('dil_conv_5x5', 0),
            ('skip_connect', 1),
            ('dil_conv_3x3', 0)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('sep_conv_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0)],
    reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('skip_connect', 0),
            ('dil_conv_3x3', 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('max_pool_3x3', 1),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('max_pool_3x3', 1)],
    reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

DARTS_CIFAR10_TS_1ST = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('dil_conv_5x5', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('skip_connect', 0),
            ('dil_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('max_pool_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_CIFAR100_TS_1ST = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6))

DARTS_CIFAR10_TS_18_V1 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_5x5', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2)],
    reduce_concat=range(2, 6))

DARTS_CIFAR10_TS_18_V2 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_5x5', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2)],
    reduce_concat=range(2, 6))

MY_DARTS_CIFAR10 = Genotype(
    normal=[('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_5x5', 2),
            ('skip_connect', 0),
            ('dil_conv_5x5', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 3),
            ('skip_connect', 2)],
    reduce_concat=range(2, 6))

DARTS_CIFAR10_TS_50 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('dil_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('skip_connect', 3),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

DARTS_CIFAR100_TS_50 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('dil_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1),
            ('max_pool_3x3', 0),
            ('max_pool_3x3', 0),
            ('dil_conv_5x5', 2),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('max_pool_3x3', 0)],
    reduce_concat=range(2, 6))

DARTS_CIFAR100_TS_34 = Genotype(
    normal=[('skip_connect', 0),
            ('dil_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_5x5', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6))

DARTS_CIFAR10_TS_34 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_3x3', 2),
            ('sep_conv_3x3', 0),
            ('dil_conv_5x5', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1),
            ('skip_connect', 0),
            ('skip_connect', 2),
            ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 1),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('skip_connect', 3)],
    reduce_concat=range(2, 6))

# DARTS_CIFAR100 = Genotype(
#     normal=[('skip_connect', 0),
#             ('skip_connect', 1),
#             ('skip_connect', 0),
#             ('skip_connect', 1),
#             ('skip_connect', 0),
#             ('skip_connect', 1),
#             ('skip_connect', 0),
#             ('skip_connect', 1)],
#     normal_concat=range(2, 6),
#     reduce=[('max_pool_3x3', 0),
#             ('dil_conv_3x3', 1),
#             ('skip_connect', 2),
#             ('avg_pool_3x3', 0),
#             ('skip_connect', 2),
#             ('max_pool_3x3', 0),
#             ('skip_connect', 2),
#             ('avg_pool_3x3', 0)],
#     reduce_concat=range(2, 6))
DARTS_CIFAR100_1ST = Genotype(
    normal=[('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6))

DARTS_CIFAR100 = Genotype(
    normal=[('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('skip_connect', 1),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6))

DARTS_CIFAR100_TS_18 = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1),
            ('skip_connect', 0),
            ('skip_connect', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('skip_connect', 1),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6))

DARTS_CIFAR100_ES = Genotype(
    normal=[('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0),
            ('skip_connect', 1),
            ('max_pool_3x3', 0),
            ('dil_conv_3x3', 2),
            ('max_pool_3x3', 0),
            ('sep_conv_5x5', 2),
            ('max_pool_3x3', 0),
            ('skip_connect', 2)],
    reduce_concat=range(2, 6))

DARTS_CIFAR100_TS_18_ES = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 2),
            ('skip_connect', 0)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0),
            ('skip_connect', 1),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('sep_conv_5x5', 3),
            ('max_pool_3x3', 0),
            ('skip_connect', 2)],
    reduce_concat=range(2, 6))

DARTS_CIFAR10_ES = Genotype(
    normal=[('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('dil_conv_5x5', 2),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('skip_connect', 0),
            ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 2)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('skip_connect', 2),
            ('max_pool_3x3', 0)],
    reduce_concat=range(2, 6))

DARTS_CIFAR10_TS_18_ES = Genotype(
    normal=[('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1),
            ('skip_connect', 0),
            ('sep_conv_5x5', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2),
            ('skip_connect', 2),
            ('max_pool_3x3', 0),
            ('avg_pool_3x3', 0),
            ('skip_connect', 2)],
    reduce_concat=range(2, 6))

PDARTS_TS_CIFAR100_GAMMA_2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0),
            ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))
PDARTS_TS_CIFAR100_GAMMA_3 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0),
            ('dil_conv_5x5', 3), ('dil_conv_3x3', 3), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
PDARTS_TS_CIFAR100_GAMMA_0_1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 2),
            ('dil_conv_3x3', 3), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
PDARTS_TS_CIFAR100_GAMMA_0_5 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0),
            ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

DARTS_TS_18_CIFAR10_GAMMA_0_5 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_GAMMA_0_1 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0),
            ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_GAMMA_2 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0),
            ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2),
            ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_GAMMA_3 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0),
            ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2),
            ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

DARTS_TS_18_CIFAR10_LAMBDA_2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 1),
            ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_LAMBDA_0_1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_LAMBDA_0_5 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0),
            ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_LAMBDA_3 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

PDARTS_TS_18_CIFAR100_LAMBDA_3 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 0),
            ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))
PDARTS_TS_18_CIFAR100_LAMBDA_0_1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0),
            ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))
PDARTS_TS_18_CIFAR100_LAMBDA_0_5 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('avg_pool_3x3', 0),
            ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
PDARTS_TS_18_CIFAR100_LAMBDA_2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0),
            ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

PDARTS_TS_18_CIFAR100_AB_1 = Genotype(
    normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0),
            ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0),
            ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
PDARTS_TS_18_CIFAR100_AB_4 = Genotype(
    normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0),
            ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

DARTS_TS_18_CIFAR10_AB_1 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
DARTS_TS_18_CIFAR10_AB_4 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2),
            ('dil_conv_3x3', 3), ('skip_connect', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))

PDARTS_TUNED_CIFAR100 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0),
            ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
PDARTS_TUNED_CIFAR10 = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0),
            ('sep_conv_3x3', 3), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

PCDARTS_TS_IMAGENET = Genotype(
    normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 2),
            ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
PCDARTS_TS_IMAGENET_GAMMA_0_5 = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

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

#### architecture in original paper
PC_DARTS_cifar = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0),
            ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
PC_DARTS_image = Genotype(
    normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

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

# Namespace(arch_learning_rate=0.0006, arch_weight_decay=0.001, batch_size=112, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_parallel=0,
#           layers=8, learning_rate=0.1, learning_rate_beta=0.002, learning_rate_min=0.0, model_beta=-1.0, momentum=0.9,
#           num_workers=4, report_freq=50, save='search-EXP-20210923-072315', seed=6, set='cifar10', train_portion=0.5,
#           unrolled=False, weight_decay=0.0003)
#  param size = 0.299578MB beta = 0.675380 valid_acc 87.431999 99.487999 valid_loss 3.663423e-01
# 16.98%
pcdarts_lfm_cifar10_1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2),
            ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 1),
            ('dil_conv_5x5', 3), ('dil_conv_3x3', 4), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

# Namespace(arch_learning_rate=0.0006, arch_weight_decay=0.001, batch_size=112, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_parallel=0,
#           layers=8, learning_rate=0.1, learning_rate_beta=0.002, learning_rate_min=0.0, model_beta=-1.0,
#           model_path='saved_models', momentum=0.9, num_workers=4, report_freq=50, save='search-EXP-20211017-230851',
#           seed=2, set='cifar10', train_portion=0.5, unrolled=False, weight_decay=0.0003)
# param size = 0.299578MB beta = 0.664686 valid_acc 87.703999 99.539999 valid_loss 3.625478e-01
pcdarts_lfm_cifar10_2 = Genotype(
    normal=[('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 3),
            ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2),
            ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

# Namespace(arch_learning_rate=0.0006, arch_weight_decay=0.001, batch_size=112, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_parallel=0,
#           layers=8, learning_rate=0.1, learning_rate_beta=0.002, learning_rate_min=0.0, model_beta=-1.0,
#           model_path='saved_models', momentum=0.9, num_workers=4, report_freq=50, save='search-EXP-20211018-075429',
#           seed=4, set='cifar10', train_portion=0.5, unrolled=False, weight_decay=0.0003)
# param size = 0.299578MB beta = 0.661631 valid_acc 87.619999 99.571999 valid_loss 3.608355e-01
pcdarts_lfm_cifar10_3 = Genotype(
    normal=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

# Namespace(arch_learning_rate=0.0006, arch_weight_decay=0.001, batch_size=112, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_parallel=0,
#           layers=8, learning_rate=0.1, learning_rate_beta=0.002, learning_rate_min=0.0, model_beta=-1.0, momentum=0.9,
#           num_workers=4, report_freq=50, save='search-EXP-20210923-072228', seed=2, set='cifar100', train_portion=0.5,
#           unrolled=False, weight_decay=0.0003)
# param size = 0.322708MB beta = 0.682372 valid_acc 59.884000 87.575999 valid_loss 1.420301e+00
pcdarts_lfm_cifar100_1 = Genotype(
    normal=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0),
            ('dil_conv_5x5', 3), ('sep_conv_5x5', 4), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
# 17.82% at 900 epochs
# Namespace(arch_learning_rate=0.0006, arch_weight_decay=0.001, batch_size=112, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_parallel=0,
#           layers=8, learning_rate=0.1, learning_rate_beta=0.002, learning_rate_min=0.0, model_beta=-1.0,
#           model_path='saved_models', momentum=0.9, num_workers=4, report_freq=50, save='search-EXP-20211017-004609',
#           seed=4, set='cifar100', train_portion=0.5, unrolled=False, weight_decay=0.0003)
# param size = 0.322708MB  beta = 0.672272 valid_acc 59.971999 87.771999 valid_loss 1.418068e+00
# 18.5% at 900 epochs
pcdarts_lfm_cifar100_2 = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1),
            ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 3),
            ('sep_conv_3x3', 2), ('max_pool_3x3', 3), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

# Namespace(arch_learning_rate=0.0006, arch_weight_decay=0.001, batch_size=112, cutout=False, cutout_length=16,
#           data='../../data', drop_path_prob=0.3, epochs=50, gpu='0', grad_clip=5, init_channels=16, is_parallel=0,
#           layers=8, learning_rate=0.1, learning_rate_beta=0.002, learning_rate_min=0.0, model_beta=-1.0,
#           model_path='saved_models', momentum=0.9, num_workers=4, report_freq=50, save='search-EXP-20211017-004558',
#           seed=6, set='cifar100', train_portion=0.5, unrolled=False, weight_decay=0.0003)
# param size = 0.322708MB beta = 0.665984 valid_acc 60.076000 87.715999 valid_loss 1.419644e+00
# 19% at 900 epochs
pcdarts_lfm_cifar100_3 = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 0), ('sep_conv_3x3', 4), ('sep_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0),
            ('sep_conv_5x5', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

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
#           layers=8, learning_rate=0.1, learning_rate_beta=0.002, learning_rate_min=0.0, model_beta='0.8 fixed', momentum=0.9,
#           num_workers=2, report_freq=50, save='search-EXP-20211005-184741', seed=2, set='cifar100', train_portion=0.5,
#           unrolled=False, weight_decay=0.0003)
# param size = 0.322708MB beta = 0.800000 valid_acc 59.292000 87.251999 valid_loss 1.444009e+00
Genotype(
    normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2),
            ('sep_conv_3x3', 0), ('sep_conv_3x3', 4), ('sep_conv_5x5', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2),
            ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

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
