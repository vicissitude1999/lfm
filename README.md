# LFM (learning from mistakes)
**Composing LFM with DARTS:**

```
CIFAR-10/100: cd darts-LFM && python train_search_ts.py \\
--weight_lambda 1 --weight_gamma 1 --unrolled\\
--is_cifar100 0/1 --gpu 0 --save xxx
```


# LCT

Code accompanying the paper  
***Learning by Creating Tests, with Application to Neural Architecture Search*** [paper]()

This code is based on the implementation of [P-DARTS](https://github.com/chenxin061/pdarts), [DARTS](https://github.com/quark0/darts) and [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS).

## Architecture Search

<!-- **Search on NAS-Bench-201 Space: (3 datasets to choose from)**

* Data preparation: Please first download the 201 benchmark file and prepare the api follow [this repository](https://github.com/D-X-Y/NAS-Bench-201).

* ```cd 201-space && python train_search_progressive.py``` -->

**Composing LCT with DARTS:**

```
CIFAR-10/100: cd darts-LCT && python train_search_ts.py --teacher_arch 18 \\
--weight_lambda 1 --weight_gamma 1 --unrolled\\
--is_cifar100 0/1 --gpu 0 --save xxx
```

**Composing LCT with P-DARTS:**

```
CIFAR-100: cd pdarts-LCT && python train_search_ts.py --tmp_data_dir ../data --save EXP \\
--add_layers 6 --add_layers 12 \\
--dropout_rate 0.1 --dropout_rate 0.4 --dropout_rate 0.7 \\
--note cifar100-xxx --gpu 0 --cifar100 \\
--weight_lambda 1 --weight_gamma 1 --teacher_arch 18
```

```
CIFAR-10: cd pdarts-LCT && python train_search_ts.py --tmp_data_dir ../data --save EXP \\
--add_layers 6 --add_layers 12 \\
--dropout_rate 0.1 --dropout_rate 0.4 --dropout_rate 0.7 \\
--note cifar10-xxx --gpu 0 \\
--weight_lambda 1 --weight_gamma 1 --teacher_arch 18
```
<!-- * ```ImageNet: cd DARTS-space && python train_search_imagenet.py``` -->

**Composing LCT with PC-DARTS:**

* Data preparation: Please first sample 10% and 2.5% images for earch class as the training and validation set, which is done by pcdarts-LCT/sample_images.py.

```
ImageNet: cd pcdarts-LCT && python train_search_imagenet_ts.py --save xxx --tmp_data_dir xxx \\
--weight_lambda 1 --weight_gamma 1 --teacher_arch 18
```

where you can change the value of lambda and gamma and also the teacher architecture.

## Architecture Evaluation

**Composing LCT with DARTS:**

```
CIFAR-10/100: cd darts-LCT && python train.py --cutout --auxiliary \\
--is_cifar 100 0/1 --arch xxx \\
--seed 3 --save xxx
```

```
ImageNet: cd darts-LCT && python train_imagenet.py --auxiliary --arch xxx
```

**Composing LCT with P-DARTS:**

```
CIFAR-100: cd pdarts-LCT && python train_cifar.py --tmp_data_dir ../data \\
--auxiliary --cutout --save xxx \\
--note xxx --cifar100 --gpu 0 \\
--arch xxx --seed 3
```

```
CIFAR-10: cd pdarts-LCT && python train_cifar.py --tmp_data_dir ../data \\
--auxiliary --cutout --save xxx \\
--note xxx --gpu 0 \\
--arch xxx --seed 3
```

```
ImageNet: cd pdarts-LCT && python train_imagenet.py --auxiliary --arch xxx
```

**Composing LCT with PC-DARTS:**

```
ImageNet: cd pcdarts-LCT && python train_imagenet.py --note xxx --auxiliary --arch xxx
```


## Ablation Study (Search)

**Composing LCT with DARTS:**

```
CIFAR-10/100: cd darts-LCT && python train_search_ts_ab1/train_search_ts_ab4.py \\
--teacher_arch 18 \\
--weight_lambda 1 --weight_gamma 1 --unrolled\\
--is_cifar100 0/1 --gpu 0 --save xxx
```

**Composing LCT with P-DARTS:**

```
CIFAR-100: cd pdarts-LCT && python train_search_ts_ab1/train_search_ts_ab4.py\\
 --tmp_data_dir ../data --save EXP \\
--add_layers 6 --add_layers 12 \\
--dropout_rate 0.1 --dropout_rate 0.4 --dropout_rate 0.7 \\
--note cifar100-xxx --gpu 0 --cifar100 \\
--weight_lambda 1 --weight_gamma 1 --teacher_arch 18
```

```
CIFAR-10: cd pdarts-LCT && python train_search_ts_ab1/train_search_ts_ab4.py \\
--tmp_data_dir ../data --save EXP \\
--add_layers 6 --add_layers 12 \\
--dropout_rate 0.1 --dropout_rate 0.4 --dropout_rate 0.7 \\
--note cifar10-xxx --gpu 0 \\
--weight_lambda 1 --weight_gamma 1 --teacher_arch 18
```

The evaluation is the same as the above.
