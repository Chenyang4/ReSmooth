# ReSmooth: Detecting and Utilizing OOD Samples when Training with Data Augmentation
The official PyTorch implementation of ReSmooth introduced in the following paper:

> [Chenyang Wang](https://chenyang4.github.io/),  Junjun Jiang, Xiong Zhou, Xianming Liu;
>
> ReSmooth: Detecting and Utilizing OOD Samples when Training with Data Augmentation;
>
> IEEE Transactions on Neural Networks and Learning Systems, 2022.

The overall framework of the proposed method is as follows.
<div align=left>
    <img src=".\figs\framework.bmp" alt="framework" width=64%;" /> </div>

## Introduction

Data augmentation (DA) is a widely used technique for enhancing the training of deep neural networks. Recent DA techniques which achieve state-of-the-art performance always meet the need for diversity in augmented training samples. However, an augmentation strategy that has a high diversity usually introduces out-of-distribution (OOD) augmented samples and these samples consequently impair the performance. To alleviate this issue, we propose ReSmooth, a framework that firstly detects OOD samples in augmented samples and then leverages them. To be specific, we first use a Gaussian mixture model to fit the loss distribution of both the original and augmented samples and accordingly split these samples into in-distribution (ID) samples and OOD samples. Then we start a new training where ID and OOD samples are incorporated with different smooth labels. By treating ID samples and OOD samples unequally, we can make better use of the diverse augmented data. Further, we incorporate our ReSmooth framework with negative data augmentation strategies. By properly handling their intentionally created OOD samples, the classification performance of negative data augmentations is largely ameliorated. Experiments on several classification benchmarks show that ReSmooth can be easily extended to existing augmentation strategies (such as RandAugment, rotate, and jigsaw) and improve on them.

## Requirements

```tex
python=3.9
pytorch>=1.8.1
torchvision>=0.9.1
skimage
sklearn
tqdm
matplotlib
tensorboard
git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```

## Experiments

First, for diverse data augmentation, we provide examples for reproducing results in Table 1. (data will be downloaded automatically)

```bash
# Pretrain models
python train.py --dataset cifar10 --prob 1.0 --gpu 0 --tag cifar10/res18/baseline
python train.py --dataset cifar100 --prob 1.0 --gpu 0 --tag cifar100/res18/baseline
python train.py --dataset svhn --prob 1.0 --gpu 0 --tag svhn/res18/baseline
python train.py --dataset cifar10 --model wresnet28_10 --prob 1.0 --gpu 0 --tag cifar10/w10/baseline
python train.py --dataset cifar100 --model wresnet28_10 --prob 1.0 --gpu 0 --tag cifar100/w10/baseline
python train.py --dataset svhn --model wresnet28_10 --prob 1.0 --gpu 0 --tag svhn/w10/baseline

# ReSmooth results
python train.py --dataset cifar10 --prob 0.0 --M 28 --N 2 --smooth-aug 0.4 --gmm --loss SampleSmooth --gpu 0 --tag cifar10/res18/rs_ra
python train.py --dataset cifar100 --prob 0.2 --M 28 --N 2 --smooth-aug 0.6 --gmm --loss SampleSmooth --gpu 0 --tag cifar100/res18/rs_ra
python train.py --dataset svhn --prob 0.0 --M 28 --N 3 --smooth-aug 0.3 --gmm --loss SampleSmooth --gpu 0 --tag svhn/res18/rs_ra
python train.py --dataset cifar10 --model wresnet28_10 --prob 0.0 --M 28 --N 2 --cutout 16 --smooth-aug 0.4 --gmm --loss SampleSmooth --gpu 0 --tag cifar10/w10/rs_ra
python train.py --dataset cifar100 --model wresnet28_10 --prob 0.2 --M 28 --N 2 --cutout 16 --smooth-aug 0.6 --gmm --loss SampleSmooth --gpu 0 --tag cifar100/w10/rs_ra
python train.py --dataset svhn --model wresnet28_10 --prob 0.0 --M 28 --N 3 --cutout 16 --smooth-aug 0.3 --gmm --loss SampleSmooth --gpu 0 --tag svhn/w10/rs_ra
```

Then, for NDA , we provide examples for reproducing results in Table 2.

```bash
# ReSmooth results
python train.py --dataset cifar10 --aug jigsaw --prob 0.6 --smooth-aug 0.2 --gpu 0 --tag cifar10/res18/rs_jigsaw
python train.py --dataset cifar10 --aug rotate --prob 0.6 --smooth-aug 0.5 --gpu 0 --tag cifar10/res18/rs_rotate
python train.py --dataset cifar100 --aug jigsaw --prob 0.6 --smooth-aug 0.4 --gpu 0 --tag cifar100/res18/rs_jigsaw
python train.py --dataset cifar100 --aug rotate --prob 0.6 --smooth-aug 0.5 --gpu 0 --tag cifar100/res18/rs_rotate
python train.py --dataset cifar10 --model wresnet28_10 --aug jigsaw --cutout 16 --prob 0.6 --smooth-aug 0.3 --gpu 0 --tag cifar10/w10/rs_jigsaw
python train.py --dataset cifar10 --model wresnet28_10 --aug rotate --cutout 16 --prob 0.6 --smooth-aug 0.5 --gpu 0 --tag cifar10/w10/rs_rotate
python train.py --dataset cifar100 --model wresnet28_10 --aug jigsaw --cutout 16 --prob 0.6 --smooth-aug 0.4 --gpu 0 --tag cifar100/w10/rs_jigsaw
python train.py --dataset cifar100 --model wresnet28_10 --aug rotate --cutout 16 --prob 0.6 --smooth-aug 0.5 --gpu 0 --tag cifar100/w10/rs_rotate
```

## Citation

If you find our code or paper useful for your research, please cite our [paper](https://arxiv.org/abs/2205.12606).

```
@article{wang2022resmooth,
  title={ReSmooth: Detecting and Utilizing OOD Samples when Training with Data Augmentation},
  author={Wang, Chenyang and Jiang, Junjun and Zhou, Xiong and Liu, Xianming},
  journal={arXiv preprint arXiv:2205.12606},
  year={2022}
}
```

## References 

- RandAugment: [Paper](https://arxiv.org/abs/1909.13719) [Code](https://github.com/ildoonet/pytorch-randaugment)
- KDforAA: [Paper](https://arxiv.org/abs/2003.11342)
