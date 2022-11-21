import logging
import os

import torchvision

from torch.utils.data import SubsetRandomSampler
import torchvision.datasets as datasets
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy

from RandAugment.augmentations import *
from RandAugment.common import get_logger
from RandAugment.imagenet import ImageNet


logger = get_logger('RandAugment')
logger.setLevel(logging.INFO)
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
_TINY_MEAN, _TINY_STD = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)

aa_policy = {'cifar10': AutoAugmentPolicy.CIFAR10, 'cifar100': AutoAugmentPolicy.CIFAR10, 'svhn': AutoAugmentPolicy.SVHN}


def get_meanvar(dataset):
    if 'cifar' in dataset or 'svhn' in dataset:
        mean, std = _CIFAR_MEAN, _CIFAR_STD
    elif dataset == 'tiny':
        mean, std = _TINY_MEAN, _TINY_STD
    else:
        raise ValueError
    return mean, std


def denormalization(imgs, dataset='cifar10'):
    mean, std = get_meanvar(dataset)
    device = imgs.device
    imgs = imgs * torch.tensor(std, device=device).view(3,1,1) + torch.tensor(mean, device=device).view(3,1,1)
    return imgs


class TwoCropsTransform:
    """Get the augmented img in company with the un-augmented img."""
    def __init__(self, base_transform):
        self.transform = base_transform
        self.transforms = base_transform.transforms
        self.transform_ori = transforms.Compose(self.transforms.copy())

    def __call__(self, x):
        img_ori = self.transform_ori(x)
        img_aug = self.transform(x)
        return img_ori, img_aug


def get_dataloaders(args):
    if 'cifar' in args.dataset or 'svhn' in args.dataset:
        normalize = transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
        transform_train = TwoCropsTransform(transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    elif 'imagenet' in args.dataset:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif 'tiny' in args.dataset:
        normalize = transforms.Normalize(_TINY_MEAN, _TINY_STD)
        transform_train = TwoCropsTransform(transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise ValueError('dataset=%s' % args.dataset)

    logger.debug('augmentation: %s' % args.aug)
    if args.aug == 'ra':
        transform_train.transforms.insert(0, RandAugment(args.N, args.M))
    elif args.aug == 'aa':
        transform_train.transforms.insert(0, AutoAugment(policy=aa_policy[args.dataset]))
    elif args.aug == 'severe':
        transform_train.transforms.insert(0, SevereAugment(args.N, args.M))
    elif args.aug == 'jigsaw':
        transform_train.transforms.append(Jigsaw())
    elif args.aug == 'rotate':
        transform_train.transforms.insert(0, rotate)
    elif args.aug == 'vertical':
        transform_train.transforms.insert(2, transforms.RandomVerticalFlip(p=1.))
    elif args.aug == 'blur':
        transform_train.transforms.insert(0, GaussianBlur())
    elif args.aug == 'sobel':
        transform_train.transforms.insert(0, sobel)
    elif args.aug in ['default']:
        pass
    else:
        raise ValueError('not found augmentations. %s' % args.aug)

    if args.cutout > 0:
        transform_train.transforms.append(CutoutDefault(args.cutout))

    if hasattr(transform_train, 'transform_ori'):
        print(transform_train.transform_ori)
        print(transform_train.transform)

    if args.dataset == 'cifar10':
        total_trainset = datasets.CIFAR10(root=args.dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.dataroot, train=False, download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        total_trainset = datasets.CIFAR100(root=args.dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.dataroot, train=False, download=True, transform=transform_test)
    elif args.dataset == 'svhn':
        total_trainset = datasets.SVHN(root=args.dataroot, split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root=args.dataroot, split='test', download=True, transform=transform_test)
    elif args.dataset == 'imagenet':
        total_trainset = ImageNet(root=os.path.join(args.dataroot, 'imagenet-pytorch'), transform=transform_train)
        testset = ImageNet(root=os.path.join(args.dataroot, 'imagenet-pytorch'), split='val', transform=transform_test)
        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]
    elif args.dataset == 'tiny':
        total_trainset = datasets.ImageFolder(root=os.path.join(args.dataroot, 'tiny-imagenet-200', 'train'), transform=transform_train)
        testset = datasets.ImageFolder(root=os.path.join(args.dataroot, 'tiny-imagenet-200', 'val'), transform=transform_test)
    else:
        raise ValueError('invalid dataset name=%s' % args.dataset)

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    if args.evalDA is True:
        DAOODloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root='DAOOD', transform=transform_test), batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False)
        DAIDloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root='DAID', transform=transform_test), batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False)
        testloader = {'DAOOD': DAOODloader, 'DAID': DAIDloader, 'test': testloader}
    return trainloader, testloader
