from torch import nn
import torch.backends.cudnn as cudnn
import torchvision

from RandAugment.networks.resnet import ResNet
from RandAugment.networks.pyramidnet import PyramidNet
from RandAugment.networks.shakeshake.shake_resnet import ShakeResNet
from RandAugment.networks.wideresnet import WideResNet
from RandAugment.networks.shakeshake.shake_resnext import ShakeResNeXt


class Model(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, num_classes=10, arch=None):
        super(Model, self).__init__()

        resnet_arch = getattr(torchvision.models.resnet, arch)
        net = resnet_arch(num_classes=num_classes)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.feature = nn.Sequential(*self.net[:-1])
        self.classifier = self.net[-1]

    def forward(self, x, statistics=False):
        feat = self.feature(x)  # NxD
        logits = self.classifier(feat)  # NxC
        if statistics:
            return logits, feat
        else:
            return logits


def get_model(name, num_classes=10):
    if name == 'resnet18':
        model = Model(arch='resnet18', num_classes=num_classes)
    elif name == 'resnet50':
        model = ResNet(dataset='imagenet', depth=50, num_classes=num_classes, bottleneck=True)
    elif name == 'resnet200':
        model = ResNet(dataset='imagenet', depth=200, num_classes=num_classes, bottleneck=True)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_classes)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_classes)
    elif name == 'shakeshake26_2x32d':
        model = ShakeResNet(26, 32, num_classes)
    elif name == 'shakeshake26_2x64d':
        model = ShakeResNet(26, 64, num_classes)
    elif name == 'shakeshake26_2x96d':
        model = ShakeResNet(26, 96, num_classes)
    elif name == 'shakeshake26_2x112d':
        model = ShakeResNet(26, 112, num_classes)
    elif name == 'shakeshake26_2x96d_next':
        model = ShakeResNeXt(26, 96, 4, num_classes)
    else:
        raise NameError('no model named, %s' % name)

    model = model.cuda()
    cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
        'tiny': 200,
    }[dataset]
