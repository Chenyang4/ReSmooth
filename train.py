import itertools
import json
import logging
import math
import os
import shutil
import argparse
import time
import datetime
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel.data_parallel import DataParallel

from tqdm import tqdm

from RandAugment.common import get_logger
from RandAugment.data import get_dataloaders, denormalization
from RandAugment.metrics import accuracy, Accumulator
from RandAugment.networks import get_model, num_class
from warmup_scheduler import GradualWarmupScheduler

from RandAugment.common import add_filehandler
from RandAugment.loss import get_criterion, get_loss
from RandAugment.mixture_model import get_gmm
from RandAugment.distillation import get_teacher

logger = get_logger('RandAugment')
logger.setLevel(logging.INFO)


def fair_comparison(model, loader, optimizer, epoch, criterion, scheduler=None, gmm=None, writer=None, verbose=True):
    if verbose:
        loader = tqdm(loader)
        loader.set_description('[fair comparison %04d/%04d]' % (epoch, args.epochs))

    metrics = Accumulator()
    cnt = 0
    steps = 0

    model.train()
    for batch in loader:
        data_ori, data_aug, label = batch[0] + [batch[1],]
        # split data according to prob
        assert args.prob < 1.
        n_ori = int(data_aug.size(0) * args.prob)
        data = data_aug[n_ori:]
        label = label[n_ori:]
        data, label = data.cuda(), label.cuda()

        steps += 1

        preds = model(data)

        assert gmm is not None
        prob = torch.tensor(gmm.predict(data, label), device='cuda')
        index_OOD = prob < 0.5
        index_ID = ~index_OOD

        batch_size = min(index_ID.sum(), index_OOD.sum(), 256)
        if args.fair_comparison == 'DAOOD':
            preds = preds[index_OOD][:batch_size]
            label = label[index_OOD][:batch_size]
        elif args.fair_comparison == 'DAID':
            preds = preds[index_ID][:batch_size]
            label = label[index_ID][:batch_size]
        elif args.fair_comparison == 'mix':
            preds = preds[:batch_size]
            label = label[:batch_size]
        else:
            assert ValueError
        loss = get_loss(criterion, preds, label, prob=prob[:batch_size], statistics=False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        top1, top5 = accuracy(preds, label, (1, 5))

        metrics.add_dict({
            'loss': loss.item() * batch_size,
            'top1': top1.item() * batch_size,
            'top5': top5.item() * batch_size,
        })

        cnt += batch_size
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader.set_postfix(postfix)

    scheduler.step()
    metrics /= cnt
    metrics.metrics['lr'] = optimizer.param_groups[0]['lr']

    writer.add_scalar('train/1.train_acc', metrics['top1'], epoch)
    writer.add_scalar('train/2.train_loss', metrics['loss'], epoch)
    writer.add_image('train/1.origin_image', denormalization(data_ori[0], args.dataset), epoch)
    writer.add_image('train/2.augmented_image', denormalization(data_aug[0], args.dataset), epoch)

    return metrics


def train(model, loader, optimizer, epoch, criterion_org, criterion_aug, scheduler=None, gmm=None, teacher=None, writer=None, statistics=True, verbose=True):
    if verbose:
        loader = tqdm(loader)
        loader.set_description('[train %04d/%04d]' % (epoch, args.epochs))

    metrics = Accumulator()
    cnt = 0
    steps = 0
    if statistics:
        loss_list1 = []
        loss_list2 = []
        smooth_list = []

    model.train()
    for batch in loader:
        data_ori, data_aug, label = batch[0] + [batch[1],]
        data_ori, data_aug, label = data_ori.cuda(), data_aug.cuda(), label.cuda()

        steps += 1

        # split data according to prob
        n_ori = int(data_ori.size(0) * args.prob)
        data = torch.cat([data_ori[:n_ori], data_aug[n_ori:]], dim=0)

        preds = model(data)

        # mixture model posterior prob
        if gmm is not None:
            prob = torch.ones_like(label, dtype=torch.float)
            prob[n_ori:] = torch.tensor(gmm.predict(data[n_ori:], label[n_ori:]), device=label.device) if len(data[n_ori:]) else prob[n_ori:]
            index_OOD = prob < 0.5
            index_ID = ~index_OOD
            if args.weighting == 'regular':
                pass
            elif args.weighting == 'reverse':
                prob = 1 - prob
            elif args.weighting == 'random':
                idx = torch.randperm(prob.nelement())
                prob = prob.view(-1)[idx].view(prob.size())
            elif args.weighting == 'uniform':
                prob = torch.rand_like(prob)
            else:
                raise ValueError
        else:
            prob = None
            index_OOD = None
            index_ID = None

        # distillation
        if teacher is not None:
            teacher.eval()
            with torch.no_grad():
                preds_teacher = teacher(data)
        else:
            preds_teacher = None

        if statistics:
            loss_org, stats_org = get_loss(criterion_org, preds[:n_ori], label[:n_ori],
                                           prob[:n_ori] if prob is not None else None,
                                           preds_teacher[:n_ori] if preds_teacher is not None else None,
                                           statistics=True)
            loss_aug, stats_aug = get_loss(criterion_aug, preds[n_ori:], label[n_ori:],
                                           prob[n_ori:] if prob is not None else None,
                                           preds_teacher[n_ori:] if preds_teacher is not None else None,
                                           statistics=True)

            if 'loss_ce' in stats_org:
                loss_list1.append(stats_org['loss_ce'])
            if 'smooth' in stats_org:
                smooth_list.append(stats_org['smooth'])
            if 'loss_ce' in stats_aug:
                loss_list2.append(stats_aug['loss_ce'])
            if 'smooth' in stats_aug:
                smooth_list.append(stats_aug['smooth'])

        else:
            loss_org = get_loss(criterion_org, preds[:n_ori], label[:n_ori],
                                           prob[:n_ori] if prob is not None else None,
                                           preds_teacher[:n_ori] if preds_teacher is not None else None,
                                           statistics=False)
            loss_aug = get_loss(criterion_aug, preds[n_ori:], label[n_ori:],
                                           prob[n_ori:] if prob is not None else None,
                                           preds_teacher[n_ori:] if preds_teacher is not None else None,
                                           statistics=False)
        loss = args.prob * loss_org + (1 - args.prob) * loss_aug

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        top1, top5 = accuracy(preds, label, (1, 5))
        if index_OOD is not None:
            top1_OOD, top5_OOD = accuracy(preds[index_OOD], label[index_OOD], (1, 5))
            top1_ID, top5_ID = accuracy(preds[index_ID], label[index_ID], (1, 5))
            metrics.add_dict({
                'top1_OOD': top1_OOD.item() * len(data),
                'top5_OOD': top5_OOD.item() * len(data),
                'top1_ID': top1_ID.item() * len(data),
                'top5_ID': top5_ID.item() * len(data),
            })

        metrics.add_dict({
            'loss': loss.item() * len(data),
            'loss_org': loss_org.item() * len(data),
            'loss_aug': loss_aug.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })

        if statistics:
            metrics.add_dict({i[:5]+'org_'+i[5:]:j.mean().item() * len(data) for i,j in stats_org.items() if i.startswith('loss_')})
            metrics.add_dict({i[:5]+'aug_'+i[5:]:j.mean().item() * len(data) for i,j in stats_aug.items() if i.startswith('loss_')})

        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader.set_postfix(postfix)

    scheduler.step()

    metrics /= cnt
    metrics.metrics['lr'] = optimizer.param_groups[0]['lr']

    if statistics:
        writer.add_scalar('train/1.train_acc', metrics['top1'], epoch)
        for i,j in metrics.metrics.items():
            if i.startswith('top1_'):
                writer.add_scalar('train/1.train_acc_'+i[5:], j, epoch)
        writer.add_scalar('train/2.train_loss', metrics['loss'], epoch)
        writer.add_scalar('train/3.train_loss_origin', metrics['loss_org'], epoch)
        writer.add_scalar('train/4.train_loss_augmentation', metrics['loss_aug'], epoch)
        writer.add_image('train/1.origin_image', denormalization(data_ori[0], args.dataset), epoch)
        writer.add_image('train/2.augmented_image', denormalization(data_aug[0], args.dataset), epoch)
        for i,j in metrics.metrics.items():
            if i.startswith('loss_org_'):
                writer.add_scalar('train/3.train_loss_origin'+i[8:], j, epoch)
            elif i.startswith('loss_aug_'):
                writer.add_scalar('train/4.train_loss_augmentation'+i[8:], j, epoch)

        fig, ax = plt.subplots(figsize=(10, 6.2))
        if len(loss_list1):
            loss_list1 = torch.log(torch.cat(loss_list1, dim=0)).tolist()
            ax.hist(loss_list1, bins=1000, density=True, alpha=0.3, range=(-14, 4), histtype='stepfilled')
        if len(loss_list2):
            loss_list2 = torch.log(torch.cat(loss_list2, dim=0)).tolist()
            ax.hist(loss_list2, bins=1000, density=True, alpha=0.3, range=(-14, 4), histtype='stepfilled')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        writer.add_figure('train/3.loss_histogram', fig, epoch)
        if len(smooth_list):
            smooth_list = torch.cat(smooth_list, dim=0)
            writer.add_histogram('train/4.smoothness', smooth_list, epoch)

    return metrics


def test(model, loader, epoch=0, writer=None, verbose=1, prefix='test'):
    if verbose:
        loader = tqdm(loader)
        loader.set_description('[' + prefix +' %04d]' % epoch)

    metrics = Accumulator()
    cnt = 0
    steps = 0
    model.eval()
    with torch.no_grad():
        for data, label in loader:
            steps += 1
            data, label = data.cuda(), label.cuda()

            preds = model(data)
            loss = F.cross_entropy(preds, label)

            top1, top5 = accuracy(preds, label, (1, 5))
            metrics.add_dict({
                'loss': loss.item() * len(data),
                'top1': top1.item() * len(data),
                'top5': top5.item() * len(data),
            })
            cnt += len(data)
            if verbose:
                postfix = metrics / cnt
                loader.set_postfix(postfix)

            del preds, loss, top1, top5, data, label

    metrics /= cnt
    if writer is not None:
        writer.add_scalar(prefix + '/1.test_acc', metrics['top1'], epoch)
        writer.add_scalar(prefix + '/2.test_loss', metrics['loss'], epoch)
    return metrics


def train_and_eval(metric='last'):
    trainloader, testloader = get_dataloaders(args)

    # create a model & an optimizer
    model = get_model(args.model, num_class(args.dataset))

    criterion_org, criterion_aug = get_criterion(args)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov
    )

    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.)
    else:
        scheduler = None
    if args.warm:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=args.warm_multiplier,
            total_epoch=args.warm_epoch,
            after_scheduler=scheduler
        )

    if not args.tag:
        from RandAugment.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided, no tensorboard log.')
    else:
        from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir='./logs/%s' % args.tag)

    result = OrderedDict()
    epoch_start = 1
    if args.save and os.path.exists(args.save):
        logger.info('%s file found. loading...' % args.save)
        data = torch.load(args.save)
        if 'model' in data or 'state_dict' in data:
            key = 'model' if 'model' in data else 'state_dict'
            logger.info('checkpoint epoch@%d' % data['epoch'])
            if not isinstance(model, DataParallel):
                model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
            else:
                model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
            optimizer.load_state_dict(data['optimizer'])
            if data['epoch'] < args.epochs:
                epoch_start = data['epoch']
            else:
                args.only_eval = True
        else:
            model.load_state_dict({k: v for k, v in data.items()})
        del data
    else:
        logger.info('"%s" file not found. skip to pretrain weights...' % args.save)
        if args.only_eval:
            logger.warning('model checkpoint not found. only-evaluation mode is off.')
        args.only_eval = False

    if args.only_eval:
        logger.info('evaluation only+')
        rs = dict()
        if isinstance(testloader, dict):
            rs['test'] = test(model, testloader['test'], epoch=0, prefix='test')
            rs['DAOOD'] = test(model, testloader['DAOOD'], epoch=0, prefix='DAOOD')
            rs['DAID'] = test(model, testloader['DAID'], epoch=0, prefix='DAID')
            setnames = ['test', 'DAOOD', 'DAID']
        else:
            rs['test'] = test(model, testloader, epoch=0, writer=writer, prefix='test')
            setnames = ['test']
        for key, setname in itertools.product(['loss', 'top1', 'top5'], setnames):
            if setname not in rs:
                continue
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    if args.gmm:
        gmm = get_gmm(args, trainloader, args.pretrained)
    else:
        gmm = None

    if args.loss == 'Distillation':
        teacher = get_teacher(args, args.pretrained)
    else:
        teacher = None

    best_top1 = 0
    for epoch in range(epoch_start, args.epochs + 1):
        rs = dict()
        if args.fair_comparison in ['DAOOD', 'DAID', 'mix']:
            rs['train'] = fair_comparison(model, trainloader, optimizer, epoch, criterion_aug, scheduler=scheduler,
                                          gmm=gmm, writer=writer, verbose=True)
        elif args.fair_comparison == 'disable':
            rs['train'] = train(model, trainloader, optimizer, epoch, criterion_org, criterion_aug, scheduler=scheduler,
                                gmm=gmm, teacher=teacher, writer=writer, statistics=True, verbose=True)
        else:
            raise ValueError
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % 1 == 0 or epoch == args.epochs:
            if isinstance(testloader, dict):
                rs['test'] = test(model, testloader['test'], epoch=epoch, writer=writer, verbose=True, prefix='test')
                rs['DAOOD'] = test(model, testloader['DAOOD'], epoch=epoch, writer=writer, verbose=True, prefix='DAOOD')
                rs['DAID'] = test(model, testloader['DAID'], epoch=epoch, writer=writer, verbose=True, prefix='DAID')
                setnames = ['train', 'test', 'DAOOD', 'DAID']
            else:
                rs['test'] = test(model, testloader, epoch=epoch, writer=writer, verbose=True, prefix='test')
                setnames = ['train', 'test']
            if metric == 'last' or rs[metric]['top1'] > best_top1:
                if metric != 'last':
                    best_top1 = rs[metric]['top1']
                    logger.info('save model@%d with acc %.4f' % (epoch, best_top1))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, args.save.replace('model.pth', 'best.pth'))
                for key, setname in itertools.product(['loss', 'top1', 'top5'], setnames):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                # save checkpoint
                if args.save:
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict()
                    }, args.save)

    del model

    result['top1_test'] = best_top1
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')

    parser.add_argument('--dataroot', type=str, default='data', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='', help='ckpt save dir')
    parser.add_argument('--tag', type=str, default='', help='log dir')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model dir')
    parser.add_argument('--gpu', default=1, type=int, help='gpu id to use')

    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 [default] or cifar100)')
    parser.add_argument('--aug', type=str, default='ra')
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--cutout', type=int, default=0)
    parser.add_argument('--prob', type=float, default=0.2)

    parser.add_argument('--model', type=str, default='resnet18')

    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--cosine', default=True, type=bool)
    parser.add_argument('--warm', default=True, type=bool)
    parser.add_argument('--warm-multiplier', default=2, type=int)
    parser.add_argument('--warm-epoch', default=5, type=int)
    parser.add_argument('--batch-size', default=256, type=int, help='mini-batch size')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--nesterov', default=True, type=bool)
    parser.add_argument('--only-eval', action='store_true')
    parser.add_argument('--evalDA', action='store_true')
    data_component = ['mix', 'DAOOD', 'DAID', 'disable']
    parser.add_argument('--fair-comparison', choices=data_component, default='disable')

    # gmm parameter
    parser.add_argument('--gmm', action='store_true', help='the gmm option')
    weighting = ['regular', 'reverse', 'random', 'uniform']
    parser.add_argument('--weighting', choices=weighting, default='regular')

    # loss function
    loss_name = ['LabelSmooth', 'SampleSmooth', 'Distillation']
    parser.add_argument('--loss', choices=loss_name, default='LabelSmooth', help='loss function')

    # smooth parameter (for LabelSmooth and SampleSmooth)
    parser.add_argument('--smooth-org', default=0.0, type=float, help='smooth for original image')
    parser.add_argument('--smooth-aug', default=0.0, type=float, help='smooth for augmentation')

    # distillation parameter (for Distillation)
    parser.add_argument('--K', type=int, default=-1, help='top K classes to distill')
    parser.add_argument('--lamda', type=float, default=0.5, help='parameter for distillation term')
    parser.add_argument('--T', type=float, default=5, help='temperature parameter for the CE loss')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'
    if args.tag == '':
        args.tag = 'cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-randaugment")
    elif os.path.exists('logs/' + args.tag) and not args.only_eval:
        shutil.rmtree('logs/' + args.tag)
    os.makedirs('logs/' + args.tag, exist_ok=True)
    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            args.save = './logs/%s/model.pth' % args.tag
            logger.info('checkpoint will be saved at %s' % args.save)
    if args.save and not args.only_eval:
        add_filehandler(logger, args.save.replace('.pth', '') + '.log')

    logger.info(args)

    t = time.time()
    result = train_and_eval(metric='test')
    elapsed = time.time() - t

    logger.info('done.')
    logger.info('model: %s' % args.model)
    logger.info('augmentation: %s' % args.aug)
    logger.info('\n' + json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info(args.save)
