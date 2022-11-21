import torch
import torch.nn.functional as F


class LabelSmoothing(torch.nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target, statistics=False):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if statistics:
            return loss.mean(), {'loss_ce': nll_loss.detach(), 'loss_uniform': smooth_loss.detach()}
        else:
            return loss.mean()


class SampleLabelSmoothing(LabelSmoothing):
    """
    Sample-wise NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__(smoothing)

    def forward(self, x, target, prob=None, statistics=False):
        smoothing = self.smoothing
        self.smoothing = (1 - prob) * self.smoothing
        if statistics:
            loss, stats = super().forward(x, target, True)
            stats['smooth'] = self.smoothing
            self.smoothing = smoothing
            return loss, stats
        else:
            loss = super().forward(x, target, False)
            self.smoothing = smoothing
            return loss


class DistillationLoss(torch.nn.Module):
    """
    loss used for knowledge distillation with topK prob.
    """
    def __init__(self, K=-1, lamda=0.5, T=5):
        """
        Constructor for the LabelSmoothing module.
        :param K: topK prob for distillation
        :param lamda: weight for distillation
        :param T: temperature for softmax
        """
        super().__init__()
        self.K = K
        self.lamda = lamda
        self.T = T

        self.CE =  torch.nn.CrossEntropyLoss(reduction='none')
        self.KL = torch.nn.KLDivLoss(reduction='none')
        self.LogSoftmax = torch.nn.LogSoftmax(dim=1)
        self.Softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, target, x_teacher, statistics=False):
        assert len(x.size()) == 2
        assert len(x_teacher.size()) == 2
        assert x_teacher.requires_grad is False
        x, x_teacher = x / self.T, x_teacher / self.T
        loss_ce = self.CE(x, target)
        assert loss_ce.size(0) == x.size(0)
        if self.K == -1:
            self.K = x_teacher.size(1)
        x_teacher, index = torch.topk(x_teacher, k=self.K, dim=1)
        x = torch.gather(x, dim=1, index=index)
        loss_kl = self.T**2 * torch.sum(self.KL(self.LogSoftmax(x_teacher), self.Softmax(x)), dim=1)
        loss = loss_ce + self.lamda * loss_kl
        if statistics:
            return loss.mean(), {'loss_ce': loss_ce.detach(), 'loss_kl': loss_kl.detach()}
        else:
            return loss.mean()


def get_criterion(args):
    if args.loss == 'LabelSmooth':
        return LabelSmoothing(smoothing=args.smooth_org).cuda(), LabelSmoothing(smoothing=args.smooth_aug).cuda()
    elif args.loss == 'SampleSmooth':
        if args.weighting == 'regular':
            return LabelSmoothing(smoothing=args.smooth_org).cuda(), SampleLabelSmoothing(
                smoothing=args.smooth_aug).cuda()
        elif args.weighting == 'reverse':
            return LabelSmoothing(smoothing=args.smooth_aug).cuda(), SampleLabelSmoothing(
                smoothing=args.smooth_aug).cuda()
        elif args.weighting in ['random', 'uniform']:
            return SampleLabelSmoothing(smoothing=args.smooth_aug).cuda(), SampleLabelSmoothing(
                smoothing=args.smooth_aug).cuda()
        else:
            raise ValueError
    elif args.loss == 'Distillation':
        return LabelSmoothing(smoothing=args.smooth_org).cuda(), DistillationLoss(
            K=args.K, lamda=args.lamda, T=args.T).cuda()
    else:
        raise ValueError


def get_loss(loss_fn, preds, label, prob=None, preds_teacher=None, statistics=False):
    if preds.size(0) == 0:
        if statistics:
            return torch.tensor(0., device='cuda'), {}
        else:
            return torch.tensor(0., device='cuda')

    if isinstance(loss_fn, SampleLabelSmoothing):
        return loss_fn(preds, label, prob, statistics)
    elif isinstance(loss_fn, LabelSmoothing):
        return loss_fn(preds, label, statistics)
    elif isinstance(loss_fn, DistillationLoss):
        return loss_fn(preds, label, preds_teacher, statistics)
    else:
        raise ValueError
