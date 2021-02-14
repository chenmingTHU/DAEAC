import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor((alpha, 1 - alpha))
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, x, target):

        if x.dim() > 2:
            x = x.view(x.size(0), x.size(1), -1)
            x = x.transpose(1, 2)
            x = x.contiguous().view(-1, x.size(2))

        target = target.view(-1, 1)

        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.gather(1, index=target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != x.data.type():
                self.alpha = self.alpha.type_as(x.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class ClassBalanceLoss(nn.Module):

    def __init__(self, beta, n, v, s, f):
        super(ClassBalanceLoss, self).__init__()

        self.beta = beta
        self.num_n = n
        self.num_v = v
        self.num_s = s
        self.num_f = f

        sum_ = (1 - self.beta) / (1 - self.beta ** self.num_n) + (1 - self.beta) / (1 - self.beta ** self.num_v) \
               + (1 - self.beta) / (1 - self.beta ** self.num_s) + (1 - self.beta) / (1 - self.beta ** self.num_f)
        print("The Class-balance weights are: {}, {}, {}, {}".format(
            (1 - self.beta) / (1 - self.beta ** self.num_n) / sum_,
            (1 - self.beta) / (1 - self.beta ** self.num_v) / sum_,
            (1 - self.beta) / (1 - self.beta ** self.num_s) / sum_,
            (1 - self.beta) / (1 - self.beta ** self.num_f) / sum_
        ))

    def forward(self, x, target, weight=None):

        '''
        :param x: the predictions of models
        :param labels: annotations: 0-3
        0: N, 1: V, 2: S, 3: F
        :return: the losses
        '''

        target = target.view(-1, 1)

        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.gather(1, index=target)
        logpt = logpt.view(-1)

        target_cpu = target.detach().cpu()

        n = torch.ones(target_cpu.size()) * self.num_n
        v = torch.ones(target_cpu.size()) * self.num_v
        s = torch.ones(target_cpu.size()) * self.num_s
        f = torch.ones(target_cpu.size()) * self.num_f

        ns = torch.where(target_cpu == 0, n, torch.zeros(target.size()))
        vs = torch.where(target_cpu == 1, v, torch.zeros(target.size()))
        ss = torch.where(target_cpu == 2, s, torch.zeros(target.size()))
        fs = torch.where(target_cpu == 3, f, torch.zeros(target.size()))

        w = ns + vs + ss + fs
        w = (1 - self.beta) / (1 - self.beta ** w)
        sum_ = (1 - self.beta) / (1 - self.beta ** self.num_n) + (1 - self.beta) / (1 - self.beta ** self.num_v) \
               + (1 - self.beta) / (1 - self.beta ** self.num_s) + (1 - self.beta) / (1 - self.beta ** self.num_f)
        w = w / sum_
        w = w.cuda()

        loss = -1.0 * w * logpt

        return loss.mean()


class DynamicLoss(nn.Module):

    def __init__(self, n, v, s, f, beta=100):
        super(DynamicLoss, self).__init__()

        self.num_n = math.exp(-n / beta)
        self.num_v = math.exp(-v / beta)
        self.num_s = math.exp(-s / beta)
        self.num_f = math.exp(-f / beta)

    def forward(self, x, target):

        '''
        :param x: the predictions of models
        :param labels: annotations: 0-3
        0: N, 1: V, 2: S, 3: F
        :return: the losses
        '''

        target = target.view(-1, 1)

        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.gather(1, index=target)
        logpt = logpt.view(-1)

        wd = -1.0 * logpt.detach()

        target_cpu = target.detach().cpu()

        n = torch.ones(target_cpu.size()) * self.num_n
        v = torch.ones(target_cpu.size()) * self.num_v
        s = torch.ones(target_cpu.size()) * self.num_s
        f = torch.ones(target_cpu.size()) * self.num_f

        ns = torch.where(target_cpu == 0, n, torch.zeros(target.size()))
        vs = torch.where(target_cpu == 1, v, torch.zeros(target.size()))
        ss = torch.where(target_cpu == 2, s, torch.zeros(target.size()))
        fs = torch.where(target_cpu == 3, f, torch.zeros(target.size()))

        w = ns + vs + ss + fs
        sum_ = self.num_n + self.num_s + self.num_v + self.num_f
        w = w / sum_
        w = w.cuda()

        loss = -1.0 * w * wd * logpt

        return loss.mean()


class ExpWeightedLoss(nn.Module):

    def __init__(self, n, v, s, f, beta=100, T=1):
        super(ExpWeightedLoss, self).__init__()

        self.num_n = math.exp(-n / beta)
        self.num_v = math.exp(-v / beta)
        self.num_s = math.exp(-s / beta)
        self.num_f = math.exp(-f / beta)
        self.T = T

        sum_ = self.num_n + self.num_v + self.num_s + self.num_f
        print("The Exponential weights are: {}, {}, {}, {}".format(self.num_n / sum_, self.num_v / sum_,
                                                       self.num_s / sum_, self.num_f / sum_))

    def forward(self, x, target):

        '''
        :param x: the predictions of models
        :param labels: annotations: 0-3
        0: N, 1: V, 2: S, 3: F
        :return: the losses
        '''

        target = target.view(-1, 1)

        logpt = F.log_softmax(x / self.T, dim=1)
        logpt = logpt.gather(1, index=target)
        logpt = logpt.view(-1)

        target_cpu = target.detach().cpu()

        n = torch.ones(target_cpu.size()) * self.num_n
        v = torch.ones(target_cpu.size()) * self.num_v
        s = torch.ones(target_cpu.size()) * self.num_s
        f = torch.ones(target_cpu.size()) * self.num_f

        ns = torch.where(target_cpu == 0, n, torch.zeros(target.size()))
        vs = torch.where(target_cpu == 1, v, torch.zeros(target.size()))
        ss = torch.where(target_cpu == 2, s, torch.zeros(target.size()))
        fs = torch.where(target_cpu == 3, f, torch.zeros(target.size()))

        w = ns + vs + ss + fs
        sum_ = self.num_n + self.num_s + self.num_v + self.num_f
        w = w / sum_
        w = w.cuda()

        loss = -1.0 * w * logpt

        return loss.mean()


class WeightedLoss(nn.Module):

    def __init__(self, n, v, s, f):
        super(WeightedLoss, self).__init__()

        self.num_n = n
        self.num_v = v
        self.num_s = s
        self.num_f = f

    def forward(self, x, target):

        '''
        :param x: the predictions of models
        :param labels: annotations: 0-3
        0: N, 1: V, 2: S, 3: F
        :return: the losses
        '''

        target = target.view(-1, 1)

        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.gather(1, index=target)
        logpt = logpt.view(-1)

        target_cpu = target.detach().cpu()

        n = torch.ones(target_cpu.size()) * (1 / self.num_n)
        v = torch.ones(target_cpu.size()) * (1 / self.num_v)
        s = torch.ones(target_cpu.size()) * (1 / self.num_s)
        f = torch.ones(target_cpu.size()) * (1 / self.num_f)

        ns = torch.where(target_cpu == 0, n, torch.zeros(target.size()))
        vs = torch.where(target_cpu == 1, v, torch.zeros(target.size()))
        ss = torch.where(target_cpu == 2, s, torch.zeros(target.size()))
        fs = torch.where(target_cpu == 3, f, torch.zeros(target.size()))

        w = ns + vs + ss + fs
        sum_ = 1 / self.num_n + 1 / self.num_s + 1 / self.num_v + 1 / self.num_f
        w = w / sum_
        w = w.cuda()

        loss = -1.0 * w * logpt

        return loss.mean()
    
    
class BatchWeightedLoss(nn.Module):
    
    def __init__(self, beta):
        super(BatchWeightedLoss, self).__init__()
        self.eps = 1e-6
        self.beta = beta

    def forward(self, x, target):
        target = target.view(-1, 1)

        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.gather(1, index=target)
        logpt = logpt.view(-1)

        target_cpu = target.detach().cpu()

        zeros = torch.zeros(target_cpu.size())
        ones = torch.ones(target_cpu.size())
        num_n = torch.sum(torch.where(target_cpu == 0, ones, zeros)).item()
        num_v = torch.sum(torch.where(target_cpu == 1, ones, zeros)).item()
        num_s = torch.sum(torch.where(target_cpu == 2, ones, zeros)).item()
        num_f = torch.sum(torch.where(target_cpu == 3, ones, zeros)).item()

        n = torch.ones(target_cpu.size()) * num_n
        v = torch.ones(target_cpu.size()) * num_v
        s = torch.ones(target_cpu.size()) * num_s
        f = torch.ones(target_cpu.size()) * num_f

        ns = torch.where(target_cpu == 0, n, zeros)
        vs = torch.where(target_cpu == 1, v, zeros)
        ss = torch.where(target_cpu == 2, s, zeros)
        fs = torch.where(target_cpu == 3, f, zeros)

        w = ns + vs + ss + fs
        w = (1 - self.beta) / (1 - self.beta ** w + self.eps)
        sum_ = (1 - self.beta) / (1 - self.beta ** num_n + self.eps) \
               + (1 - self.beta) / (1 - self.beta ** num_v + self.eps) \
               + (1 - self.beta) / (1 - self.beta ** num_s + self.eps) \
               + (1 - self.beta) / (1 - self.beta ** num_f + self.eps)
        w = w / sum_
        w = w.cuda()

        loss = -1.0 * w * logpt

        return loss.mean()
    

class BinClassBalanceLoss(nn.Module):

    def __init__(self, beta, n, vs):
        super(BinClassBalanceLoss, self).__init__()

        self.beta = beta
        self.num_n = n
        self.num_vs = vs

    def forward(self, x, target):

        '''
        :param x: the predictions of models
        :param labels: annotations: 0, 1
        0: N, 1: V, S, F
        :return: the losses
        '''

        target = target.view(-1, 1)

        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.gather(1, index=target)
        logpt = logpt.view(-1)

        target_cpu = target.detach().cpu()

        n = torch.ones(target_cpu.size()) * self.num_n
        vs = torch.ones(target_cpu.size()) * self.num_vs

        ns = torch.where(target_cpu == 0, n, torch.zeros(target.size()))
        vss = torch.where(target_cpu != 0, vs, torch.zeros(target.size()))

        w = ns + vss
        w = (1 - self.beta) / (1 - self.beta ** w)
        sum_ = (1 - self.beta) / (1 - self.beta ** self.num_n) + (1 - self.beta) / (1 - self.beta ** self.num_vs)

        w = w / sum_
        w = w.cuda()

        loss = -1.0 * w * logpt

        return loss.mean()


class ClassBalanceFocalLoss(nn.Module):

    def __init__(self, beta, n, v, s, f, gamma, alpha=None, size_average=True):
        super(ClassBalanceFocalLoss, self).__init__()

        self.beta = beta
        self.num_n = n
        self.num_v = v
        self.num_s = s
        self.num_f = f

        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor((alpha, 1 - alpha))
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, x, target):

        if x.dim() > 2:
            x = x.view(x.size(0), x.size(1), -1)
            x = x.transpose(1, 2)
            x = x.contiguous().view(-1, x.size(2))

        target = target.view(-1, 1)

        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.gather(1, index=target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        target_cpu = target.detach().cpu()

        n = torch.ones(target_cpu.size()) * self.num_n
        v = torch.ones(target_cpu.size()) * self.num_v
        s = torch.ones(target_cpu.size()) * self.num_s
        f = torch.ones(target_cpu.size()) * self.num_f

        ns = torch.where(target_cpu == 0, n, torch.zeros(target.size()))
        vs = torch.where(target_cpu == 1, v, torch.zeros(target.size()))
        ss = torch.where(target_cpu == 2, s, torch.zeros(target.size()))
        fs = torch.where(target_cpu == 3, f, torch.zeros(target.size()))

        w = ns + vs + ss + fs
        w = (1 - self.beta) / (1 - self.beta ** w)
        sum_ = (1 - self.beta) / (1 - self.beta ** self.num_n) + (1 - self.beta) / (1 - self.beta ** self.num_v) \
               + (1 - self.beta) / (1 - self.beta ** self.num_s) + (1 - self.beta) / (1 - self.beta ** self.num_f)
        w = w / sum_
        w = w.cuda()

        if self.alpha is not None:
            if self.alpha.type() != x.data.type():
                self.alpha = self.alpha.type_as(x.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * w * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, target, T=1):
        '''
        :param x: The logits output of Discriminator which has the shape of (N, 2)
        :param target: The labels of real input or generated input, which has the shape of (N,)
        :return:
        '''
        x_ = x / T
        loss = self.criterion(x_, target)

        return loss
    

class ConsistencyLoss(nn.Module):
    
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

        self.criterion = nn.MSELoss()

    def forward(self, logit1, logit2):

        loss = self.criterion(logit1, logit2)
        return loss


class CosineLoss(nn.Module):

    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, x, y, dim=0, if_mean=True):

        cosine_similarity = torch.cosine_similarity(x, y, dim=dim)

        if if_mean:
            loss = 1 - cosine_similarity.mean()
        else:
            loss = 1 - cosine_similarity
        return loss


class L2Distance(nn.Module):

    def __init__(self):
        super(L2Distance, self).__init__()

    def forward(self, x, y, dim=0, if_mean=True):

        if if_mean:
            distance = torch.mean(torch.sqrt(torch.sum((x - y) ** 2, dim=dim)))
        else:
            distance = torch.sqrt(torch.sum((x - y) ** 2, dim=dim))

        return distance


class L1Distance(nn.Module):

    def __init__(self):
        super(L1Distance, self).__init__()

    def forward(self, x, y, dim=0, if_mean=True):

        if if_mean:
            distance = torch.mean(torch.sum(torch.abs(x - y), dim=dim))
        else:
            distance = torch.sum(torch.abs(x - y), dim=dim)

        return distance


def get_L2norm_loss_self_driven(x, weight_L2norm):

    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 0.3
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return weight_L2norm * l


def get_L2norm_loss_self_driven_hard(x, radius, weight_L2norm):
    l = (x.norm(p=2, dim=1).mean() - radius) ** 2
    return weight_L2norm * l


def get_entropy_loss(p_softmax, weight_entropy):
    mask = p_softmax.ge(0.000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return weight_entropy * (entropy / float(p_softmax.size(0)))

