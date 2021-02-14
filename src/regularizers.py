import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


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


def coral(feat_s_l, feat_t_l):

    m = feat_s_l.size(1)

    n_s = feat_s_l.size(0)
    n_t = feat_t_l.size(0)

    Jb_s = torch.eye(n_s) - 1 / n_s * torch.ones(n_s, n_s)
    Jb_t = torch.eye(n_t) - 1 / n_t * torch.ones(n_t, n_t)

    Jb_s = Jb_s.cuda()
    Jb_t = Jb_t.cuda()

    feat_s_l = feat_s_l.view(-1, m)
    feat_t_l = feat_t_l.view(-1, m)

    Conv_s = torch.mm(torch.mm(feat_s_l.t(), Jb_s), feat_s_l)
    Conv_t = torch.mm(torch.mm(feat_t_l.t(), Jb_t), feat_t_l)

    return torch.norm(Conv_s - Conv_t) / m / m



