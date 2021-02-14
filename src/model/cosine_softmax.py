import torch
import torch.nn as nn
import torch.nn.functional as F


class Cosine_Softmax(nn.Module):

    def __init__(self, in_plane, class_num):
        super(Cosine_Softmax, self).__init__()

        self.in_plane = in_plane
        self.class_num = class_num

        self.fc = nn.Linear(in_plane, class_num, bias=False)

    def forward(self, x):

        p = self.fc(x)

        w_norm = torch.norm(self.fc.weight, p=2, dim=1)
        x_norm = torch.norm(x, p=2)

        p_ = p / w_norm / x_norm

        return p_
