import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, ksize=3, dilation=1, stride=1, groups=1, use_bn=True):
        super(BasicConvBlock, self).__init__()

        padding = int((in_channel * (stride - 1) + dilation * (ksize - 1) + 1 - stride) / 2)
        self.use_bn = use_bn

        self.conv = nn.Conv1d(in_channel, out_channel, ksize, stride, padding, dilation, groups)
        self.bn = nn.BatchNorm1d(out_channel)
        self.rl = nn.ReLU()

    def forward(self, x):

        net = self.conv(x)
        if self.use_bn:
            net = self.bn(net)
        net = self.rl(net)

        return net


class conv_layer_block(nn.Module):

    def __init__(self, in_channel, ksize=3, dilation=1, stride=1, groups=1, double_out_channel=True):
        super(conv_layer_block, self).__init__()

        if double_out_channel:
            out_channel = 2 * in_channel
        else:
            out_channel = in_channel

        self.conv1 = nn.Conv1d(in_channel, out_channel, ksize, stride, int((ksize - 1) / 2), dilation, groups)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.rl1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channel, out_channel, ksize, stride, int((ksize - 1) / 2), dilation, groups)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.rl2 = nn.ReLU()

        self.conv3 = nn.Conv1d(out_channel, out_channel, ksize, stride, int((ksize - 1) / 2), dilation, groups)
        self.bn3 = nn.BatchNorm1d(out_channel)
        self.rl3 = nn.ReLU()

    def forward(self, x):

        net = self.rl1(self.bn1(self.conv1(x)))
        net = self.rl2(self.bn2(self.conv2(net)))
        net = self.rl3(self.bn3(self.conv3(net)))

        return net


class BasicCNN(nn.Module):
    
    def __init__(self):
        super(BasicCNN, self).__init__()

        self.stem = nn.Sequential(
            BasicConvBlock(1, 4),
            BasicConvBlock(4, 8),
            BasicConvBlock(8, 8)
        )

        self.layer_1 = nn.Sequential(
            conv_layer_block(8),
            conv_layer_block(16, double_out_channel=False),
            nn.AvgPool1d(kernel_size=2)
        )

        self.layer_2 = nn.Sequential(
            conv_layer_block(16),
            conv_layer_block(32, double_out_channel=False),
            nn.AvgPool1d(kernel_size=2)
        )

        self.layer_3 = nn.Sequential(
            conv_layer_block(32),
            conv_layer_block(64, double_out_channel=False),
            nn.AvgPool1d(kernel_size=2)
        )

        self.layer_4 = nn.Sequential(
            conv_layer_block(64),
            conv_layer_block(128, double_out_channel=False),
            nn.AvgPool1d(kernel_size=2)
        )

        self.head = nn.Sequential(
            nn.Linear(128 * 25, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 4),
        )

    def forward(self, x):

        net = self.stem(x)

        layer1 = self.layer_1(net)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        layer4 = self.layer_4(layer3)

        net = layer4.view(-1, 128 * 25)
        logits = self.head(net)

        return logits
