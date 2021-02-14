import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, ksize=17, dilation=1, stride=1, groups=1):
        super(conv_layer, self).__init__()

        padding = int((in_channel * (stride - 1) + dilation * (ksize - 1) + 1 - stride) / 2)

        self.conv = nn.Conv1d(in_channel, out_channel, ksize, stride, padding, dilation, groups)

    def forward(self, x):
        net = self.conv(x)
        return net


class conv_stem(nn.Module):

    def __init__(self, in_channel, ksize=17, dilation=1, stride=1, groups=1):
        super(conv_stem, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channel)
        self.tanh = nn.Tanh()

        self.conv_layer_1 = conv_layer(in_channel, 64, ksize, dilation, stride, groups)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.75)

        self.conv_layer_2 = conv_layer(64, 64, ksize, dilation, stride, groups)

    def forward(self, x):
        net = self.bn1(x)
        net = self.tanh(net)

        net = self.conv_layer_1(net)
        net = self.bn2(net)
        net = self.relu1(net)
        net = self.dropout_1(net)

        net = self.conv_layer_2(net)

        return net


class residual_block(nn.Module):

    def __init__(self, in_channel, out_channel, ksize=17, dilation=1, stride=1, groups=1):
        super(residual_block, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channel)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.75)

        self.conv_layer_1 = conv_layer(in_channel, out_channel, ksize, dilation, stride, groups)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.75)

        self.conv_layer_2 = conv_layer(out_channel, out_channel, ksize, dilation, stride, groups)

    def forward(self, x):
        net = self.bn1(x)
        net = self.relu1(net)
        net = self.dropout1(net)

        net = self.conv_layer_1(net)
        net = self.bn2(net)
        net = self.relu2(net)
        net = self.dropout2(net)

        net = self.conv_layer_2(net) + x

        return net


class BaselineModel(nn.Module):

    def __init__(self):
        super(BaselineModel, self).__init__()

        self.conv_layer1 = conv_layer(1, 64)

        self.stem = conv_stem(64)

        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.bottle_neck1 = conv_layer(64, 128, ksize=1)
        self.residual_block_1 = residual_block(128, 128)

        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.bottle_neck2 = conv_layer(128, 256, ksize=1)
        self.residual_block_2 = residual_block(256, 256)

        self.pool3 = nn.AvgPool1d(kernel_size=2)
        self.bottle_neck3 = conv_layer(256, 512, ksize=1)
        self.residual_block_3 = residual_block(512, 512)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(512, 4)

    def forward(self, x):
        '''
        :param x: x should be a vector containing three successive heartbeats,
                  which has a length of 600
        :return:
        '''

        conv1 = self.conv_layer1(x)
        net = self.stem(conv1)

        net = net + conv1

        net = self.pool1(net)
        net = self.bottle_neck1(net)

        net = self.residual_block_1(net)

        net = self.pool2(net)
        net = self.bottle_neck2(net)

        net = self.residual_block_2(net)

        net = self.pool3(net)
        net = self.bottle_neck3(net)

        net = self.residual_block_3(net)

        waveform_feat = self.global_pool(net).view(-1, 512)

        logits = self.fc(waveform_feat)

        return waveform_feat, logits


