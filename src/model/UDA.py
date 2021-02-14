import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class PEF(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=10, output_dim=2):
        super(PEF, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        hidden_layer = self.fc1(x)
        hidden_layer = self.relu1(hidden_layer)

        logits = self.fc2(hidden_layer)

        return logits


class BasicConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, ksize=7, dilation=1, stride=1, groups=1, use_bn=True):
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


class lstm(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_layer, batch_first=True, bidirectional=True):
        super(lstm, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.lstm_layer = nn.LSTM(input_size=in_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=n_layer,
                                  batch_first=batch_first,
                                  bidirectional=bidirectional)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)

        batch_size = x.size()[0]
        hidden_0 = self.init_hidden(batch_size)

        output, hidden = self.lstm_layer(x, hidden_0)

        return output, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM

        number = self.n_layer
        direction = 2 if self.bidirectional else 1

        hidden = (torch.from_numpy(np.zeros(shape=(direction * number, batch_size, self.hidden_dim))).cuda(),
                  torch.from_numpy(np.zeros(shape=(direction * number, batch_size, self.hidden_dim))).cuda())

        return hidden


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, features):

        logits = self.head(features)

        return logits


class UniDiscriminator(nn.Module):
    
    def __init__(self):
        super(UniDiscriminator, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 2)
        )

    def forward(self, x):

        logits = self.head(x)

        return logits


class Residual_Block(nn.Module):

    def __init__(self, in_channel, ksize=3, dilation=1,
                 stride=1, groups=1, double_out_channel=True):
        super(Residual_Block, self).__init__()

        self.double_channel = double_out_channel

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

        self.conv = nn.Conv1d(in_channel, out_channel, 1, 1, 0, 1, 1)

    def forward(self, x):

        net = self.rl1(self.bn1(self.conv1(x)))
        net = self.rl2(self.bn2(self.conv2(net)))
        net = self.rl3(self.bn3(self.conv3(net)))

        if self.double_channel:
            sp = self.conv(x)
        else:
            sp = x

        net = net + sp

        return net


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.bottleneck = BasicConvBlock(9, 8, ksize=3)

        self.layer_1 = nn.Sequential(
            Residual_Block(8),
            Residual_Block(16, double_out_channel=False),
            nn.AvgPool1d(kernel_size=2)
        )

        self.layer_2 = nn.Sequential(
            Residual_Block(16),
            Residual_Block(32, double_out_channel=False),
            nn.AvgPool1d(kernel_size=2)
        )

        self.layer_3 = nn.Sequential(
            Residual_Block(32),
            Residual_Block(64, double_out_channel=False),
            nn.AvgPool1d(kernel_size=2)
        )

        self.layer_4 = nn.Sequential(
            Residual_Block(64),
            Residual_Block(128, double_out_channel=False),
            nn.AvgPool1d(kernel_size=2)
        )

    def forward(self, features, noise):
        x = torch.cat([features, noise], dim=1)
        x = self.bottleneck(x)
        layer1 = self.layer_1(x)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        layer4 = self.layer_4(layer3)

        return layer4


