import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


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


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class CNN_BiLSTM(nn.Module):

    def __init__(self):
        super(CNN_BiLSTM, self).__init__()

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

        self.lstm_layer = lstm(128, 128, 2)

        self.BN_lstm = nn.BatchNorm1d(128 * 25 * 2)

        self.head = nn.Sequential(
            nn.Linear(128 * 25 * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.fc = nn.Linear(128, 4)

    def forward(self, x):
        net = self.stem(x)
        layer1 = self.layer_1(net)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        layer4 = self.layer_4(layer3)

        layer4_td = layer4.permute(0, 2, 1)
        lstm_out, hidden_state = self.lstm_layer(layer4_td)

        net = lstm_out
        net = net.reshape(-1, 25 * 128 * 2)
        net = self.BN_lstm(net)

        net = self.head(net)
        logits = self.fc(net)

        # return layer4, logits
        return net, logits


