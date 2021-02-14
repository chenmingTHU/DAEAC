import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# class MultiAttention(nn.Module):
#
#     def __init__(self, in_dim1, in_dim2):
#         super(MultiAttention, self).__init__()
#
#         self.linear1 = nn.Linear(in_dim1, 1)
#         self.linear2 = nn.Linear(in_dim2, 1)
#
#     def forward(self, waveform, pef):
#
#         '''
#         :param waveform: the waveform feature; Nx128
#         :param pef: the rrs feature; Nx2
#         :return:
#         '''
#
#         in_dim1 = waveform.size()[1]
#         in_dim2 = pef.size()[1]
#
#         w_1 = F.relu(self.linear1(waveform))
#         w_2 = F.relu(self.linear2(pef))
#
#         w = torch.cat([w_1, w_2], dim=1)
#         w = F.softmax(w, dim=1)
#
#         w_1, w_2 = torch.split(w, 1, dim=1)
#         w_1 = w_1.repeat(1, in_dim1)
#         w_2 = w_2.repeat(1, in_dim2)
#
#         waveform_ = waveform * w_1
#         pef_ = pef * w_2
#
#         return torch.cat([waveform_, pef_], dim=1)
#
#
# class TimeDistributed(nn.Module):
#     def __init__(self, module, batch_first=False):
#         super(TimeDistributed, self).__init__()
#         self.module = module
#         self.batch_first = batch_first
#
#     def forward(self, x):
#
#         if len(x.size()) <= 2:
#             return self.module(x)
#
#         # Squash samples and timesteps into a single axis
#         x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
#
#         y = self.module(x_reshape)
#
#         # We have to reshape Y
#         if self.batch_first:
#             y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
#         else:
#             y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
#
#         return y


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


class BiLSTM(nn.Module):

    def __init__(self, fixed_len=128):
        super(BiLSTM, self).__init__()

        self.fixed_len = fixed_len

        self.lstm_layer_1 = lstm(1, 50, 1)
        self.lstm_layer_2 = lstm(50 * 2, 150, 1)

        self.head = nn.Sequential(
            nn.Linear(2 * 150, 20),
            nn.LeakyReLU(0.3)
        )

        self.PEF_fc1 = nn.Sequential(
            nn.Linear(4, 10),
            nn.LeakyReLU(0.3)
        )

        self.PEF_fc2 = nn.Linear(10, 2)

        self.fc = nn.Linear(10 + 20, 4)

    def forward(self, x, rrs):

        x = x.permute(0, 2, 1)
        lstm_out_1, hidden_state_1 = self.lstm_layer_1(x)
        lstm_out_2, hidden_state_2 = self.lstm_layer_2(lstm_out_1)

        net = lstm_out_2[:, -1]
        # net = net.reshape(-1, self.fixed_len * 2 * 150)

        waveform_feat = self.head(net)
        pef = self.PEF_fc1(rrs)
        pef_logits = self.PEF_fc2(pef)

        net = torch.cat([waveform_feat, pef], dim=1)

        logits = self.fc(net)

        return waveform_feat, pef_logits, net, logits