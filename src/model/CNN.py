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

    def __init__(self, in_channel, ksize=17, dilation=1, stride=1, groups=1, p=1.0):
        super(conv_stem, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channel)
        self.tanh = nn.Tanh()

        self.conv_layer_1 = conv_layer(in_channel, 64, ksize, dilation, stride, groups)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=p)

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
    
    def __init__(self, in_channel, out_channel, ksize=17,
                 dilation=1, stride=1, groups=1, p=0.0):
        super(residual_block, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channel)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=p)

        self.conv_layer_1 = conv_layer(in_channel, out_channel, ksize, dilation, stride, groups)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=p)

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


class CNN(nn.Module):

    def __init__(self, fixed_len=200, p=0.0):
        super(CNN, self).__init__()

        self.fixed_len = fixed_len

        self.conv_layer1 = conv_layer(1, 64)

        self.stem = conv_stem(64, p=p)

        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.bottle_neck1 = conv_layer(64, 128, ksize=1)
        self.residual_block_1 = residual_block(128, 128, p=p)

        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.bottle_neck2 = conv_layer(128, 256, ksize=1)
        self.residual_block_2 = residual_block(256, 256, p=p)

        self.pool3 = nn.AvgPool1d(kernel_size=2)
        self.bottle_neck3 = conv_layer(256, 512, ksize=1)
        self.residual_block_3 = residual_block(512, 512, p=p)

        self.fc1 = nn.Sequential(
            nn.Linear(4, 20),
            nn.Tanh(),

            nn.Linear(20, 10),
            nn.Tanh()
        )
        self.fc2 = nn.Linear(10, 2)

        # self.fc = nn.Linear(512 * int(self.fixed_len / 8) + 10, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 + 10, 4)

    def forward(self, x, rrs):

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

        net = self.gap(net)

        waveform_feat = net.view(-1, 512)

        feat_rrs = self.fc1(rrs)
        pef = self.fc2(feat_rrs)

        feat_con = torch.cat([waveform_feat, feat_rrs], dim=1)

        logits = self.fc(feat_con)

        return net, pef, feat_con, logits


class Encoder(nn.Module):

    def __init__(self, fixed_len=128, p=0.0):
        super(Encoder, self).__init__()

        self.fixed_len = fixed_len

        self.conv_layer1 = conv_layer(1, 64)

        self.stem = conv_stem(64, p=p)

        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.bottle_neck1 = conv_layer(64, 128, ksize=1)
        self.residual_block_1 = residual_block(128, 128, p=p)

        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.bottle_neck2 = conv_layer(128, 256, ksize=1)
        self.residual_block_2 = residual_block(256, 256, p=p)

        self.pool3 = nn.AvgPool1d(kernel_size=2)
        self.bottle_neck3 = conv_layer(256, 512, ksize=1)
        self.residual_block_3 = residual_block(512, 512, p=p)

        self.gap = nn.AdaptiveAvgPool1d(1)

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

        net = self.gap(net)

        return net


class Decoder(nn.Module):
    
    def __init__(self, fixed_len=128):
        super(Decoder, self).__init__()

        self.deconv0 = nn.ConvTranspose1d(512, 512, kernel_size=int(fixed_len / 8), stride=1, padding=0)
        self.relu0 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(512)

        self.deconv1 = nn.ConvTranspose1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(256)

        self.deconv2 = nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(256)

        self.ups1 = nn.Upsample(scale_factor=2)

        self.deconv3 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(128)

        self.deconv4 = nn.ConvTranspose1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm1d(128)

        self.ups2 = nn.Upsample(scale_factor=2)

        self.deconv5 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.LeakyReLU()
        self.bn5 = nn.BatchNorm1d(64)

        self.deconv6 = nn.ConvTranspose1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.LeakyReLU()
        self.bn6 = nn.BatchNorm1d(64)

        self.ups3 = nn.Upsample(scale_factor=2)

        self.deconv7 = nn.ConvTranspose1d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        net = self.bn0(self.relu0(self.deconv0(x)))

        net = self.bn1(self.relu1(self.deconv1(net)))
        net = self.bn2(self.relu2(self.deconv2(net)))

        net = self.ups1(net)

        net = self.bn3(self.relu3(self.deconv3(net)))
        net = self.bn4(self.relu4(self.deconv4(net)))

        net = self.ups2(net)

        net = self.bn5(self.relu5(self.deconv5(net)))
        net = self.bn6(self.relu6(self.deconv6(net)))

        net = self.ups3(net)
        net = self.deconv7(net)

        return net


class ATT(nn.Module):
    
    def __init__(self, len_1, len_2):
        super(ATT, self).__init__()
        self.len_1 = len_1
        self.len_2 = len_2
        self.fc = nn.Linear(2, 2, bias=False)

    def forward(self, v1, v2):
        '''
        :param v1: the feature vector with shape: (N, len_1)
        :param v2: the feature vector with shape: (N, len_2)
        :return: The re-weighted v1_ and v2_
        '''

        x_1 = torch.mean(v1, dim=1, keepdim=True)
        x_2 = torch.mean(v2, dim=1, keepdim=True)

        x = torch.cat([x_1, x_2], dim=1)
        x_ = F.softmax(self.fc(x), dim=1)
        x1_ = torch.index_select(x_, dim=1, index=torch.LongTensor([0]).cuda()).repeat((1, self.len_1))
        x2_ = torch.index_select(x_, dim=1, index=torch.LongTensor([1]).cuda()).repeat((1, self.len_2))

        v1_ = v1 * x1_
        v2_ = v2 * x2_

        return v1_, v2_


class CNN_ATT(nn.Module):

    def __init__(self, fixed_len=200, p=0.0):
        super(CNN_ATT, self).__init__()

        self.fixed_len = fixed_len

        self.conv_layer1 = conv_layer(1, 64)

        self.stem = conv_stem(64, p=p)

        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.bottle_neck1 = conv_layer(64, 128, ksize=1)
        self.residual_block_1 = residual_block(128, 128, p=p)

        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.bottle_neck2 = conv_layer(128, 256, ksize=1)
        self.residual_block_2 = residual_block(256, 256, p=p)

        self.pool3 = nn.AvgPool1d(kernel_size=2)
        self.bottle_neck3 = conv_layer(256, 512, ksize=1)
        self.residual_block_3 = residual_block(512, 512, p=p)

        self.fc1 = nn.Sequential(
            nn.Linear(4, 20),
            nn.Tanh(),

            nn.Linear(20, 10),
            nn.Tanh()
        )
        self.fc2 = nn.Linear(10, 2)

        self.att = ATT(len_1=512, len_2=10)

        # self.fc = nn.Linear(512 * int(self.fixed_len / 8) + 10, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 + 10, 4)

    def forward(self, x, rrs):

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

        waveform_feat = self.gap(net).view(-1, 512)

        feat_rrs = self.fc1(rrs)
        pef = self.fc2(feat_rrs)

        waveform_feat, feat_rrs = self.att(waveform_feat, feat_rrs)
        feat_con = torch.cat([waveform_feat, feat_rrs], dim=1)

        logits = self.fc(feat_con)

        return net, pef, feat_con, logits


