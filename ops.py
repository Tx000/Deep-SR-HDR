import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class get_offset(nn.Module):
    def __init__(self, filters):
        super(get_offset, self).__init__()
        self.conv_offset1 = nn.Conv2d(filters * 2, filters, 3, 1, 1, bias=True)
        self.conv_offset2 = nn.Conv2d(filters, filters, 3, 1, 1, bias=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ali, ref):
        offset = self.lrule(self.conv_offset1(torch.cat([ali, ref], dim=1)))
        offset = self.lrule(self.conv_offset2(offset))
        return offset


class make_res(nn.Module):
    def __init__(self, nFeat, kernel_size=3):
        super(make_res, self).__init__()
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out) + x
        return out


class Res_block(nn.Module):
    def __init__(self, nFeat, nReslayer):
        super(Res_block, self).__init__()
        modules = []
        for i in range(nReslayer):
            modules.append(make_res(nFeat))
        self.dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        out = self.dense_layers(x)
        return out


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv(x), inplace=True)
        out = torch.cat((x, out), 1)
        return out


class make_dense_light(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(make_dense_light, self).__init__()
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(growthRate, growthRate, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB)
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act='no', bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, 1, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, 1, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)