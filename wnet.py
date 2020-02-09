###
# Based on https://github.com/QuantScientist/V-Net.pytorch/blob/master/vnet.py
# Authors:  Tom Nuno Wolf
#           Eren Arkangil
#           Fikret Yalcinbas
###


import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(4, outChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 3, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(3)
        self.conv2 = nn.Conv3d(3, 3, kernel_size=1)
        self.relu1 = ELUCons(elu, 3)
        # self.nll = nll
        # if nll:
        #     self.softmax = F.log_softmax(dim=1)
        # else:
        #     self.softmax = F.softmax(dim=1)

    def forward(self, x):
        # convolve 32 down to 3 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # if self.nll:
        #     our = F.log_softmax(out, dim=1)
        # else:
        #     out = F.softmax(out, dim=1)
        out = torch.sigmoid(out)
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=False, nll=False):
        # super(VNet, self).__init__()
        # self.in_tr = InputTransition(16, elu)
        # self.down_tr32 = DownTransition(16, 1, elu)
        # self.down_tr64 = DownTransition(32, 2, elu)
        # self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        # self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        # self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        # self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        # self.up_tr64 = UpTransition(128, 64, 1, elu)
        # self.up_tr32 = UpTransition(64, 32, 1, elu)
        # self.out_tr = OutputTransition(32, elu, nll)

        super(VNet, self).__init__()
        self.in_tr = InputTransition(4, elu)
        self.down_tr32 = DownTransition(4, 1, elu)
        self.down_tr64 = DownTransition(8, 2, elu)
        self.down_tr128 = DownTransition(16, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(32, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(64, 64, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(64, 32, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(32, 16, 1, elu)
        self.up_tr32 = UpTransition(16, 8, 1, elu)
        self.out_tr = OutputTransition(8, elu, nll)

        # super(VNet, self).__init__()
        # self.in_tr = InputTransition(2, elu)
        # self.down_tr32 = DownTransition(2, 1, elu)
        # self.down_tr64 = DownTransition(4, 2, elu)
        # self.down_tr128 = DownTransition(8, 3, elu, dropout=True)
        # self.down_tr256 = DownTransition(16, 2, elu, dropout=True)
        # self.up_tr256 = UpTransition(32, 32, 2, elu, dropout=True)
        # self.up_tr128 = UpTransition(32, 16, 2, elu, dropout=True)
        # self.up_tr64 = UpTransition(16, 8, 1, elu)
        # self.up_tr32 = UpTransition(8, 4, 1, elu)
        # self.out_tr = OutputTransition(4, elu, nll)

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):

        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

# import numpy as np
# loss = torch.nn.BCELoss()
# model = VNet()
# model.train()
# print(sum(p.numel() for p in model.parameters()))
# a = np.zeros((8, 4, 64, 64, 64), dtype=np.float32)
# a = torch.from_numpy(a)
# c = np.zeros((8,3,64, 64, 64), dtype=np.float32)
# c=  torch.from_numpy(c)
# b = model(a)
# print(b.size(), c.size())
# l = loss(b, c)
# print(l)
