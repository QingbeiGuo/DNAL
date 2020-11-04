import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, 
              kernel_size, stride, padding, dilation, groups, bias)
        self.register_buffer('_mask', torch.ones(self.weight.size()))

    def forward(self, x):
        ### Masked output
        weight = self.weight * self.mask
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @property
    def mask(self):
        return Variable(self._mask)

    @mask.setter
    def mask(self, value):
        self._mask = value


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias=True)
        self.register_buffer('_mask', torch.ones(self.weight.size()))

    def forward(self, x):
        ### Masked output
        weight = self.weight * self.mask
        return F.linear(x, weight, self.bias)

    @property
    def mask(self):
        return Variable(self._mask)

    @mask.setter
    def mask(self, value):
        self._mask = value


class ScaleLayer2d(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.scale = nn.Parameter(torch.FloatTensor(out_channels))
        self.register_buffer('_mask', torch.ones(out_channels))
        self.register_buffer('_delta', torch.ones(out_channels))
        self.register_buffer('_ignore', torch.tensor(1))
        self.reset_parameter()

    def reset_parameter(self):
        self.scale.data = torch.zeros(self.out_channels)

    def forward(self, x):
        if self.ignore == 0:
            b,c,w,h = x.size()
            x = x.transpose(0,1).contiguous().reshape(c,-1)
            x = x * torch.sigmoid(self.delta.reshape(c,1) * self.scale.reshape(c,1)) * self.mask.reshape(c,1)
            return x.reshape(c,b,w,h).transpose(0,1)
        elif self.ignore == 1:
            b,c,w,h = x.size()
            x = x.transpose(0,1).contiguous().reshape(c,-1)
            x = x * self.mask.reshape(c,1)
            return x.reshape(c,b,w,h).transpose(0,1)
        elif self.ignore == 2:
            return x

    @property
    def mask(self):
        return Variable(self._mask)

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def delta(self):
        return Variable(self._delta)

    @delta.setter
    def delta(self, value):
        self._delta = value

    @property
    def ignore(self):
        return Variable(self._ignore)

    @ignore.setter
    def ignore(self, value):
        self._ignore = value


class ScaleLayer1d(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.scale = nn.Parameter(torch.FloatTensor(out_channels))
        self.register_buffer('_mask', torch.ones(out_channels))
        self.register_buffer('_delta', torch.ones(out_channels))
        self.register_buffer('_ignore', torch.tensor(1))
        self.reset_parameter()

    def reset_parameter(self):
        self.scale.data = torch.zeros(self.out_channels)

    def forward(self, x):
        if self.ignore == 0:
            b,c = x.size()
            x = x.transpose(0,1).contiguous().reshape(c,-1)
            x = x * torch.sigmoid(self.delta.reshape(c,1) * self.scale.reshape(c,1)) * self.mask.reshape(c,1)
            return x.reshape(c,b).transpose(0,1)
        elif self.ignore == 1:
            b,c = x.size()
            x = x.transpose(0,1).contiguous().reshape(c,-1)
            x = x * self.mask.reshape(c,1)
            return x.reshape(c,b).transpose(0,1)
        elif self.ignore == 2:
            return x

    @property
    def mask(self):
        return Variable(self._mask)

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def delta(self):
        return Variable(self._delta)

    @delta.setter
    def delta(self, value):
        self._delta = value

    @property
    def ignore(self):
        return Variable(self._ignore)

    @ignore.setter
    def ignore(self, value):
        self._ignore = value
