# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import math
import torch.nn as nn
from models.SuperProxylessNAS.modules_AutoML.layers import *
from models.SuperProxylessNAS.modules_AutoML.mix_op import *


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileInvertedResidualBlock(nn.Module):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut is None:
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res


class SuperProxylessNASNets_AutoML3(nn.Module):

    def __init__(self, n_classes=1000, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0):
        super(SuperProxylessNASNets_AutoML3, self).__init__()

        width_stages = [24, 40, 80, 96, 192, 320]
        n_cell_stages = [4, 4, 4, 4, 4, 1]
        stride_stages = [2, 2, 2, 1, 2, 1]
        conv_candidates = [
            '3x3_MBConv3', '3x3_MBConv6',
            '5x5_MBConv3', '5x5_MBConv6',
            '7x7_MBConv3', '7x7_MBConv6',
        ]

        input_channel = make_divisible(32 * width_mult, 8)
        first_cell_width = make_divisible(16 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = make_divisible(width_stages[i] * width_mult, 8)

        # first conv layer
        self.first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )

        # first block
        first_block_conv = MixedEdge(candidate_ops=build_candidate_ops(
            ['3x3_MBConv1'],
            input_channel, first_cell_width, 1, 'weight_bn_act',
        ), )
        if first_block_conv.n_choices == 1:
            first_block_conv = first_block_conv.candidate_ops[0]
        first_block = MobileInvertedResidualBlock(first_block_conv, None)
        input_channel = first_cell_width

        # blocks
        blocks = [first_block]
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                # conv
                modified_conv_candidates = conv_candidates
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    modified_conv_candidates, input_channel, width, stride, 'weight_bn_act',
                ), )
                # shortcut
                if stride == 1 and input_channel == width:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
                blocks.append(inverted_residual_block)
                input_channel = width

        self.blocks = nn.ModuleList(blocks)
        # feature mix layer
        last_channel = make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_ss_act',
        )

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        # set bn param
        self.set_bn_param(momentum=0.1, eps=1e-3)

        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def set_conv_mask(self, layer_index, pruned_channel_index):
        for index, module in enumerate(self.modules()):
            if (layer_index == index) and module.__str__().startswith('MaskedConv2d'):  #[cout, cin, k, k]
                for i in pruned_channel_index:
                    module._mask[i,:,:,:] = 0

                if module.groups > 1:    #[cout, cin/g, k, k]
                    # set the mask of its previous convolutions
                    for index, module in enumerate(self.modules()):
                        if (layer_index > index) and module.__str__().startswith('MaskedConv2d'):  #[cout, cin, k, k]
                            module_ = module
                        if (layer_index == index) and module.__str__().startswith('MaskedConv2d'):  #[cout, cin, k, k]
                            for i in pruned_channel_index:
                                module_._mask[i,:,:,:] = 0
                            break

                    for index, module in enumerate(self.modules()):
                        if (layer_index < index) and module.__str__().startswith('MaskedConv2d'):  #[cout, cin, k, k]
                            for i in pruned_channel_index:
                                module._mask[:,i,:,:] = 0
                            break
                break

#        for index, module in enumerate(self.modules()):
#            if (layer_index < index) and module.__str__().startswith('MaskedConv2d'):  #[cout, cin, k, k]
#                for j in pruned_channel_index:
#                    module._mask[:,j,:,:] = 0
#                break
#            if (layer_index < index) and module.__str__().startswith('MaskedLinear'):  #[cout, cin, k, k]
#                for j in pruned_channel_index:
#                    module._mask[:,j] = 0
#                break

    def set_linear_mask(self, layer_index, pruned_channel_index):
        for index, module in enumerate(self.modules()):
            if (layer_index == index) and module.__str__().startswith('MaskedLinear'):  #[cout, cin, k, k]
                for i in pruned_channel_index:
                    module._mask[i,:] = 0
                break

#        for index, module in enumerate(self.modules()):
#            if (layer_index < index) and module.__str__().startswith('MaskedLinear'):  #[cout, cin, k, k]
#                for j in pruned_channel_index:
#                    module._mask[:,j] = 0
#                break
