from torch import nn
import math
#from .utils import load_state_dict_from_url
from models.mobilenet.layers import MaskedConv2d
from models.mobilenet.layers import MaskedLinear
from models.mobilenet.layers import ScaleLayer2d, ScaleLayer1d

__all__ = ['MobileNetV2_AutoML', 'mobilenetv2_AutoML']


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        MaskedConv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        MaskedConv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        ScaleLayer2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                MaskedConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                ScaleLayer2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                MaskedConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                ScaleLayer2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                MaskedConv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #ScaleLayer2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                MaskedConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                ScaleLayer2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                MaskedConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                ScaleLayer2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_AutoML(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.):
        super(MobileNetV2_AutoML, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            #nn.Dropout(0.2),
            MaskedLinear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
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
                break

        for index, module in enumerate(self.modules()):
            if (layer_index < index) and module.__str__().startswith('MaskedConv2d'):  #[cout, cin, k, k]
                for j in pruned_channel_index:
                    module._mask[:,j,:,:] = 0
                break
            if (layer_index < index) and module.__str__().startswith('MaskedLinear'):  #[cout, cin, k, k]
                for j in pruned_channel_index:
                    module._mask[:,j] = 0
                break

    def set_linear_mask(self, layer_index, pruned_channel_index):
        for index, module in enumerate(self.modules()):
            if (layer_index == index) and module.__str__().startswith('MaskedLinear'):  #[cout, cin, k, k]
                for i in pruned_channel_index:
                    module._mask[i,:] = 0
                break

        for index, module in enumerate(self.modules()):
            if (layer_index < index) and module.__str__().startswith('MaskedLinear'):  #[cout, cin, k, k]
                for j in pruned_channel_index:
                    module._mask[:,j] = 0
                break


def mobilenetv2_AutoML(**kwargs):
    model = MobileNetV2_AutoML(**kwargs)
    return model