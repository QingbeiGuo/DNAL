import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

from models.vgg.layers import MaskedConv2d
from models.vgg.layers import MaskedLinear
from models.vgg.layers import ScaleLayer2d, ScaleLayer1d

__all__ = [
    'VGG_AutoML', 'vgg11_AutoML', 'vgg11_bn_AutoML', 'vgg13_AutoML', 'vgg13_bn_AutoML', 'vgg16_AutoML', 'vgg16_bn_AutoML',
    'vgg19_bn_AutoML', 'vgg19_AutoML',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    }


class VGG_AutoML(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG_AutoML, self).__init__()

        self.features = features
        self.classifier = nn.Sequential(
            MaskedLinear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            #ScaleLayer1d(4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            MaskedLinear(4096, 4096),
            nn.BatchNorm1d(4096),
            #ScaleLayer1d(4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            MaskedLinear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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

    def set_conv_mask(self, layer_index, layer_item):
        convlayers = 0
        for module in self.modules():
            if module.__str__().startswith('MaskedConv2d'):  #[cout, cin, k, k]
                if convlayers == layer_index:
                    for i in layer_item:
                        #print(module._mask.size())
                        module._mask[i,:,:,:] = 0
                        #print(module._mask[i,j,:,:])
                if convlayers == layer_index + 1:
                    for j in layer_item:
                        #print(module._mask.size())
                        module._mask[:,j,:,:] = 0
                        #print(module._mask[i,j,:,:])
                convlayers = convlayers + 1

    def set_linear_mask(self, layer_index, layer_item):
        linearlayers = 0
        for module in self.modules():
            if module.__str__().startswith('MaskedLinear'):  #[cout, cin]
                if linearlayers == layer_index:
                    for i in layer_item:
                        #print(module._mask[i,j])
                        module._mask[i,:] = 0
                        #print(module._mask[i,j])
                if linearlayers == layer_index + 1:
                    for j in layer_item:
                        #print(module._mask[i,j])
                        module._mask[:,j] = 0
                        #print(module._mask[i,j])
                linearlayers = linearlayers + 1

    def set_conv_linear_mask(self, conv_layer_index, linear_layer_index, conv_layer_item, fc_layer_item):
        convlayers = 0
        for module in self.modules():
            if module.__str__().startswith('MaskedConv2d'):  #[cout, cin, k, k]
                if convlayers == conv_layer_index:
                    for i in conv_layer_item:
                        #print(module._mask[i,j,:,:])
                        module._mask[i,:,:,:] = 0
                        #print(module._mask[i,j,:,:])
                convlayers = convlayers + 1

        linearlayers = 0
        for module in self.modules():
            if module.__str__().startswith('MaskedLinear'):  #[cout, cin]
                if linearlayers ==  linear_layer_index:
                    for j in fc_layer_item:
                        #print(module._mask[i,j])
                        module._mask[:,j] = 0
                        #print(module._mask[i,j])
                linearlayers = linearlayers + 1


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i,v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if i == 0:
                conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), ScaleLayer2d(v), nn.ReLU(inplace=True)]
                    #layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, ScaleLayer2d(v), nn.ReLU(inplace=True)]
                    #layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

            else:
                conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), ScaleLayer2d(v), nn.ReLU(inplace=True)]
                    #layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, ScaleLayer2d(v), nn.ReLU(inplace=True)]
                    #layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11_AutoML(pretrained=False, model_root=None, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    model = VGG_AutoML(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11'], model_root))
    return model


def vgg11_bn_AutoML(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG_AutoML(make_layers(cfg['A'], batch_norm=True), **kwargs)


def vgg13_AutoML(pretrained=False, model_root=None, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    model = VGG_AutoML(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13'], model_root))
    return model


def vgg13_bn_AutoML(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG_AutoML(make_layers(cfg['B'], batch_norm=True), **kwargs)


def vgg16_AutoML(pretrained=False, model_root=None, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG_AutoML(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
    return model


def vgg16_bn_AutoML(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG_AutoML(make_layers(cfg['D'], batch_norm=True), **kwargs)


def vgg19_AutoML(pretrained=False, model_root=None, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    model = VGG_AutoML(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19'], model_root))
    return model


def vgg19_bn_AutoML(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG_AutoML(make_layers(cfg['E'], batch_norm=True), **kwargs)
