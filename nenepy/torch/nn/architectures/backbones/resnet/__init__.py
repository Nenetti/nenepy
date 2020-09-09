import torch.utils.model_zoo as model_zoo

from .modules import BasicBlock, Bottleneck
from .resnet import ResNet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class ResNet18(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))


class ResNet34(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
            print("A")


class ResNet50(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], **kwargs)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))


class ResNet101(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 23, 3], **kwargs)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet101']))


class ResNet152(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3], **kwargs)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
