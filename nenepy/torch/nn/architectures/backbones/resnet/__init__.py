import torch.utils.model_zoo as model_zoo

from .decoder_resnet import DecoderResNet, BackwardBottleneck, BackwardBasicBlock
from .resnet import ResNet, BasicBlock, Bottleneck

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
        kwargs["pretrained_state_dict"] = model_zoo.load_url(model_urls['resnet18']) if pretrained else None
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], **kwargs)


class ResNet34(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        kwargs["pretrained_state_dict"] = model_zoo.load_url(model_urls['resnet34']) if pretrained else None
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3], **kwargs)


class ResNet50(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        kwargs["pretrained_state_dict"] = model_zoo.load_url(model_urls['resnet50']) if pretrained else None
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], **kwargs)


class ResNet101(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        kwargs["pretrained_state_dict"] = model_zoo.load_url(model_urls['resnet101']) if pretrained else None
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 23, 3], **kwargs)


class ResNet152(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        kwargs["pretrained_state_dict"] = model_zoo.load_url(model_urls['resnet152']) if pretrained else None
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3], **kwargs)


class ResNeXt50_32x4d(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 4
        kwargs["pretrained_state_dict"] = model_zoo.load_url(model_urls['resnext50_32x4d']) if pretrained else None
        super(ResNeXt50_32x4d, self).__init__(Bottleneck, [3, 4, 6, 3], **kwargs)


class ResNeXt101_32x8d(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 8
        kwargs["pretrained_state_dict"] = model_zoo.load_url(model_urls['resnext101_32x8d']) if pretrained else None
        super(ResNeXt101_32x8d, self).__init__(Bottleneck, [3, 4, 23, 3], **kwargs)


class WideResNet50_2(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        kwargs['width_per_group'] = 64 * 2
        kwargs["pretrained_state_dict"] = model_zoo.load_url(model_urls['wide_resnet50_2']) if pretrained else None
        super(WideResNet50_2, self).__init__(Bottleneck, [3, 4, 6, 3], **kwargs)


class WideResNet101_2(ResNet):
    def __init__(self, pretrained=False, **kwargs):
        kwargs['width_per_group'] = 64 * 2
        kwargs["pretrained_state_dict"] = model_zoo.load_url(model_urls['wide_resnet101_2']) if pretrained else None
        super(WideResNet101_2, self).__init__(Bottleneck, [3, 4, 23, 3], **kwargs)


class DecoderResNet18(DecoderResNet):
    """
    Input Channel = 512

    Shapes:
        ReductionRate=16: [-1, 512, W, H] -> [-1, C, 16W, 16H]
        ReductionRate=32: [-1, 512, W, H] -> [-1, C, 32W, 32H]

    """

    def __init__(self, **kwargs):
        super(DecoderResNet18, self).__init__(BackwardBasicBlock, [2, 2, 2, 2], **kwargs)


class DecoderResNet34(DecoderResNet):
    """
    Input Channel = 512

    Shapes:
        ReductionRate=16: [-1, 512, W, H] -> [-1, C, 16W, 16H]
        ReductionRate=32: [-1, 512, W, H] -> [-1, C, 32W, 32H]

    """

    def __init__(self, **kwargs):
        super(DecoderResNet34, self).__init__(BackwardBasicBlock, [3, 4, 6, 3], **kwargs)


class DecoderResNet50(DecoderResNet):
    """
    Input Channel = 2048

    Shapes:
        ReductionRate=16: [-1, 2048, W, H] -> [-1, C, 16W, 16H]
        ReductionRate=32: [-1, 2048, W, H] -> [-1, C, 32W, 32H]

    """

    def __init__(self, **kwargs):
        super(DecoderResNet50, self).__init__(BackwardBottleneck, [3, 4, 6, 3], **kwargs)


class DecoderResNet101(DecoderResNet):
    """
    Input Channel = 2048

    Shapes:
        ReductionRate=16: [-1, 2048, W, H] -> [-1, C, 16W, 16H]
        ReductionRate=32: [-1, 2048, W, H] -> [-1, C, 32W, 32H]

    """

    def __init__(self, **kwargs):
        super(DecoderResNet101, self).__init__(BackwardBottleneck, [3, 4, 23, 3], **kwargs)


class DecoderResNet152(DecoderResNet):
    """
    Input Channel = 2048

    Shapes:
        ReductionRate=16: [-1, 2048, W, H] -> [-1, C, 16W, 16H]
        ReductionRate=32: [-1, 2048, W, H] -> [-1, C, 32W, 32H]

    """

    def __init__(self, **kwargs):
        super(DecoderResNet152, self).__init__(BackwardBottleneck, [3, 8, 36, 3], **kwargs)


class DecoderResNeXt50_32x4d(DecoderResNet):
    """
    Input Channel = 2048

    Shapes:
        ReductionRate=16: [-1, 2048, W, H] -> [-1, C, 16W, 16H]
        ReductionRate=32: [-1, 2048, W, H] -> [-1, C, 32W, 32H]

    """

    def __init__(self, **kwargs):
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 4
        super(DecoderResNeXt50_32x4d, self).__init__(BackwardBottleneck, [3, 4, 6, 3], **kwargs)


class DecoderResNeXt101_32x8d(DecoderResNet):
    """
    Input Channel = 2048

    Shapes:
        ReductionRate=16: [-1, 2048, W, H] -> [-1, C, 16W, 16H]
        ReductionRate=32: [-1, 2048, W, H] -> [-1, C, 32W, 32H]

    """

    def __init__(self, **kwargs):
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 8
        super(DecoderResNeXt101_32x8d, self).__init__(BackwardBottleneck, [3, 4, 23, 3], **kwargs)


class DecoderWideResNet50_2(DecoderResNet):
    """
    Input Channel = 2048

    Shapes:
        ReductionRate=16: [-1, 2048, W, H] -> [-1, C, 16W, 16H]
        ReductionRate=32: [-1, 2048, W, H] -> [-1, C, 32W, 32H]

    """

    def __init__(self, **kwargs):
        kwargs['width_per_group'] = 64 * 2
        super(DecoderWideResNet50_2, self).__init__(BackwardBottleneck, [3, 4, 6, 3], **kwargs)


class DecoderWideResNet101_2(DecoderResNet):
    """
    Input Channel = 2048

    Shapes:
        ReductionRate=16: [-1, 2048, W, H] -> [-1, C, 16W, 16H]
        ReductionRate=32: [-1, 2048, W, H] -> [-1, C, 32W, 32H]

    """

    def __init__(self, **kwargs):
        kwargs['width_per_group'] = 64 * 2
        super(DecoderWideResNet101_2, self).__init__(BackwardBottleneck, [3, 4, 23, 3], **kwargs)


__all__ = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNeXt50_32x4d",
    "ResNeXt101_32x8d",
    "WideResNet50_2",
    "WideResNet101_2",
    "DecoderResNet18",
    "DecoderResNet34",
    "DecoderResNet50",
    "DecoderResNet101",
    "DecoderResNet152",
    "DecoderResNeXt50_32x4d",
    "DecoderResNeXt101_32x8d",
    "DecoderWideResNet50_2",
    "DecoderWideResNet101_2",
]
