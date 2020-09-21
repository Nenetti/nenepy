import inspect
import pprint

import torch
from torch import nn
from torchvision.models.resnet import ResNet as TorchResNet, conv1x1
from torchvision.models.resnet import BasicBlock as TorchBasicBlock
from torchvision.models.resnet import Bottleneck as TorchBottleneck


class BasicBlock(TorchBasicBlock):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None,
                 activation_layer=None, activation_kwargs={}):
        super(BasicBlock, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        if activation_layer is not None:
            self.relu = activation_layer(**activation_kwargs)


class Bottleneck(TorchBottleneck):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None,
                 activation_layer=None, activation_kwargs={}):
        super(Bottleneck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        if activation_layer is not None:
            self.relu = activation_layer(**activation_kwargs)


class ResNet(TorchResNet):

    def __init__(self, block, layers,
                 in_channels=3, pretrained_state_dict=None, is_train=False, is_feature_extraction=False,
                 **kwargs
                 ):
        nn.Module.__init__(self)

        # ----- Default Initialization (From torchvision) ----- #
        self._default_initialize(block, layers, **kwargs)

        if pretrained_state_dict is not None:
            self.load_state_dict(pretrained_state_dict, strict=True)
        else:
            is_train = True

        # ----- Custom ----- #
        self._is_train = is_train
        self._is_feature_extraction = is_feature_extraction
        self._training_layers = []

        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            self._training_layers.append(self.conv1)

        if self._is_feature_extraction:
            del self.avgpool, self.fc

    # ==============================================================================
    #
    #   From torchvision's ResNet
    #
    # ==============================================================================

    def _default_initialize(self, block, layers, num_classes=1000, zero_init_residual=False,
                            groups=1, width_per_group=64, replace_stride_with_dilation=None,
                            norm_layer=None, reduction_rate=32, activation_layer=None, activation_kwargs={}):

        layer_strides = [1, 2, 2, 2]
        if reduction_rate < 32:
            layer_strides[3] = 1
        if reduction_rate < 16:
            layer_strides[2] = 1
        if reduction_rate < 8:
            layer_strides[1] = 1

        # if reduction_rate == 32:
        #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # elif reduction_rate == 16:
        #     self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        # else:
        #     raise ValueError()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = activation_layer(**activation_kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=layer_strides[0],
                                       activation_layer=activation_layer, activation_kwargs=activation_kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=layer_strides[1], dilate=replace_stride_with_dilation[0],
                                       activation_layer=activation_layer, activation_kwargs=activation_kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=layer_strides[2], dilate=replace_stride_with_dilation[1],
                                       activation_layer=activation_layer, activation_kwargs=activation_kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=layer_strides[3], dilate=replace_stride_with_dilation[2],
                                       activation_layer=activation_layer, activation_kwargs=activation_kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, activation_layer=None, activation_kwargs={}):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            activation_layer=activation_layer, activation_kwargs=activation_kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer, activation_kwargs=activation_kwargs))

        return nn.Sequential(*layers)

    # ==============================================================================
    #
    #   Custom Method
    #
    # ==============================================================================

    def forward(self, x):
        if self._is_feature_extraction:
            return self._forward_feature_impl(x)
        else:
            return self._forward_impl(x)

    def _forward_feature_impl(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self._is_feature_extraction:
            return x0, x1, x2, x3, x4

        x5 = self.avgpool(x4)
        x5 = torch.flatten(x5, 1)
        x5 = self.fc(x5)

        return x5

    def train(self, mode=True):
        if mode:
            if self._is_train:
                self.requires_grad_(True)
            else:
                self.requires_grad_(False)

            for module in self._training_layers:
                module.requires_grad_(True)
        else:
            self.requires_grad_(False)
        super(ResNet, self).train()
