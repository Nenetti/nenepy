from torch import nn
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import ResNet as TorchResNet


def deconv3x3(in_channels, out_channels, kernel_size=3, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def deconv1x1(in_channels, out_channels, kernel_size=1, stride=1):
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)


class BackwardBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, base_channels=64, kernel_size=3, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, activation_layer=None, activation_kwargs={}):
        super(BackwardBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if stride == 1:
            self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        else:
            self.conv1 = deconv3x3(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = activation_layer(**activation_kwargs)
        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BackwardBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, base_channels, kernel_size=3, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, activation_layer=None, activation_kwargs={}):
        super(BackwardBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU
        width = int(base_channels * (base_width / 64.)) * groups

        self.conv1 = deconv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = deconv3x3(width, width, kernel_size, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = deconv1x1(width, out_channels)
        self.bn3 = norm_layer(out_channels)
        self.relu = activation_layer(**activation_kwargs)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecoderResNet(TorchResNet):

    def __init__(self, block, layers, pretrained_state_dict=None, is_train=False,
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
        self._training_layers = []

    # ==============================================================================
    #
    #   From torchvision's ResNet
    #
    # ==============================================================================

    def _default_initialize(self, block, layers, num_classes=1000, zero_init_residual=False,
                            groups=1, width_per_group=64, replace_stride_with_dilation=None,
                            norm_layer=None, reduction_rate=32, out_channels=3, activation_layer=None, activation_kwargs={}):

        if reduction_rate == 32:
            layer4_stride = 2
        elif reduction_rate == 16:
            layer4_stride = 1
        else:
            raise ValueError()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU
        self._norm_layer = norm_layer

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
        self.conv1 = nn.ConvTranspose2d(64, out_channels, kernel_size=8, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.relu = activation_layer(**activation_kwargs)
        self.maxpool = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=layer4_stride, dilate=replace_stride_with_dilation[2],
                                       activation_layer=activation_layer, activation_kwargs=activation_kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       activation_layer=activation_layer, activation_kwargs=activation_kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       activation_layer=activation_layer, activation_kwargs=activation_kwargs)
        self.layer1 = self._make_layer(block, 64, layers[0], out_channels=64,
                                       activation_layer=activation_layer, activation_kwargs=activation_kwargs)
        self._post_processing = nn.Sigmoid()

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
                if isinstance(m, BackwardBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, base_channels, n_blocks, stride=1, dilate=False, out_channels=None, activation_layer=None, activation_kwargs={}):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation /= stride
            stride = 1

        in_channels = base_channels * block.expansion

        if out_channels is None:
            out_channels = in_channels // 2

        kernel_size = (stride * 2) if (stride % 2 == 0) else 3

        if (stride != 1) or (out_channels != base_channels // block.expansion):
            downsample = nn.Sequential(
                deconv1x1(in_channels=in_channels, out_channels=out_channels, kernel_size=stride, stride=stride),
                norm_layer(out_channels),
            )

        layers = []
        for _ in range(0, n_blocks - 1):
            layers.append(
                block(
                    in_channels=in_channels, out_channels=in_channels, base_channels=base_channels, kernel_size=3,
                    groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer,
                    activation_layer=activation_layer, activation_kwargs=activation_kwargs
                )
            )

        layers.append(
            block(
                in_channels=in_channels, out_channels=out_channels, base_channels=base_channels, kernel_size=kernel_size, stride=stride,
                downsample=downsample, groups=self.groups, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer,
                activation_layer=activation_layer, activation_kwargs=activation_kwargs
            )
        )

        return nn.Sequential(*layers)

    # ==============================================================================
    #
    #   Custom Method
    #
    # ==============================================================================

    def forward(self, x):
        return self._forward_impl(x)

    def _forward_impl(self, x):
        x0 = self.layer4(x)
        x1 = self.layer3(x0)
        x2 = self.layer2(x1)
        x3 = self.layer1(x2)
        x4 = self.maxpool(x3)
        x4 = self.conv1(x4)

        # x4 = self._post_processing(x4)

        return x4

    def train(self, mode=True):
        super(DecoderResNet, self).train()
        self.requires_grad_(True)

    def eval(self):
        super(DecoderResNet, self).eval()
        self.requires_grad_(False)
