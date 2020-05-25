import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride,
                     padding=dilation, groups=groups,
                     bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample
        # layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class BottleneckWithBuffer(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckWithBuffer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, shift_buffer):
        identity = x

        c = x.size(1)
        x1, x2 = x[:, :c // 8], x[:, c // 8:]
        out = torch.cat((shift_buffer, x2), dim=1)
        out = self.conv1(out)
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

        return out, x1


class BottleneckSequence(nn.Sequential):
    def forward(self, x, *shift_buffer):
        output_shift_buffer = []
        for idx, module in enumerate(self._modules.values()):
            x, cur_shift_buffer = module(x, shift_buffer[idx])
            output_shift_buffer.append(cur_shift_buffer)
        return x, output_shift_buffer


class ResNet(nn.Module):

    def __init__(self,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.blocks_cnt = []
        for layer in layers:
            self.blocks_cnt.append(layer if len(self.blocks_cnt) == 0
                                   else layer + self.blocks_cnt[-1])

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7,
                               stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                norm_layer(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(BottleneckWithBuffer(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(BottleneckWithBuffer(self.inplanes, planes,
                                               groups=self.groups,
                                               base_width=self.base_width,
                                               dilation=self.dilation,
                                               norm_layer=norm_layer))

        return BottleneckSequence(*layers)

    def forward(self, x, *shift_buffer):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, b1 = self.layer1(x, *shift_buffer[:self.blocks_cnt[0]])
        x, b2 = self.layer2(
            x, *shift_buffer[self.blocks_cnt[0]:self.blocks_cnt[1]])
        x, b3 = self.layer3(
            x, *shift_buffer[self.blocks_cnt[1]:self.blocks_cnt[2]])
        x, b4 = self.layer4(x, *shift_buffer[self.blocks_cnt[2]:])

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.softmax(x, dim=1)

        return (x, *(b1 + b2 + b3 + b4))


def resnet50(**kwargs):
    return ResNet([3, 4, 6, 3], **kwargs)


if __name__ == '__main__':
    # [3, 4, 6, 3]
    resnet50_shift_buffer = [
        torch.zeros([1, 8, 56, 56]),
        torch.zeros([1, 32, 56, 56]),
        torch.zeros([1, 32, 56, 56]),

        torch.zeros([1, 32, 56, 56]),
        torch.zeros([1, 64, 28, 28]),
        torch.zeros([1, 64, 28, 28]),
        torch.zeros([1, 64, 28, 28]),

        torch.zeros([1, 64, 28, 28]),
        torch.zeros([1, 128, 14, 14]),
        torch.zeros([1, 128, 14, 14]),
        torch.zeros([1, 128, 14, 14]),
        torch.zeros([1, 128, 14, 14]),
        torch.zeros([1, 128, 14, 14]),

        torch.zeros([1, 128, 14, 14]),
        torch.zeros([1, 256, 7, 7]),
        torch.zeros([1, 256, 7, 7]),
    ]
    model = resnet50()
    outputs = model(torch.randn((1, 3, 224, 224)), *resnet50_shift_buffer)
    for output in outputs:
        print(output.shape)
