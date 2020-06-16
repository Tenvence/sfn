import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck
from ..Modules.conv import Conv


class DualResNet(nn.Module):
    def __init__(self, layers, groups=1, width_per_group=64):
        super(DualResNet, self).__init__()

        self.in_planes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        self.conv_1_x1 = nn.Conv2d(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_1_x1 = nn.BatchNorm2d(self.in_planes)
        self.relu_1_x1 = nn.LeakyReLU(inplace=True)
        self.max_pool_x1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_1_x1 = self._make_layer(64, layers[0])
        self.layer_2_x1 = self._make_layer(128, layers[1], stride=2)

        self.conv_1_x2 = nn.Conv2d(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_1_x2 = nn.BatchNorm2d(self.in_planes)
        self.relu_1_x2 = nn.ReLU(inplace=True)
        self.max_pool_x2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_1_x2 = self._make_layer(64, layers[0])
        self.layer_2_x2 = self._make_layer(128, layers[1], stride=2)

        self.conv_concat = Conv(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.layer_3 = self._make_layer(256, layers[2], stride=2)
        self.layer_4 = self._make_layer(512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x1, x2):
        x1 = self.conv_1_x1(x1)
        x2 = self.conv_1_x2(x2)

        x1 = self.bn_1_x1(x1)
        x2 = self.bn_1_x2(x2)

        x1 = self.relu_1_x1(x1)
        x2 = self.relu_1_x2(x2)

        x1 = self.max_pool_x1(x1)
        x2 = self.max_pool_x2(x2)

        x1 = self.layer_1_x1(x1)
        x2 = self.layer_1_x2(x2)

        x1 = self.layer_2_x1(x1)
        x2 = self.layer_2_x2(x2)

        # x = torch.cat([x1, x2], dim=1)
        # x = self.conv_concat(x)
        x = torch.mul(x1, x2)

        route_s = x
        x = self.layer_3(x)
        route_m = x
        x = self.layer_4(x)
        route_l = x

        return route_s, route_m, route_l

    def _make_layer(self, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_planes != planes * Bottleneck.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = [Bottleneck(self.in_planes, planes, stride, down_sample, self.groups, self.base_width, self.dilation)]
        self.in_planes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)


def dual_resnet50():
    return DualResNet([3, 4, 6, 3])


def dual_resnet101():
    return DualResNet([3, 4, 23, 3])


def dual_resnext50():
    return DualResNet([3, 4, 6, 3], groups=32, width_per_group=4)


def dual_resnext101():
    return DualResNet([3, 4, 23, 3], groups=32, width_per_group=8)
