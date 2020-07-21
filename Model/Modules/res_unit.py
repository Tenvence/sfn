import torch.nn as nn
from .conv import Conv


class ResUnit(nn.Module):
    def __init__(self, in_channels, activation='leaky_relu'):
        super(ResUnit, self).__init__()
        self.conv_1 = Conv(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, activation=activation)
        self.conv_2 = Conv(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, activation=activation)

    def forward(self, x):
        conv = self.conv_1(x)
        conv = self.conv_2(conv)
        x = x + conv

        return x
