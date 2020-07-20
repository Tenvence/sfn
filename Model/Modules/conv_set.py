import torch
import torch.nn as nn
from .conv import Conv


class ConvSet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvSet, self).__init__()
        self.conv_1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_2 = Conv(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv_3 = Conv(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_4 = Conv(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv_5 = Conv(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)

        return x


class SPPConvSet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPConvSet, self).__init__()
        self.conv_1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_2 = Conv(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv_3 = Conv(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv_cat = Conv(out_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_4 = Conv(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv_5 = Conv(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        pool_1 = self.max_pool_1(x)
        pool_2 = self.max_pool_2(x)
        pool_3 = self.max_pool_3(x)
        x = torch.cat([pool_1, pool_2, pool_3, x], dim=1)
        x = self.conv_cat(x)
        x = self.conv_4(x)
        x = self.conv_5(x)

        return x
