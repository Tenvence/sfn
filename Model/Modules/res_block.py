import torch
import torch.nn as nn
from .conv import Conv
from .res_unit import ResUnit


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_block, activation='leaky_relu'):
        super(ResBlock, self).__init__()
        self.zero_padding = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv = Conv(in_channels, out_channels, kernel_size=3, stride=2, padding=0, activation=activation)
        self.res_block = nn.ModuleList([ResUnit(out_channels, activation=activation) for _ in range(num_block)])

    def forward(self, x):
        x = self.zero_padding(x)
        x = self.conv(x)
        for res_unit in self.res_block:
            x = res_unit(x)

        return x


class CSPResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_block, activation='mish'):
        super(CSPResBlock, self).__init__()
        self.zero_padding = nn.ZeroPad2d((1, 0, 1, 0))
        self.prev_conv = Conv(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=0, activation=activation)
        self.shot_cut_conv = Conv(out_channels // 2, out_channels // 2, kernel_size=1, stride=1, padding=0, activation=activation)
        self.main_conv = Conv(out_channels // 2, out_channels // 2, kernel_size=1, stride=1, padding=0, activation=activation)
        self.res_block = nn.ModuleList([ResUnit(out_channels // 2, activation=activation) for _ in range(num_block)])
        self.post_conv = Conv(out_channels // 2, out_channels // 2, kernel_size=1, stride=1, padding=0, activation=activation)

    def forward(self, x):
        x = self.zero_padding(x)
        x = self.prev_conv(x)
        short_cut = self.shot_cut_conv(x)
        x = self.main_conv(x)
        for res_unit in self.res_block:
            x = res_unit(x)
        x = self.post_conv(x)
        x = torch.cat([x, short_cut], dim=1)
        return x
