import torch.nn as nn
from .conv import Conv
from .res_unit import ResUnit


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_block):
        super(ResBlock, self).__init__()
        self.zero_padding = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv = Conv(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.res_block = nn.ModuleList([ResUnit(out_channels) for _ in range(num_block)])

    def forward(self, x):
        x = self.zero_padding(x)
        x = self.conv(x)
        for res_unit in self.res_block:
            x = res_unit(x)

        return x
