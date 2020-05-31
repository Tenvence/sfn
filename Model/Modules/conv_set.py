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
