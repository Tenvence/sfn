import torch.nn as nn


class ConvolutionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)

        return x


class ResidualUnitModule(nn.Module):
    def __init__(self, in_channels):
        super(ResidualUnitModule, self).__init__()
        self.conv_1 = ConvolutionModule(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)
        self.conv_2 = ConvolutionModule(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv = self.conv_1(x)
        conv = self.conv_2(conv)
        x = x + conv

        return x


class ResidualBlockModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_block):
        super(ResidualBlockModule, self).__init__()
        self.num_block = num_block
        self.zero_padding = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv = ConvolutionModule(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.res_block = []
        for i in range(num_block):
            self.res_block.append(ResidualUnitModule(out_channels))

    def forward(self, x):
        x = self.zero_padding(x)
        x = self.conv(x)
        for res_unit in self.res_block:
            x = res_unit(x)

        return x
