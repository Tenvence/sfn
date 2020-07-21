import torch
import torch.nn as nn
import torch.nn.functional as f


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='leaky_relu'):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

        nn.init.normal_(self.conv.weight, std=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.activation == 'leaky_relu':
            x = nn.LeakyReLU(negative_slope=0.1)(x)
        elif self.activation == 'mish':
            x = x * (torch.tanh(f.softplus(x)))

        return x
