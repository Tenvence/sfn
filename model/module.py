import numpy as np
import torch
import torch.nn as nn


class ConvolutionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        nn.init.normal_(self.conv.weight, std=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)

        return x


class ConvolutionSetModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionSetModule, self).__init__()
        self.conv_1 = ConvolutionModule(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_2 = ConvolutionModule(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv_3 = ConvolutionModule(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_4 = ConvolutionModule(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv_5 = ConvolutionModule(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)

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
        self.zero_padding = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv = ConvolutionModule(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.res_block = nn.ModuleList([ResidualUnitModule(out_channels) for _ in range(num_block)])

    def forward(self, x):
        x = self.zero_padding(x)
        x = self.conv(x)
        for res_unit in self.res_block:
            x = res_unit(x)

        return x


class DecodeModule(nn.Module):
    def __init__(self, anchors, scale):
        super(DecodeModule, self).__init__()
        self.anchors = anchors
        self.scale = scale

    def forward(self, output):
        batch_size = output.shape[0]
        output_size = output.shape[2]
        anchor_num = self.anchors.shape[0]

        # PyTorch 的网络输出尺寸为 [B×C×W×H]，需要将其转换为 [B×W×H×C]，再将通道分解
        output = output.permute(0, 2, 3, 1).view((batch_size, output_size, output_size, anchor_num, 5))

        output_dx_dy = output[:, :, :, :, 0:2]  # 预测的 x 和 y 的偏移
        output_dw_dh = output[:, :, :, :, 2:4]  # 预测的 w 和 h 的偏移
        output_conf = output[:, :, :, :, 4:5]  # 预测的置信度

        y = torch.Tensor.repeat(torch.arange(output_size)[:, np.newaxis], [1, output_size]).float()  # grid cell 的 y 坐标
        x = torch.Tensor.repeat(torch.arange(output_size)[np.newaxis, :], [output_size, 1]).float()  # grid cell 的 x 坐标

        # xy_grid 的尺寸为 (output_size)×(output_size)×2，其中的每一个2长度的向量都代表该处的 grid cell 坐标
        xy_grid = torch.cat([x[:, :, np.newaxis], y[:, :, np.newaxis]], dim=-1)
        xy_grid = torch.Tensor.repeat(xy_grid[np.newaxis, :, :, np.newaxis, :], [batch_size, 1, 1, anchor_num, 1])

        # Support GPU computation
        if output.is_cuda:
            xy_grid = xy_grid.to(output.device)
            self.anchors = self.anchors.to(output.device)

        output_xy = (torch.sigmoid(output_dx_dy) + xy_grid) * self.scale
        output_wh = (torch.exp(output_dw_dh) * self.anchors) * self.scale
        output_conf = torch.sigmoid(output_conf)

        return torch.cat([output_xy, output_wh, output_conf], dim=-1)
