import torch
import torch.nn as nn
from ..Modules.conv import Conv
from ..Modules.res_block import ResBlock, CSPResBlock


class DualDarknet53(nn.Module):
    def __init__(self, use_csp=False, activation='leaky_relu'):
        super(DualDarknet53, self).__init__()

        if use_csp:
            res_block = CSPResBlock
        else:
            res_block = ResBlock

        self.conv_x1 = Conv(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, activation=activation)
        self.conv_x2 = Conv(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, activation=activation)

        self.res_block_1_x1 = res_block(in_channels=32, out_channels=64, num_block=1, activation=activation)
        self.res_block_1_x2 = res_block(in_channels=32, out_channels=64, num_block=1, activation=activation)

        self.res_block_2_x1 = res_block(in_channels=64, out_channels=128, num_block=2, activation=activation)
        self.res_block_2_x2 = res_block(in_channels=64, out_channels=128, num_block=2, activation=activation)

        self.res_block_8_x1 = res_block(in_channels=128, out_channels=256, num_block=8, activation=activation)
        self.res_block_8_x2 = res_block(in_channels=128, out_channels=256, num_block=8, activation=activation)

        self.conv_concat = Conv(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, activation=activation)

        self.res_block_8_2 = res_block(in_channels=256, out_channels=512, num_block=8, activation=activation)
        self.res_block_4 = res_block(in_channels=512, out_channels=1024, num_block=4, activation=activation)

    def forward(self, x1, x2):
        x1 = self.conv_x1(x1)
        x2 = self.conv_x2(x2)

        x1 = self.res_block_1_x1(x1)
        x2 = self.res_block_1_x2(x2)

        x1 = self.res_block_2_x1(x1)
        x2 = self.res_block_2_x2(x2)

        x1 = self.res_block_8_x1(x1)
        x2 = self.res_block_8_x2(x2)

        x = torch.cat([x1, x2], dim=1)
        x = self.conv_concat(x)

        route_s = x
        x = self.res_block_8_2(x)
        route_m = x
        x = self.res_block_4(x)
        route_l = x

        return route_s, route_m, route_l
