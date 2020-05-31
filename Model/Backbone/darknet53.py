import torch.nn as nn
from ..Modules.conv import Conv
from ..Modules.res_block import ResBlock


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv = Conv(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.res_block_1 = ResBlock(in_channels=32, out_channels=64, num_block=1)
        self.res_block_2 = ResBlock(in_channels=64, out_channels=128, num_block=2)
        self.res_block_8_1 = ResBlock(in_channels=128, out_channels=256, num_block=8)
        self.res_block_8_2 = ResBlock(in_channels=256, out_channels=512, num_block=8)
        self.res_block_4 = ResBlock(in_channels=512, out_channels=1024, num_block=4)

    def forward(self, x):
        x = self.conv(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_8_1(x)
        route_s = x
        x = self.res_block_8_2(x)
        route_m = x
        x = self.res_block_4(x)
        route_l = x

        return route_s, route_m, route_l
