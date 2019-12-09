import torch.nn as nn
import net.module as module


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv = module.ConvolutionModule(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.res_block_1 = module.ResidualBlockModule(in_channels=32, out_channels=64, num_block=1)
        self.res_block_2 = module.ResidualBlockModule(in_channels=64, out_channels=128, num_block=2)
        self.res_block_8_1 = module.ResidualBlockModule(in_channels=128, out_channels=256, num_block=8)
        self.res_block_8_2 = module.ResidualBlockModule(in_channels=256, out_channels=512, num_block=8)
        self.res_block_4 = module.ResidualBlockModule(in_channels=512, out_channels=1024, num_block=4)

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
