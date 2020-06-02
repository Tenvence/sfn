import torch
import torch.nn as nn

from ..Modules.conv import Conv
from ..Neck.fpn import FPN


class PAN(nn.Module):
    def __init__(self):
        super(PAN, self).__init__()
        self.fpn = FPN()

        self.conv_l = Conv(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.branch_conv_l = Conv(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.conv_m = Conv(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.branch_conv_m = Conv(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.conv_s = Conv(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)

    def forward(self, input_s, input_m, input_l):
        input_s, input_m, input_l = self.fpn(input_s, input_m, input_l)

        output_l = self.conv_l(input_l)
        branch_l = self.branch_conv_l(output_l)
        branch_l = nn.UpsamplingNearest2d(scale_factor=2)(branch_l)

        input_m = torch.cat([input_m, branch_l], dim=1)
        output_m = self.conv_m(input_m)
        branch_m = self.branch_conv_m(output_m)
        branch_m = nn.UpsamplingNearest2d(scale_factor=2)(branch_m)

        input_s = torch.cat([input_s, branch_m], dim=1)
        output_s = self.conv_s(input_s)

        return output_s, output_m, output_l
