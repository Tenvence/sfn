import torch
import torch.nn as nn

from ..Modules.conv import Conv
from ..Modules.conv_set import ConvSet


class FPN(nn.Module):
    def __init__(self, output_channels):
        super(FPN, self).__init__()

        self.conv_set_l = ConvSet(in_channels=1024, out_channels=512)
        self.conv_l = Conv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv_l_output = nn.Conv2d(in_channels=1024, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        self.conv_l_branch = Conv(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.conv_set_m = ConvSet(in_channels=768, out_channels=256)
        self.conv_m = Conv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_m_output = nn.Conv2d(in_channels=512, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        self.conv_m_branch = Conv(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.conv_set_s = ConvSet(in_channels=384, out_channels=128)
        self.conv_s = Conv(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_s_output = nn.Conv2d(in_channels=256, out_channels=output_channels, kernel_size=1, stride=1, padding=0)

        nn.init.constant_(self.conv_l_output.bias, 0.)
        nn.init.constant_(self.conv_m_output.bias, 0.)
        nn.init.constant_(self.conv_s_output.bias, 0.)

        nn.init.normal_(self.conv_l_output.weight, std=0.01)
        nn.init.normal_(self.conv_m_output.weight, std=0.01)
        nn.init.normal_(self.conv_s_output.weight, std=0.01)

    def forward(self, input_s, input_m, input_l):
        input_l = self.conv_set_l(input_l)

        output_l = self.conv_l(input_l)
        output_l = self.conv_l_output(output_l)

        branch_l = self.conv_l_branch(input_l)
        branch_l = nn.UpsamplingNearest2d(scale_factor=2)(branch_l)

        input_m = torch.cat([branch_l, input_m], dim=1)
        input_m = self.conv_set_m(input_m)

        output_m = self.conv_m(input_m)
        output_m = self.conv_m_output(output_m)

        branch_m = self.conv_m_branch(input_m)
        branch_m = nn.UpsamplingNearest2d(scale_factor=2)(branch_m)

        input_s = torch.cat([branch_m, input_s], dim=1)
        input_s = self.conv_set_s(input_s)

        output_s = self.conv_s(input_s)
        output_s = self.conv_s_output(output_s)

        return output_s, output_m, output_l
