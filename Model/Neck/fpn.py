import torch
import torch.nn as nn
from ..Modules.conv import Conv
from ..Modules.conv_set import ConvSet, SPPConvSet


class FPN(nn.Module):
    def __init__(self, l_in_channel, m_in_channel, s_in_channel, as_sub_module=False):
        super(FPN, self).__init__()

        self.as_sub_module = as_sub_module

        self.conv_set_l = ConvSet(in_channels=l_in_channel, out_channels=l_in_channel // 2)
        self.conv_l = Conv(in_channels=l_in_channel // 2, out_channels=l_in_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l_branch = Conv(in_channels=l_in_channel // 2, out_channels=l_in_channel // 4, kernel_size=1, stride=1, padding=0)

        self.conv_set_m = ConvSet(in_channels=m_in_channel + l_in_channel // 4, out_channels=m_in_channel // 2)
        self.conv_m = Conv(in_channels=m_in_channel // 2, out_channels=m_in_channel, kernel_size=3, stride=1, padding=1)
        self.conv_m_branch = Conv(in_channels=m_in_channel // 2, out_channels=m_in_channel // 4, kernel_size=1, stride=1, padding=0)

        self.conv_set_s = ConvSet(in_channels=s_in_channel + m_in_channel // 4, out_channels=s_in_channel // 2)
        self.conv_s = Conv(in_channels=s_in_channel // 2, out_channels=s_in_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, input_s, input_m, input_l):
        input_l = self.conv_set_l(input_l)

        branch_l = self.conv_l_branch(input_l)
        branch_l = nn.UpsamplingNearest2d(scale_factor=2)(branch_l)

        input_m = torch.cat([branch_l, input_m], dim=1)
        input_m = self.conv_set_m(input_m)

        branch_m = self.conv_m_branch(input_m)
        branch_m = nn.UpsamplingNearest2d(scale_factor=2)(branch_m)

        input_s = torch.cat([branch_m, input_s], dim=1)
        input_s = self.conv_set_s(input_s)

        if not self.as_sub_module:
            output_l = self.conv_l(input_l)
            output_m = self.conv_m(input_m)
            output_s = self.conv_s(input_s)
            return output_s, output_m, output_l
        else:
            return input_s, input_m, input_l
