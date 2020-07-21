import torch
import torch.nn as nn

from ..Modules.conv import Conv
from ..Modules.conv_set import ConvSet
from ..Neck.fpn import FPN


class PAN(nn.Module):
    def __init__(self, l_in_channel, m_in_channel, s_in_channel):
        super(PAN, self).__init__()
        self.fpn = FPN(l_in_channel, m_in_channel, s_in_channel, as_sub_module=True)

        l_in_channel = l_in_channel // 2
        m_in_channel = m_in_channel // 2
        s_in_channel = s_in_channel // 2

        self.zero_padding = nn.ZeroPad2d((1, 0, 1, 0))

        self.s_down_sample = Conv(in_channels=s_in_channel, out_channels=s_in_channel, kernel_size=3, stride=2, padding=0)
        self.s_output = Conv(in_channels=s_in_channel, out_channels=s_in_channel * 2, kernel_size=3, stride=1, padding=1)

        self.m_conv_set = ConvSet(in_channels=s_in_channel + m_in_channel, out_channels=m_in_channel)
        self.m_down_sample = Conv(in_channels=m_in_channel, out_channels=m_in_channel, kernel_size=3, stride=2, padding=0)
        self.m_output = Conv(in_channels=m_in_channel, out_channels=m_in_channel * 2, kernel_size=3, stride=1, padding=1)

        self.l_conv_set = ConvSet(in_channels=m_in_channel + l_in_channel, out_channels=l_in_channel)
        self.l_output = Conv(in_channels=l_in_channel, out_channels=l_in_channel * 2, kernel_size=3, stride=1, padding=1)

        # self.conv_set_s = ConvSet(in_channels=s_in_channel, out_channels=s_in_channel // 2)
        # self.conv_s = Conv(in_channels=s_in_channel // 2, out_channels=s_in_channel, kernel_size=3, stride=1, padding=1)
        # self.conv_s_branch = Conv(in_channels=s_in_channel // 2, out_channels=s_in_channel // 4, kernel_size=1, stride=1, padding=0)

        # self.conv_set_m = ConvSet(in_channels=m_in_channel + s_in_channel // 4, out_channels=m_in_channel // 2)
        # self.conv_m = Conv(in_channels=m_in_channel // 2, out_channels=m_in_channel, kernel_size=3, stride=1, padding=1)
        # self.conv_m_branch = Conv(in_channels=m_in_channel // 2, out_channels=m_in_channel // 4, kernel_size=1, stride=1, padding=0)

        # self.conv_set_l = ConvSet(in_channels=l_in_channel + m_in_channel // 4, out_channels=l_in_channel // 2)
        # self.conv_l = Conv(in_channels=l_in_channel // 2, out_channels=l_in_channel, kernel_size=3, stride=1, padding=1)

        # self.down_sample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, input_s, input_m, input_l):
        input_s, input_m, input_l = self.fpn(input_s, input_m, input_l)
        # print(input_s.shape, input_m.shape, input_l.shape)

        output_s = self.s_output(input_s)

        branch_s = self.zero_padding(input_s)
        branch_s = self.s_down_sample(branch_s)

        input_m = torch.cat([branch_s, input_m], dim=1)
        input_m = self.m_conv_set(input_m)

        output_m = self.m_output(input_m)

        branch_m = self.zero_padding(input_m)
        branch_m = self.m_down_sample(branch_m)

        input_l = torch.cat([branch_m, input_l], dim=1)
        input_l = self.l_conv_set(input_l)

        output_l = self.l_output(input_l)

        return output_s, output_m, output_l
