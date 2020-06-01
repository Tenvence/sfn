import torch
import torch.nn as nn

from ..Modules.conv import Conv
from ..Neck.fpn import FPN


class PAN(nn.Module):
    def __init__(self, output_channels):
        super(PAN, self).__init__()
        self.fpn = FPN()
        # self.conv_l = Conv(in_channels=)

    def forward(self, input_s, input_m, input_l):
        input_s, input_m, input_l = self.fpn(input_s, input_m, input_l)
