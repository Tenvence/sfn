import torch.nn as nn
from .out_decoder import OutDecoder


class YoloHead(nn.Module):
    def __init__(self, in_channels, out_channels, anchors, scale):
        super(YoloHead, self).__init__()

        self.output_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.out_decoder = OutDecoder(anchors, scale)

        nn.init.constant_(self.output_conv.bias, 0.)
        nn.init.normal_(self.output_conv.weight, std=0.01)

    def forward(self, x):
        x = self.output_conv(x)
        x = self.out_decoder(x)
        return x
