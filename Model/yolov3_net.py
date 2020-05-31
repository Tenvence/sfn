import torch.nn as nn

from Model.Backbone.dual_darknet53 import DualDarknet53
from Model.Modules.output_decoder import OutputDecoder
from Model.Neck.fpn import FPN


class Yolov3Net(nn.Module):
    def __init__(self, anchors):
        super(Yolov3Net, self).__init__()

        output_channels = len(anchors) * 5

        self.backbone = DualDarknet53()
        self.neck = FPN(output_channels)

        self.decode_l = OutputDecoder(anchors[2], scale=32)
        self.decode_m = OutputDecoder(anchors[1], scale=16)
        self.decode_s = OutputDecoder(anchors[0], scale=8)

    def forward(self, x1, x2):
        route_s, route_m, route_l = self.backbone(x1, x2)
        output_s, output_m, output_l = self.neck(route_s, route_m, route_l)

        output_s = self.decode_s(output_s)
        output_m = self.decode_m(output_m)
        output_l = self.decode_l(output_l)

        return output_s, output_m, output_l
