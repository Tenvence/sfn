import torch.nn as nn

from Model.Backbone.dual_darknet53 import DualDarknet53
from Model.Neck.fpn import FPN
from Model.Modules.yolo_head import YoloHead


class Yolov3Net(nn.Module):
    def __init__(self, anchors):
        super(Yolov3Net, self).__init__()

        out_channels = len(anchors) * 5

        self.backbone = DualDarknet53()
        self.neck = FPN()

        self.head_s = YoloHead(in_channels=256, out_channels=out_channels, anchors=anchors[0], scale=8)
        self.head_m = YoloHead(in_channels=512, out_channels=out_channels, anchors=anchors[1], scale=16)
        self.head_l = YoloHead(in_channels=1024, out_channels=out_channels, anchors=anchors[2], scale=32)

    def forward(self, x1, x2):
        route_s, route_m, route_l = self.backbone(x1, x2)
        output_s, output_m, output_l = self.neck(route_s, route_m, route_l)

        output_s = self.head_s(output_s)
        output_m = self.head_m(output_m)
        output_l = self.head_l(output_l)

        return output_s, output_m, output_l
