import torch
import torch.nn as nn

from Model.Backbone.dual_darknet53 import DualDarknet53
from Model.Backbone.darknet53 import Darknet53
from Model.Backbone.dual_resnet import dual_resnet50, dual_resnet101, dual_resnext50, dual_resnext101
from Model.Neck.fpn import FPN
from Model.Neck.pan import PAN
from Model.Modules.yolo_head import YoloHead


class Net(nn.Module):
    def __init__(self, anchors):
        super(Net, self).__init__()

        out_channels = len(anchors) * 5

        self.backbone = DualDarknet53(use_csp=False, activation='leaky_relu')
        # self.backbone = Darknet53()
        self.neck = FPN(1024, 512, 256)
        # self.neck = PAN(1024, 512, 256)
        self.head_s = YoloHead(in_channels=256, out_channels=out_channels, anchors=anchors[0], scale=8)
        self.head_m = YoloHead(in_channels=512, out_channels=out_channels, anchors=anchors[1], scale=16)
        self.head_l = YoloHead(in_channels=1024, out_channels=out_channels, anchors=anchors[2], scale=32)

        # self.backbone = dual_resnext101()
        # self.neck = FPN(2048, 1024, 512)
        # self.head_s = YoloHead(in_channels=512, out_channels=out_channels, anchors=anchors[0], scale=8)
        # self.head_m = YoloHead(in_channels=1024, out_channels=out_channels, anchors=anchors[1], scale=16)
        # self.head_l = YoloHead(in_channels=2048, out_channels=out_channels, anchors=anchors[2], scale=32)

    def forward(self, x1, x2):
        route_s, route_m, route_l = self.backbone(x1, x2)
        # route_s, route_m, route_l = self.backbone((x1 + x2) / 2.)
        output_s, output_m, output_l = self.neck(route_s, route_m, route_l)

        output_s = self.head_s(output_s)
        output_m = self.head_m(output_m)
        output_l = self.head_l(output_l)

        return output_s, output_m, output_l
