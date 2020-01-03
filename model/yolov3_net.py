import torch
import torch.nn as nn
import model.module as module
import model.backbone as backbone


class Yolov3Net(nn.Module):
    def __init__(self, anchors):
        super(Yolov3Net, self).__init__()

        output_channels = len(anchors) * 5

        self.backbone = backbone.Darknet53()

        self.up_sample_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up_sample_m = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv_set_l = module.ConvolutionSetModule(in_channels=1024, out_channels=512)
        self.conv_l = module.ConvolutionModule(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv_l_output = nn.Conv2d(in_channels=1024, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        self.conv_l_branch = module.ConvolutionModule(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.conv_set_m = module.ConvolutionSetModule(in_channels=768, out_channels=256)
        self.conv_m = module.ConvolutionModule(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_m_output = nn.Conv2d(in_channels=512, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        self.conv_m_branch = module.ConvolutionModule(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.conv_set_s = module.ConvolutionSetModule(in_channels=384, out_channels=128)
        self.conv_s = module.ConvolutionModule(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_s_output = nn.Conv2d(in_channels=256, out_channels=output_channels, kernel_size=1, stride=1, padding=0)

        self.decode_l = module.DecodeModule(anchors[2], scale=32)
        self.decode_m = module.DecodeModule(anchors[1], scale=16)
        self.decode_s = module.DecodeModule(anchors[0], scale=8)

        nn.init.constant_(self.conv_l_output.bias, 0.)
        nn.init.constant_(self.conv_m_output.bias, 0.)
        nn.init.constant_(self.conv_s_output.bias, 0.)

        nn.init.normal_(self.conv_l_output.weight, std=0.01)
        nn.init.normal_(self.conv_m_output.weight, std=0.01)
        nn.init.normal_(self.conv_s_output.weight, std=0.01)

    def forward(self, x):
        route_s, route_m, route_l = self.backbone(x)

        route_l = self.conv_set_l(route_l)

        l_branch_output = self.conv_l(route_l)
        l_branch_output = self.conv_l_output(l_branch_output)

        route_l = self.conv_l_branch(route_l)
        route_l = self.up_sample_1(route_l)

        route_m = torch.cat((route_l, route_m), dim=1)
        route_m = self.conv_set_m(route_m)

        m_branch_output = self.conv_m(route_m)
        m_branch_output = self.conv_m_output(m_branch_output)

        route_m = self.conv_m_branch(route_m)
        route_m = self.up_sample_m(route_m)

        route_s = torch.cat((route_m, route_s), dim=1)
        route_s = self.conv_set_s(route_s)

        s_branch_output = self.conv_s(route_s)
        s_branch_output = self.conv_s_output(s_branch_output)

        s_branch_output = self.decode_s(s_branch_output)
        m_branch_output = self.decode_m(m_branch_output)
        l_branch_output = self.decode_l(l_branch_output)

        return s_branch_output, m_branch_output, l_branch_output
