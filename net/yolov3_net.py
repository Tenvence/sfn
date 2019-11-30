import torch
import torch.nn as nn
import net.module as module
import net.backbone as backbone


class Yolov3Net(nn.Module):
    def __init__(self, num_class):
        super(Yolov3Net, self).__init__()

        output_channels = 3 * (num_class + 5)

        self.backbone = backbone.Darknet53()
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv_set_lobj = module.ConvolutionSetModule(in_channels=1024, out_channels=512)
        self.conv_lobj = module.ConvolutionModule(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv_lobj_output = nn.Conv2d(in_channels=1024, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        self.conv_lobj_branch = module.ConvolutionModule(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.conv_set_mobj = module.ConvolutionSetModule(in_channels=768, out_channels=256)
        self.conv_mobj = module.ConvolutionModule(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_mobj_output = nn.Conv2d(in_channels=512, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        self.conv_mobj_branch = module.ConvolutionModule(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.conv_set_sobj = module.ConvolutionSetModule(in_channels=384, out_channels=128)
        self.conv_sobj = module.ConvolutionModule(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.conv_sobj_output = nn.Conv2d(in_channels=256, out_channels=output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        route_1, route_2, route_3 = self.backbone(x)

        route_3 = self.conv_set_lobj(route_3)

        lobj_branch_output = self.conv_lobj(route_3)
        lobj_branch_output = self.conv_lobj_output(lobj_branch_output)

        route_3 = self.conv_lobj_branch(route_3)
        route_3 = self.up_sample(route_3)

        route_2 = torch.cat((route_3, route_2), dim=1)
        route_2 = self.conv_set_mobj(route_2)

        mobj_branch_output = self.conv_mobj(route_2)
        mobj_branch_output = self.conv_mobj_output(mobj_branch_output)

        route_2 = self.conv_mobj_branch(route_2)
        route_2 = self.up_sample(route_2)

        route_1 = torch.cat((route_2, route_1), dim=1)
        route_1 = self.conv_set_sobj(route_1)

        sobj_branch_output = self.conv_sobj(route_1)
        sobj_branch_output = self.conv_sobj_output(sobj_branch_output)

        return lobj_branch_output, mobj_branch_output, sobj_branch_output

    def initialize_weight(self):
        idx = 0
        for m in self.modules():
            idx += 1
        print(idx)


class A:
    def a(self):
        pass


class B(A):
    def b(self):
        pass
