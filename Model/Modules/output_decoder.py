import torch
import torch.nn as nn
import numpy as np


class OutputDecoder(nn.Module):
    def __init__(self, anchors, scale):
        super(OutputDecoder, self).__init__()
        self.anchors = anchors
        self.scale = scale

    def forward(self, output):
        batch_size = output.shape[0]
        output_size = output.shape[2]
        anchor_num = self.anchors.shape[0]

        # The output size is [B*C*W*H] which needs to be converted to [B*W*H*C] for channel division.
        output = output.permute(0, 2, 3, 1).view((batch_size, output_size, output_size, anchor_num, 5))

        output_dx_dy = output[:, :, :, :, 0:2]  # predicted offset of x and y
        output_dw_dh = output[:, :, :, :, 2:4]  # predicted offset of w and h
        output_conf = output[:, :, :, :, 4:5]  # predicted confident

        y = torch.Tensor.repeat(torch.arange(output_size)[:, np.newaxis], [1, output_size]).float()  # grid cell 的 y 坐标
        x = torch.Tensor.repeat(torch.arange(output_size)[np.newaxis, :], [output_size, 1]).float()  # grid cell 的 x 坐标

        # concatenate x and y together to build a [S*S*2] gird cell
        xy_grid = torch.cat([x[:, :, np.newaxis], y[:, :, np.newaxis]], dim=-1)
        xy_grid = torch.Tensor.repeat(xy_grid[np.newaxis, :, :, np.newaxis, :], [batch_size, 1, 1, anchor_num, 1])

        # Support GPU computation
        if output.is_cuda:
            xy_grid = xy_grid.to(output.device)
            self.anchors = self.anchors.to(output.device)

        output_xy = (torch.sigmoid(output_dx_dy) + xy_grid) * self.scale
        output_wh = (torch.exp(output_dw_dh) * self.anchors) * self.scale
        output_conf = torch.sigmoid(output_conf)

        return torch.cat([output_xy, output_wh, output_conf], dim=-1)
