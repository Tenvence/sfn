import numpy as np
import torch
import torch.nn as nn
import utils.utils as utils


class Loss(nn.Module):
    def __init__(self, anchors, input_size, iou_threshold):
        super(Loss, self).__init__()

        self.s_anchors, self.m_anchors, self.l_anchors = anchors
        self.s_scale, self.m_scale, self.l_scale = 8, 16, 32

        self.input_size = input_size
        self.s_output_size = self.input_size // self.s_scale
        self.m_output_size = self.input_size // self.m_scale
        self.l_output_size = self.input_size // self.l_scale

        self.max_box_per_scale = 150

    def forward(self, s_tensor, m_tensor, l_tensor, gt_boxes, gt_num):
        self.encode_gt_boxes(gt_boxes, gt_num)
        # s_giou_loss, s_conf_loss, s_prob_loss = utils.compute_loss(s_decode_output, s_output, s_tensor, s_coords, self.s_scale, self.iou_threshold)
        # m_giou_loss, m_conf_loss, m_prob_loss = utils.compute_loss(m_decode_output, m_output, m_tensor, m_coords, self.m_scale, self.iou_threshold)
        # l_giou_loss, l_conf_loss, l_prob_loss = utils.compute_loss(l_decode_output, l_output, l_tensor, l_coords, self.l_scale, self.iou_threshold)

        # s_loss = s_giou_loss + s_conf_loss + s_prob_loss
        # m_loss = m_giou_loss + m_conf_loss + m_prob_loss
        # l_loss = l_giou_loss + l_conf_loss + l_prob_loss

        return torch.mean(s_tensor)


class GiouLoss(nn.Module):
    def __init__(self):
        super(GiouLoss, self).__init__()


class ConfLoss(nn.Module):
    def __init__(self):
        super(ConfLoss, self).__init__()


class ProbLoss(nn.Module):
    def __init__(self):
        super(ProbLoss, self).__init__()
