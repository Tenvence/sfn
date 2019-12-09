import torch
import torch.nn as nn
import utils.utils as utils


class Loss(nn.Module):
    def __init__(self, anchor_num, cla_num, detection_scale, anchors, iou_threshold):
        super(Loss, self).__init__()
        self.anchor_num = anchor_num
        self.cla_num = cla_num
        self.iou_threshold = iou_threshold
        self.s_scale, self.m_scale, self.l_scale = detection_scale
        self.s_anchors, self.m_anchors, self.l_anchors = torch.tensor(anchors)

    def forward(self, s_output, m_output, l_output, s_tensor, m_tensor, l_tensor, s_coords, m_coords, l_coords):
        s_output = utils.transform_output(s_output, self.anchor_num, self.cla_num)
        m_output = utils.transform_output(m_output, self.anchor_num, self.cla_num)
        l_output = utils.transform_output(l_output, self.anchor_num, self.cla_num)

        s_decode_output = utils.decode_output_to_tensor(s_output, self.anchor_num, self.s_scale, self.s_anchors)
        m_decode_output = utils.decode_output_to_tensor(m_output, self.anchor_num, self.m_scale, self.m_anchors)
        l_decode_output = utils.decode_output_to_tensor(l_output, self.anchor_num, self.l_scale, self.l_anchors)

        s_giou_loss, s_conf_loss, s_prob_loss = utils.compute_loss(s_decode_output, s_output, s_tensor, s_coords, self.s_scale, self.iou_threshold)
        m_giou_loss, m_conf_loss, m_prob_loss = utils.compute_loss(m_decode_output, m_output, m_tensor, m_coords, self.m_scale, self.iou_threshold)
        l_giou_loss, l_conf_loss, l_prob_loss = utils.compute_loss(l_decode_output, l_output, l_tensor, l_coords, self.l_scale, self.iou_threshold)

        s_loss = s_giou_loss + s_conf_loss + s_prob_loss
        m_loss = m_giou_loss + m_conf_loss + m_prob_loss
        l_loss = l_giou_loss + l_conf_loss + l_prob_loss

        return s_loss + m_loss + l_loss
