import math
import torch

eps = 1e-10


def compute_iou(boxes1, boxes2, is_regularize=True):
    if is_regularize:
        boxes1 = regularize_boxes(boxes1)
        boxes2 = regularize_boxes(boxes2)

    inter_area, union_area, _, _ = compute_inter_union_area(boxes1, boxes2)
    iou = inter_area / (union_area + eps)

    return iou


def compute_giou(boxes1, boxes2):
    boxes1 = regularize_boxes(boxes1)
    boxes2 = regularize_boxes(boxes2)

    inter_area, union_area, enclose_area, _ = compute_inter_union_area(boxes1, boxes2)

    iou = inter_area / (union_area + eps)
    giou = iou - (enclose_area - union_area) / (enclose_area + eps)

    return giou


def compute_diou(boxes1, boxes2, with_iou=False):
    bbox_diag = torch.pow(boxes1[..., 0] - boxes2[..., 0], 2) + torch.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    boxes1 = regularize_boxes(boxes1)
    boxes2 = regularize_boxes(boxes2)

    inter_area, union_area, _, enclose_diag = compute_inter_union_area(boxes1, boxes2)
    iou = inter_area / (union_area + eps)
    diou = iou - bbox_diag / (enclose_diag + eps)

    if with_iou:
        return diou, iou
    else:
        return diou


def compute_ciou(boxes1, boxes2):
    v = (4 / math.pi) * torch.pow(torch.atan(boxes1[..., 2] / boxes1[..., 3]) - torch.atan(boxes2[..., 2] / boxes2[..., 3]), 2)
    diou, iou = compute_diou(boxes1, boxes2, with_iou=True)
    alpha = v / (1 - iou + v + eps)
    ciou = diou + alpha * v

    return ciou


def regularize_boxes(boxes):
    boxes = torch.cat([boxes[..., :2] - boxes[..., 2:] * 0.5, boxes[..., :2] + boxes[..., 2:] * 0.5], dim=-1)
    boxes = torch.cat([torch.min(boxes[..., :2], boxes[..., 2:]), torch.max(boxes[..., :2], boxes[..., 2:])], dim=-1)

    return boxes


def compute_inter_union_area(boxes1, boxes2):
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    zero = torch.zeros((1, 2))

    # Support GPU computation
    if boxes1.is_cuda:
        zero = zero.to(device=boxes1.device)

    inter_section = torch.max(right_down - left_up, zero)

    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose = torch.max(enclose_right_down - enclose_left_up, zero)
    enclose_diag = torch.pow(enclose[..., 0], 2) + torch.pow(enclose[..., 1], 2)
    enclose_area = enclose[..., 0] * enclose[..., 1]

    return inter_area, union_area, enclose_area, enclose_diag
