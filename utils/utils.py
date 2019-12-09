import os
import random
import torch
import torch.nn.functional as F
import numpy as np


def spilt_dataset(annotations_path, train_list_path, test_list_path, train_ratio=0.7):
    example_name_list = os.listdir(annotations_path)
    example_num = len(example_name_list)
    train_num = int(example_num * train_ratio)
    train_list = random.sample(range(example_num), train_num)

    f_train = open(train_list_path, 'w')
    f_test = open(test_list_path, 'w')

    for i in range(example_num):
        example_name = example_name_list[i][:-4] + '\n'
        if i in train_list:
            f_train.write(example_name)
        else:
            f_test.write(example_name)

    f_train.close()
    f_test.close()


def generate_anchor_boxes():
    pass


def smooth_onehot(num, idx, delta):
    onehot = np.zeros(num, dtype=np.float)
    onehot[idx] = 1.0
    uniform_distribution = np.full(num, 1.0 / num)

    return onehot * (1 - delta) + delta * uniform_distribution


def compute_iou_np(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / union_area


def compute_iou(boxes1, boxes2):
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    boxes1 = torch.cat([boxes1[..., :2].min(boxes1[..., 2:]), boxes1[..., :2].max(boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2].min(boxes2[..., 2:]), boxes2[..., :2].max(boxes2[..., 2:])], dim=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = boxes1[..., :2].max(boxes2[..., :2])
    right_down = boxes1[..., 2:].min(boxes2[..., 2:])

    inter_section = (right_down - left_up).max(torch.zeros(right_down.shape))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area, boxes1, boxes2, union_area


def compute_giou(boxes1, boxes2):
    iou, boxes1, boxes2, union_area = compute_iou(boxes1, boxes2)

    enclose_left_up = boxes1[..., :2].min(boxes2[..., :2])
    enclose_right_down = boxes1[..., 2:].max(boxes2[..., 2:])
    enclose = (enclose_right_down - enclose_left_up).max(torch.zeros(enclose_left_up.shape))
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_vectors_of_one_bbox(scale_coord, bbox_coord, cla_conf, anchor_size, tensor, coords, anchor_num=3, iou_threshold=0.3):
    anchor_coord = np.zeros((anchor_num, 4))
    anchor_coord[:, 0:2] = np.floor(scale_coord[0:2]).astype(np.int32) + 0.5  # 中心坐标是 bbox 的中心点所在的 grid cell 的中心
    anchor_coord[:, 2:4] = anchor_size  # 尺寸为 anchor box 的尺寸

    iou = compute_iou_np(scale_coord[np.newaxis, :], anchor_coord)  # 计算 ground truth bbox 和该尺度下 anchor box 的IoU，大小 1×3
    iou_mask = iou > iou_threshold  # 判断IoU是否大于阈值，如果大于阈值，则认为该 anchor box 可以检测 ground truth box。大小 1×3

    is_positive_example = False

    # 任意一个anchor box的IoU大于阈值, 计算 tensor 该处的值
    if np.any(iou_mask):
        tensor, coords = compute_valid_vectors_of_one_bbox(scale_coord, bbox_coord, iou_mask, cla_conf, tensor, coords)
        is_positive_example = True  # 有符合IoU条件的 anchor box

    return iou, tensor, coords, is_positive_example


def compute_valid_vectors_of_one_bbox(scale_coord, bbox_coord, iou_mask, cla_conf, tensor, coords):
    x_idx, y_idx = np.floor(scale_coord[0:2]).astype(np.int32)  # grid cell 的坐标

    # 减少数据的敏感性
    x_idx, y_idx = abs(x_idx), abs(y_idx)
    if y_idx >= tensor.shape[1]:
        y_idx = tensor.shape[1] - 1
    if x_idx >= tensor.shape[0]:
        x_idx = tensor.shape[0] - 1

    tensor[y_idx, x_idx, iou_mask, :] = 0

    tensor[y_idx, x_idx, iou_mask, 0:4] = bbox_coord  # 坐标
    tensor[y_idx, x_idx, iou_mask, 4:5] = 1.0  # 置信度
    tensor[y_idx, x_idx, iou_mask, 5:] = cla_conf  # 分类分布

    coords[0][coords[1], :] = bbox_coord
    coords[1] += 1

    return tensor, coords


def transform_output(output, anchor_num, cla_num):
    batch_size = output.shape[0]
    output_size = output.shape[3]

    # PyTorch 的网络输出尺寸为 [B×C×W×H]，需要将其转换为 [B×W×H×C]，再将通道分解
    output = output.permute(0, 2, 3, 1).reshape((batch_size, output_size, output_size, anchor_num, 5 + cla_num))

    return output


def decode_output_to_tensor(output, anchor_num, scale, anchors):
    batch_size = output.shape[0]
    output_size = output.shape[1]

    output_dx_dy = output[:, :, :, :, 0:2]  # 预测的 x 和 y 的偏移
    output_dw_dh = output[:, :, :, :, 2:4]  # 预测的 w 和 h 的偏移
    output_conf = output[:, :, :, :, 4:5]  # 预测的置信度
    output_prob = output[:, :, :, :, 5:]  # 预测的各类别的概率

    y = torch.Tensor.repeat(torch.arange(output_size)[:, np.newaxis], [1, output_size])  # grid cell 的 y 坐标
    x = torch.Tensor.repeat(torch.arange(output_size)[np.newaxis, :], [output_size, 1])  # grid cell 的 x 坐标

    # xy_grid 的尺寸为 output_size × output_size × 2，其中的每一个2长度的向量都代表该处的 grid cell 坐标
    xy_grid = torch.cat([x[:, :, np.newaxis], y[:, :, np.newaxis]], dim=-1)
    xy_grid = torch.Tensor.repeat(xy_grid[np.newaxis, :, :, np.newaxis, :], [batch_size, 1, 1, anchor_num, 1]).float()

    output_xy = (torch.sigmoid(output_dx_dy) + xy_grid) * scale
    output_wh = (torch.exp(output_dw_dh) * anchors) * scale
    output_coord = torch.cat([output_xy, output_wh], dim=-1)
    output_conf = torch.sigmoid(output_conf)
    output_prob = torch.sigmoid(output_prob)

    return torch.cat([output_coord, output_conf, output_prob], dim=-1)


def compute_loss(decode_output, output, tensor, coords, scale, iou_threshold):
    output_size = output.shape[1]

    output_conf = output[:, :, :, :, 4:5]
    output_prob = output[:, :, :, :, 5:]

    decode_output_coord = decode_output[:, :, :, :, 0:4]
    decode_output_conf = decode_output[:, :, :, :, 4:5]

    tensor_coord = tensor[:, :, :, :, 0:4]
    tensor_conf = tensor[:, :, :, :, 4:5]
    tensor_prob = tensor[:, :, :, :, 5:]

    giou = torch.unsqueeze(compute_giou(decode_output_coord, tensor_coord), dim=-1)
    input_size = torch.tensor(output_size * scale, dtype=torch.float32)
    bbox_scale = torch.div(tensor_coord[:, :, :, :, 2:3] * tensor_coord[:, :, :, :, 3:4], torch.pow(input_size, 2))
    giou_loss_scale = torch.sub(2.0, bbox_scale)
    giou_loss = tensor_conf * giou_loss_scale * torch.sub(1, giou)
    # print(giou_loss.shape, giou_loss.dtype)

    iou = compute_iou(decode_output_coord[:, :, :, :, np.newaxis, :], coords[:, np.newaxis, np.newaxis, np.newaxis, :, :])[0]
    max_iou = torch.unsqueeze(torch.max(iou, dim=-1)[0], dim=-1)
    background_conf = torch.sub(1.0, tensor_conf) * torch.lt(max_iou, torch.tensor(iou_threshold)).float()

    conf_focal = torch.pow(tensor_conf - decode_output_conf, 2)
    conf_loss = conf_focal * (
            tensor_conf * F.binary_cross_entropy_with_logits(input=torch.sigmoid(output_conf), target=tensor_conf)
            +
            background_conf * F.binary_cross_entropy_with_logits(input=torch.sigmoid(output_conf), target=tensor_conf)
    )

    prob_loss = tensor_conf * F.binary_cross_entropy_with_logits(input=torch.sigmoid(output_prob), target=tensor_prob)

    # print(giou_loss.shape, conf_loss.shape, prob_loss.shape)

    giou_loss = torch.mean(torch.sum(giou_loss, [1, 2, 3, 4]))
    conf_loss = torch.mean(torch.sum(conf_loss, [1, 2, 3, 4]))
    prob_loss = torch.mean(torch.sum(prob_loss, [1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss
