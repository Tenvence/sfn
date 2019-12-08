import os
import random
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


def compute_iou(boxes1, boxes2):
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


def compute_vectors_of_one_bbox(scale_coord, bbox_coord, cla_conf, anchor_size, tensor, coords, anchor_num=3, iou_threshold=0.3):
    anchor_coord = np.zeros((anchor_num, 4))
    anchor_coord[:, 0:2] = np.floor(scale_coord[0:2]).astype(np.int32) + 0.5  # 中心坐标是 bbox 的中心点所在的 grid cell 的中心
    anchor_coord[:, 2:4] = anchor_size  # 尺寸为 anchor box 的尺寸

    iou = compute_iou(scale_coord[np.newaxis, :], anchor_coord)  # 计算 ground truth bbox 和该尺度下 anchor box 的IoU，大小 1×3
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

    coords.append(bbox_coord)

    return tensor, coords
