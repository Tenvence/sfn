import torch
import numpy as np


def get_dataset_list(dataset_list_path):
    f = open(dataset_list_path, 'r')
    dataset_list = [dataset_name.strip() for dataset_name in f.readlines()]
    f.close()
    return dataset_list


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors_arr = np.array([line.replace('\n', '').split(' ') for line in f.readlines()])

    anchor_tensor = np.zeros((anchors_arr.shape[0], anchors_arr.shape[1], 2))

    for scale_idx in range(0, anchors_arr.shape[0]):
        for anchor_idx in range(0, anchors_arr.shape[1]):
            [w, h] = anchors_arr[scale_idx, anchor_idx].split(',')
            anchor_tensor[scale_idx, anchor_idx, :] = np.array((float(w), float(h)))

    anchor_tensor = torch.tensor(anchor_tensor).float()

    return anchor_tensor
