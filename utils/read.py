import torch
import numpy as np


def get_dataset_list(dataset_list_path):
    f = open(dataset_list_path, 'r')
    dataset_list = [dataset_name.strip() for dataset_name in f.readlines()]
    f.close()
    return dataset_list


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32).reshape((3, 3, 2))
    return torch.tensor(anchors)
