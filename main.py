import os
import warnings

import torch
import torch.backends.cudnn

from Utils.gravel_dataset import GravelDataset
from Utils.read import get_anchors, get_dataset_list
from Utils.test import Test
from Utils.train import Train


def __main__():
    warnings.filterwarnings('ignore')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    dataset_dict = '../../DataSet/GravelDataset'

    cross_image_path = os.path.join(dataset_dict, 'Images+')
    single_image_path = os.path.join(dataset_dict, 'Images-')
    images_path = [cross_image_path, single_image_path]
    annotation_path = os.path.join(dataset_dict, 'Annotations')

    anchors = get_anchors('./docs/gravel_anchors.txt')

    train_data_list = get_dataset_list(os.path.join(dataset_dict, 'train_list.txt'))
    test_data_list = get_dataset_list(os.path.join(dataset_dict, 'test_list.txt'))

    s_scale, m_scale, l_scale = 8, 16, 32
    scale = [s_scale, m_scale, l_scale]
    batch_size = 8
    train_iou_thresh = 0.5

    epoch_num = 200
    warm_epoch_num = 10

    name = 'ciou-loss'
    output_dict = os.path.join('./output', name)

    if not os.path.exists(output_dict):
        os.mkdir(output_dict)

    train_size = 608
    test_size = 608

    train_dataset = GravelDataset(anchors, images_path, annotation_path, train_data_list, train_size, scale, device, train=True)
    test_dataset = GravelDataset(anchors, images_path, annotation_path, test_data_list, test_size, scale, device, train=False)

    train_state = False
    while not train_state:
        train = Train(train_dataset, batch_size, verify_dataset=test_dataset, device=device, iou_thresh=train_iou_thresh, name=name)
        train_state = train.run(epoch_num, warm_epoch_num, output_dict)

    test = Test(test_dataset, os.path.join(output_dict, 'model.pkl'), device=device)
    test.run(output_dict)


if __name__ == '__main__':
    __main__()
