import os
import warnings

import torch

from Utils.gravel_dataset import GravelDataset
from Utils.read import get_anchors, get_dataset_list
from Utils.train import Train
from Utils.test import Test

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    dataset_dict = '../GravelDataset'

    cross_image_path = os.path.join(dataset_dict, 'Images+')
    single_image_path = os.path.join(dataset_dict, 'Images-')
    cross_image_train_path = os.path.join(dataset_dict, 'Images+.INPUT')
    single_image_train_path = os.path.join(dataset_dict, 'Images-.INPUT')
    images_path = [cross_image_path, single_image_path, cross_image_train_path, single_image_train_path]
    annotation_path = os.path.join(dataset_dict, 'Annotations')
    annotation_train_path = os.path.join(dataset_dict, 'Annotations.INPUT')
    annotations_path = [annotation_path, annotation_train_path]

    anchors = get_anchors('./docs/gravel_anchors.txt')

    train_data_list = get_dataset_list('../GravelDataset/train_list.txt')
    test_data_list = get_dataset_list('../GravelDataset/test_list.txt')

    input_size = 608
    s_scale, m_scale, l_scale = 8, 16, 32
    scale = [s_scale, m_scale, l_scale]
    batch_size = 8

    output_path = './output/parameters_gravel_new_baseline.pkl'

    train_dataset = GravelDataset(anchors, images_path, annotations_path, train_data_list, input_size, scale, device, train=True)
    train = Train(train_dataset, batch_size, device)
    train.run(epoch_num=120, warm_epoch_num=2, output_path=output_path)

    test_dataset = GravelDataset(anchors, images_path, annotations_path, test_data_list, input_size, scale, device, train=False)
    test = Test(test_dataset, output_path, device)
    test.run()
