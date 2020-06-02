import os
import warnings

import numpy as np
import torch.multiprocessing

from Utils.gravel_dataset import GravelDataset
from Utils.read import get_anchors, get_dataset_list
from Utils.train import Train
from Utils.test import Test

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # torch.multiprocessing.set_start_method('spawn')

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

    for iou_thresh in np.arange(0.1, 0.1, 1.0):
        output_dict = './output/baseline_train_iou=%.2f' % iou_thresh
        model_file = output_dict + '/model.pkl'

        if not os.path.exists(output_dict):
            os.mkdir(output_dict)

        train_dataset = GravelDataset(anchors, images_path, annotations_path, train_data_list, input_size, scale, device, train=True)
        train = Train(train_dataset, batch_size, device, iou_thresh=iou_thresh)
        while not train.run(epoch_num=100, warm_epoch_num=2, output_model_path=model_file):
            print('Loss is nan! Training will restart!')

        test_dataset = GravelDataset(anchors, images_path, annotations_path, test_data_list, input_size, scale, device, train=False)
        test = Test(test_dataset, model_file, device)
        test.run(output_dict)
