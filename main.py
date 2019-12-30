import sys
import torch

import utils.config as cfg
from utils.gravel_dataset import GravelDataset
from utils.read import get_anchors, get_dataset_list
from utils.train import Train
from utils.test import Test

if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    cross_image_path = cfg.CROSSED_IMAGE_PATH
    single_image_path = cfg.SINGLE_IMAGE_PATH
    annotation_path = cfg.ANNOTATIONS_PATH

    anchors = get_anchors(cfg.ANCHOR_FILE_PATH)
    train_data_list = get_dataset_list(cfg.TRAIN_LIST_PATH)
    test_data_list = get_dataset_list(cfg.TEST_LIST_PATH)

    input_size = 608
    s_scale, m_scale, l_scale = 8, 16, 32
    scale = [s_scale, m_scale, l_scale]
    batch_size = 2

    train_dataset = GravelDataset(anchors, cross_image_path, single_image_path, annotation_path, test_data_list, input_size, scale, device)
    train = Train(anchors, train_dataset, input_size, batch_size, device)
    train.run(epoch_num=30, warm_epoch_num=2, output_path='output/parameters.pkl')

    # test_dataset = GravelDataset(anchors, cross_image_path, single_image_path, annotation_path, test_data_list, input_size, s_scale,
    # m_scale, l_scale)
    # test = Test(anchors, 1, test_dataset, './output/parameters_cuda.pkl', device)
    # test.run()
