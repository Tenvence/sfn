import argparse

import torch
from torch.utils.tensorboard.writer import SummaryWriter

import utils.config as cfg
from utils.gravel_dataset import GravelDataset
from utils.read import get_anchors, get_dataset_list
from utils.test import Test
from utils.train import Train

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-n', default='default', help='train and test process name, default name is \'default\'')
    name = arg_parser.parse_args().n

    log_dir = './output/%s' % name
    # writer = SummaryWriter()
    print('NAME                | %s' % name)
    print('TENSORBOARD log_dir | %s' % log_dir)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    cross_image_path = cfg.CROSSED_IMAGE_PATH
    single_image_path = cfg.SINGLE_IMAGE_PATH
    image_path = [cross_image_path, single_image_path]
    annotation_path = cfg.ANNOTATIONS_PATH

    anchors = get_anchors(cfg.ANCHOR_FILE_PATH)
    train_data_list = get_dataset_list(cfg.TRAIN_LIST_PATH)
    test_data_list = get_dataset_list(cfg.TEST_LIST_PATH)

    input_size = 608
    s_scale, m_scale, l_scale = 8, 16, 32
    scale = [s_scale, m_scale, l_scale]
    batch_size = 10

#     train_dataset = GravelDataset(anchors, image_path, annotation_path, train_data_list, input_size, scale, device, train=True)
#     train = Train(train_dataset, batch_size, device)
#     train.run(epoch_num=30, warm_epoch_num=2, output_path='./output/parameters_new_2_dual.pkl')

    test_dataset = GravelDataset(anchors, image_path, annotation_path, test_data_list, input_size, scale, device, train=False)
    test = Test(test_dataset, './output/parameters_new_2_dual.pkl', device)
    test.run()

    # writer.flush()
    # writer.close()
