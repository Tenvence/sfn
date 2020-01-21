import argparse
import os

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from utils.gravel_dataset import GravelDataset
from utils.read import get_anchors, get_dataset_list
from utils.test import Test
from utils.train import Train

if __name__ == '__main__':
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('-n', default='default', help='train and test process name, default name is \'default\'')
    # name = arg_parser.parse_args().n
    #
    # log_dir = './output/%s' % name
    # writer = SummaryWriter()
    # print('NAME                | %s' % name)
    # print('TENSORBOARD log_dir | %s' % log_dir)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    dataset_dict = './GravelDataset'

    cross_image_path = os.path.join(dataset_dict, 'Images+')
    single_image_path = os.path.join(dataset_dict, 'Images-')
    cross_image_train_path = os.path.join(dataset_dict, 'Images+.INPUT')
    single_image_train_path = os.path.join(dataset_dict, 'Images-.INPUT')
    images_path = [cross_image_path, single_image_path, cross_image_train_path, single_image_train_path]
    annotation_path = os.path.join(dataset_dict, 'Annotations')
    annotation_train_path = os.path.join(dataset_dict, 'Annotations.INPUT')
    annotations_path = [annotation_path, annotation_train_path]

    anchors = get_anchors('./docs/new_gravel_anchors.txt')
    train_data_list = get_dataset_list('./GravelDataset/train_list.txt')
    test_data_list = get_dataset_list('./GravelDataset/test_list.txt')

    input_size = 608
    s_scale, m_scale, l_scale = 8, 16, 32
    scale = [s_scale, m_scale, l_scale]
    batch_size = 8

    output_path = './output/parameters_3.pkl'

    train_dataset = GravelDataset(anchors, images_path, annotations_path, train_data_list, input_size, scale, device, train=True)
    train = Train(train_dataset, batch_size, device)
    train.run(epoch_num=120, warm_epoch_num=2, output_path=output_path)

    test_dataset = GravelDataset(anchors, images_path, annotations_path, test_data_list, input_size, scale, device, train=False)
    test = Test(test_dataset, output_path, device)
    test.run()

    # writer.flush()
    # writer.close()
