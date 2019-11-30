import os
import torch.utils.data as data
import torchvision.transforms as transforms
import utils.config as cfg
import xml.etree.cElementTree as Et
from PIL import Image


class GravelDataset(data.Dataset):
    def __init__(self, is_train=False):
        self.dataset_list_path = cfg.TRAIN_LIST_PATH if is_train else cfg.TEST_LIST_PATH
        self.crossed_image_path = cfg.CROSSED_IMAGE_PATH
        self.single_image_path = cfg.SINGLE_IMAGE_PATH
        self.annotations_path = cfg.ANNOTATIONS_PATH
        self.classes_file_path = cfg.CLASSES_FILE_PATH
        self.input_image_size = cfg.INPUT_IMAGE_SIZE
        self.data_aug = is_train

        self.dataset_list = self.get_dataset_list()
        self.class_list = self.get_class_list()
        self.class_num = 1 if cfg.IS_ONE_CLASSIFICATION else len(self.class_list)

    def __getitem__(self, index):
        transforms_convert = transforms.Compose([
            transforms.Resize([cfg.INPUT_IMAGE_SIZE, cfg.INPUT_IMAGE_SIZE]),
            transforms.ToTensor()
        ])
        dataset_name = self.dataset_list[index]
        crossed_image = Image.open(os.path.join(self.crossed_image_path, dataset_name + '.jpg'))
        single_image = Image.open(os.path.join(self.single_image_path, dataset_name + '.jpg'))
        crossed_image = transforms_convert(crossed_image)
        single_image = transforms_convert(single_image)
        bbox_info_list = self.parse_annotation_file(dataset_name)

        return [crossed_image, single_image, bbox_info_list]

    def __len__(self):
        return len(self.dataset_list)

    def get_dataset_list(self):
        f = open(self.dataset_list_path, 'r')
        dataset_list = [dataset_name.strip() for dataset_name in f.readlines()]
        f.close()
        return dataset_list

    def get_class_list(self):
        f = open(self.classes_file_path, 'r')
        class_list = [class_name.strip() for class_name in f.readlines()]
        f.close()
        return class_list

    def parse_annotation_file(self, annotation_file_name):
        bbox_info_list = []
        root = Et.parse(os.path.join(self.annotations_path, annotation_file_name + '.xml')).getroot()

        for obj in root.findall('object'):
            bbox_obj = obj.find('bndbox')

            x_min = bbox_obj.find('xmin').text
            x_max = bbox_obj.find('xmax').text
            y_min = bbox_obj.find('ymin').text
            y_max = bbox_obj.find('ymax').text

            if self.class_num == 1:
                class_idx = 0
            else:
                class_name = obj.find('name').text
                class_idx = self.class_list.index(class_name)

            bbox_info_list.append([class_idx, x_min, x_max, y_min, y_max])

        return bbox_info_list
