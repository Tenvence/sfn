import numpy as np
import xml.etree.cElementTree as Et


def get_dataset_list(dataset_list_path):
    f = open(dataset_list_path, 'r')
    dataset_list = [dataset_name.strip() for dataset_name in f.readlines()]
    f.close()
    return dataset_list


def get_class_list(classes_file_path):
    f = open(classes_file_path, 'r')
    class_list = [class_name.strip() for class_name in f.readlines()]
    f.close()
    return class_list


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape((3, 3, 2))


def parse_annotation_file(annotation_file, class_list):
    bbox_info_list = []
    root = Et.parse(annotation_file).getroot()

    for obj in root.findall('object'):
        bbox_obj = obj.find('bndbox')

        x_min = int(bbox_obj.find('xmin').text)
        y_min = int(bbox_obj.find('ymin').text)
        x_max = int(bbox_obj.find('xmax').text)
        y_max = int(bbox_obj.find('ymax').text)
        class_idx = class_list.index(obj.find('name').text)

        bbox_info_list.append([x_min, y_min, x_max, y_max, class_idx])

    return bbox_info_list
