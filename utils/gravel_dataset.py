import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import utils.config as cfg
import utils.utils as utils
from utils.read_file_utils import get_dataset_list, get_class_list, get_anchors, parse_annotation_file
from PIL import Image


class GravelDataset(data.Dataset):
    def __init__(self, is_train=False):
        self.crossed_image_path = cfg.CROSSED_IMAGE_PATH
        self.single_image_path = cfg.SINGLE_IMAGE_PATH
        self.annotations_path = cfg.ANNOTATIONS_PATH

        self.input_image_size = cfg.INPUT_IMAGE_SIZE
        self.detection_scale = np.array(cfg.DETECTION_SCALE)
        self.output_size = self.input_image_size // self.detection_scale

        self.max_bbox_per_scale = cfg.MAX_BBOX_PER_SCALE
        self.anchor_per_scale = cfg.ANCHOR_PER_SCALE

        self.data_aug = is_train

        self.dataset_list = get_dataset_list(cfg.TRAIN_LIST_PATH if is_train else cfg.TEST_LIST_PATH)
        self.class_list = get_class_list(cfg.CLASSES_FILE_PATH)
        self.anchors = get_anchors(cfg.ANCHOR_FILE_PATH)

        self.class_num = 1 if cfg.IS_ONE_CLASSIFICATION else len(self.class_list)

    def __getitem__(self, index):
        dataset_name = self.dataset_list[index]

        crossed_image = Image.open(os.path.join(self.crossed_image_path, dataset_name + '.jpg'))
        single_image = Image.open(os.path.join(self.single_image_path, dataset_name + '.jpg'))
        bbox_info_array = np.array(parse_annotation_file(os.path.join(self.annotations_path, dataset_name + '.xml'), self.class_list))

        # bbox_info 的格式为 [x_min, y_min, x_max, y_max, cla_idx]

        if self.class_num == 1:
            bbox_info_array[:, 4] = 0  # 单份类问题的 cla_idx 都置为相同的数

        w, h = crossed_image.size
        scale = min(self.input_image_size / w, self.input_image_size / h)
        resize_w, resize_h = int(w * scale), int(h * scale)
        pad_x, pad_y = (self.input_image_size - resize_w) // 2, (self.input_image_size - resize_h) // 2

        transform_image = transforms.Compose([
            transforms.Resize((resize_h, resize_w)),
            transforms.Pad((pad_x, pad_y, pad_x, pad_y), (125, 125, 125)),
            transforms.ToTensor()
        ])

        crossed_image = transform_image(crossed_image)
        single_image = transform_image(single_image)

        # bbox 的尺寸位置也要跟着 image 的变形而变化
        bbox_info_array[:, [0, 2]] = bbox_info_array[:, [0, 2]] * scale + pad_x
        bbox_info_array[:, [1, 3]] = bbox_info_array[:, [1, 3]] * scale + pad_y

        s_tensor, m_tensor, l_tensor, s_coords, m_coords, l_coords = self.encode_true_bboxes_to_tensor(bbox_info_array)

        return crossed_image, single_image, s_tensor, m_tensor, l_tensor, s_coords, m_coords, l_coords

    def __len__(self):
        return len(self.dataset_list)

    def encode_true_bboxes_to_tensor(self, bbox_info_array):
        s_output_size, m_output_size, l_output_size = self.output_size  # 小、中、大尺度检测

        # 网络中各个尺度的输出张量尺寸
        s_tensor = np.zeros((s_output_size, s_output_size, self.anchor_per_scale, 5 + self.class_num))
        m_tensor = np.zeros((m_output_size, m_output_size, self.anchor_per_scale, 5 + self.class_num))
        l_tensor = np.zeros((l_output_size, l_output_size, self.anchor_per_scale, 5 + self.class_num))

        # 保存不同尺度下的生成有效 tensor 中向量的 bbox 尺寸，保持相同的大小是为了多个矩阵成功组成一个 batch，0 为计数器
        s_coords = [np.zeros((self.max_bbox_per_scale, 4)), 0]
        m_coords = [np.zeros((self.max_bbox_per_scale, 4)), 0]
        l_coords = [np.zeros((self.max_bbox_per_scale, 4)), 0]

        # 将 ground truth 的信息映射到与网络输出尺寸相同的张量上
        for bbox in bbox_info_array:
            location = bbox[:4]  # bbox 的位置信息，为 [x_min, y_min, x_max, y_max]
            coord = np.concatenate([(location[2:] + location[:2]) * 0.5, location[2:] - location[:2]], axis=-1)  # 转换为坐标信息 [x, y, w, h]
            s_coord, m_coord, l_coord = 1.0 * coord[np.newaxis, :] / self.detection_scale[:, np.newaxis]  # bbox 在不同尺度下的坐标

            cla = bbox[4]  # bbox 的分类信息
            cla_conf = utils.smooth_onehot(num=self.class_num, idx=cla, delta=0.01)  # 将分类信息使用 smooth onehot 转换为各类别的分布信息

            s_anchor_size, m_anchor_size, l_anchor_size = self.anchors  # 将9个 anchor 分配给3种尺度检测，每种尺度有3个 anchor

            s_iou, s_tensor, s_coords, s_is_positive_example = \
                utils.compute_vectors_of_one_bbox(s_coord, coord, cla_conf, s_anchor_size, s_tensor, s_coords, anchor_num=self.anchor_per_scale)
            m_iou, m_tensor, m_coords, m_is_positive_example = \
                utils.compute_vectors_of_one_bbox(m_coord, coord, cla_conf, m_anchor_size, m_tensor, m_coords, anchor_num=self.anchor_per_scale)
            l_iou, l_tensor, l_coords, l_is_positive_example = \
                utils.compute_vectors_of_one_bbox(l_coord, coord, cla_conf, l_anchor_size, l_tensor, l_coords, anchor_num=self.anchor_per_scale)

            # 如果三个尺度都没有符合 IoU 要求的 anchor box，那么就把 IoU 最大的拿出来
            if not s_is_positive_example and not m_is_positive_example and not l_is_positive_example:
                s_iou = np.array(s_iou).reshape(-1)
                m_iou = np.array(m_iou).reshape(-1)
                l_iou = np.array(l_iou).reshape(-1)

                iou_list = np.concatenate([s_iou, m_iou, l_iou], axis=-1)
                best_anchor_idx = np.argmax(iou_list, axis=-1)
                best_scale = int(best_anchor_idx / self.anchor_per_scale)
                best_anchor = int(best_anchor_idx % self.anchor_per_scale)

                if best_scale == 0:
                    s_tensor, s_coords = utils.compute_valid_vectors_of_one_bbox(s_coord, coord, best_anchor, cla_conf, s_tensor, s_coords)
                elif best_scale == 1:
                    m_tensor, m_coords = utils.compute_valid_vectors_of_one_bbox(m_coord, coord, best_anchor, cla_conf, m_tensor, m_coords)
                else:
                    l_tensor, l_coords = utils.compute_valid_vectors_of_one_bbox(l_coord, coord, best_anchor, cla_conf, l_tensor, l_coords)

        s_tensor = torch.tensor(s_tensor, dtype=torch.float32)
        m_tensor = torch.tensor(m_tensor, dtype=torch.float32)
        l_tensor = torch.tensor(l_tensor, dtype=torch.float32)

        s_coords = torch.tensor(np.array(s_coords[0]), dtype=torch.float32)
        m_coords = torch.tensor(np.array(m_coords[0]), dtype=torch.float32)
        l_coords = torch.tensor(np.array(l_coords[0]), dtype=torch.float32)

        return s_tensor, m_tensor, l_tensor, s_coords, m_coords, l_coords
