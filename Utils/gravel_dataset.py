import os
import random
import xml.etree.cElementTree as Et

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from Utils.iou import compute_iou


class GravelDataset(data.Dataset):
    def __init__(self, anchors, image_dir_path, annotation_dir_path, data_list, input_size, scale, device, train):
        self.crossed_image_dir_path, self.single_image_dir_path = image_dir_path
        self.annotations_dir_path = annotation_dir_path

        self.input_size = input_size
        self.s_scale, self.m_scale, self.l_scale = scale
        self.s_output_size = self.input_size // self.s_scale
        self.m_output_size = self.input_size // self.m_scale
        self.l_output_size = self.input_size // self.l_scale

        self.dataset_list = data_list
        self.s_anchors, self.m_anchors, self.l_anchors = anchors
        self.device = device
        self.train = train
        self.iou_thresh = 0.3

    def __getitem__(self, index):
        dataset_name = self.dataset_list[index]

        crossed_image_path = os.path.join(self.crossed_image_dir_path, dataset_name + '.jpg')
        single_image_path = os.path.join(self.single_image_dir_path, dataset_name + '.jpg')
        annotation_path = os.path.join(self.annotations_dir_path, dataset_name + '.xml')

        crossed_image = Image.open(crossed_image_path)
        single_image = Image.open(single_image_path)
        bboxes = self.parse_annotation_file(annotation_path)
        raw_w, raw_h = crossed_image.size

        raw_bboxes = torch.tensor(bboxes)

        if self.train and random.randint(0, 1) == 0:
            crossed_image, single_image, bboxes = self.horizontal_flip(crossed_image, single_image, bboxes)

        if self.train and random.randint(0, 1) == 0:
            crossed_image, single_image, bboxes = self.vertical_flip(crossed_image, single_image, bboxes)

        if self.train and random.randint(0, 1) == 0:
            rot_degrees = np.arange(-15, 16)
            degree = random.choice(rot_degrees)
            crossed_image, single_image, bboxes = self.rotate(crossed_image, single_image, bboxes, degree)

        crossed_image, single_image, bboxes = self.resize(crossed_image, single_image, bboxes)
        crossed_image, single_image, bboxes = self.pad(crossed_image, single_image, bboxes)

        to_tensor_trans = transforms.ToTensor()
        crossed_image, single_image = to_tensor_trans(crossed_image), to_tensor_trans(single_image)

        if not self.train:
            return crossed_image, single_image, raw_h, raw_w, raw_bboxes
        else:
            s_tensor, m_tensor, l_tensor, s_coords, m_coords, l_coords = self.encode_gt_bboxes(bboxes, self.iou_thresh)
            return crossed_image, single_image, s_tensor, m_tensor, l_tensor, s_coords, m_coords, l_coords

    def __len__(self):
        return len(self.dataset_list)

    @staticmethod
    def parse_annotation_file(annotation_file):
        bbox_info_list = []
        root = Et.parse(annotation_file).getroot()

        for obj in root.findall('object'):
            bbox_obj = obj.find('bndbox')

            x_min = float(bbox_obj.find('xmin').text)
            y_min = float(bbox_obj.find('ymin').text)
            x_max = float(bbox_obj.find('xmax').text)
            y_max = float(bbox_obj.find('ymax').text)

            bbox_info_list.append([x_min, y_min, x_max, y_max])

        return torch.tensor(bbox_info_list).float()

    def encode_gt_bboxes(self, gt_boxes_position, iou_thresh):
        max_boxes_per_scale = 30

        # 3 scale output tensor from Model
        s_tensor = torch.zeros((self.s_output_size, self.s_output_size, self.s_anchors.shape[0], 5))
        m_tensor = torch.zeros((self.m_output_size, self.m_output_size, self.m_anchors.shape[0], 5))
        l_tensor = torch.zeros((self.l_output_size, self.l_output_size, self.l_anchors.shape[0], 5))

        # Save valid vector of保存不同尺度下的生成有效 tensor 中向量的 bbox 尺寸，保持相同的大小是为了多个矩阵成功组成一个 batch，0 为计数器
        s_coords = [torch.zeros((max_boxes_per_scale, 4)), 0]
        m_coords = [torch.zeros((max_boxes_per_scale, 4)), 0]
        l_coords = [torch.zeros((max_boxes_per_scale, 4)), 0]

        # Map a gt box to output tensor
        for position in gt_boxes_position:
            # Convert position to coordination [x, y, w, h]
            coord = torch.cat([(position[2:] + position[:2]) * 0.5, position[2:] - position[:2]], dim=-1)

            # Coordination at different scales of the box
            s_coord, m_coord, l_coord = coord / self.s_scale, coord / self.m_scale, coord / self.l_scale

            # Compute one vector of tensor corresponding to the gt box center
            s_iou, s_tensor, s_coords, s_has_positive = self.compute_vector(s_tensor, s_coords, self.s_anchors, s_coord, coord, iou_thresh)
            m_iou, m_tensor, m_coords, m_has_positive = self.compute_vector(m_tensor, m_coords, self.m_anchors, m_coord, coord, iou_thresh)
            l_iou, l_tensor, l_coords, l_has_positive = self.compute_vector(l_tensor, l_coords, self.l_anchors, l_coord, coord, iou_thresh)

            # If there's no valid vector, taking out the vector of largest IoU
            # print(s_has_positive or m_has_positive or l_has_positive)
            if not (s_has_positive or m_has_positive or l_has_positive):
                iou_list = torch.cat([s_iou, m_iou, l_iou], dim=-1)
                best_idx = torch.argmax(iou_list, dim=-1)
                best_scale = int(best_idx / self.s_anchors.shape[0])
                best_anchor = int(best_idx % self.s_anchors.shape[0])

                if best_scale == 0:
                    s_tensor, s_coords = self.compute_valid_vectors(s_tensor, s_coords, s_coord, coord, best_anchor)
                elif best_scale == 1:
                    m_tensor, m_coords = self.compute_valid_vectors(m_tensor, m_coords, m_coord, coord, best_anchor)
                else:
                    l_tensor, l_coords = self.compute_valid_vectors(l_tensor, l_coords, l_coord, coord, best_anchor)

        return s_tensor, m_tensor, l_tensor, s_coords[0], m_coords[0], l_coords[0]

    def compute_vector(self, tensor, coords, anchors, scale_coord, raw_coord, iou_thresh):
        anchor_coord = torch.zeros((anchors.shape[0], 4))
        anchor_coord[:, 0:2] = scale_coord[0:2].floor() + 0.5  # Center coordination is center of the cell where center of box locates
        anchor_coord[:, 2:4] = anchors  # The size is anchors' size

        iou = compute_iou(anchor_coord, scale_coord)  # Compute IoU between scaled box and some anchor box
        iou_mask = torch.gt(iou, iou_thresh)  # Is IoU > thresh

        has_positive = False

        if torch.any(iou_mask):
            has_positive = True
            tensor, coords = self.compute_valid_vectors(tensor, coords, scale_coord, raw_coord, iou_mask)

        return iou, tensor, coords, has_positive

    @staticmethod
    def compute_valid_vectors(tensor, coords, scale_coord, raw_coord, mask):
        x_idx, y_idx = torch.floor(scale_coord[0:2]).long()  # Coordination of grid cell

        # Reduce the sensitivity of data
        x_idx, y_idx = torch.abs(x_idx), torch.abs(y_idx)
        if y_idx >= tensor.shape[1]:
            y_idx = tensor.shape[1] - 1
        if x_idx >= tensor.shape[0]:
            x_idx = tensor.shape[0] - 1

        tensor[y_idx, x_idx, mask, :] = 0

        tensor[y_idx, x_idx, mask, 0:4] = raw_coord
        tensor[y_idx, x_idx, mask, 4:5] = 1.0

        coords[0][coords[1], :] = raw_coord
        coords[1] += 1

        return tensor, coords

    @staticmethod
    def horizontal_flip(crossed_img, single_img, bboxes):
        iw, _ = crossed_img.size
        horizontal_flip_trans = transforms.RandomHorizontalFlip(p=1)

        crossed_img = horizontal_flip_trans(crossed_img)
        single_img = horizontal_flip_trans(single_img)

        bboxes[:, 0] = iw - bboxes[:, 0]
        bboxes[:, 2] = iw - bboxes[:, 2]

        return crossed_img, single_img, bboxes

    @staticmethod
    def vertical_flip(crossed_img, single_img, bboxes):
        _, ih = crossed_img.size
        vertical_flip_trans = transforms.RandomVerticalFlip(p=1)

        crossed_img = vertical_flip_trans(crossed_img)
        single_img = vertical_flip_trans(single_img)

        bboxes[:, 1] = ih - bboxes[:, 1]
        bboxes[:, 3] = ih - bboxes[:, 3]

        return crossed_img, single_img, bboxes

    @staticmethod
    def rotate(crossed_img, single_img, bboxes, degree):
        iw, ih = crossed_img.size
        rotate_trans = transforms.RandomRotation(degrees=(degree, degree), expand=True, fill=128)

        crossed_img = rotate_trans(crossed_img)
        single_img = rotate_trans(single_img)

        rw, rh = crossed_img.size

        angle = -np.pi / 180. * degree
        rot_mat = torch.tensor([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ]).float()[None, None, :, :]

        coords = torch.cat([
            bboxes[:, (0, 1)][:, None, :, None],
            bboxes[:, (2, 1)][:, None, :, None],
            bboxes[:, (0, 3)][:, None, :, None],
            bboxes[:, (2, 3)][:, None, :, None]
        ], dim=1) - torch.tensor([[iw / 2], [ih / 2]])
        rot_coords = torch.matmul(rot_mat, coords) + torch.tensor([[rw / 2], [rh / 2]])

        max_coords, _ = torch.max(rot_coords, dim=1)
        min_coords, _ = torch.min(rot_coords, dim=1)
        max_coords = torch.squeeze(max_coords)
        min_coords = torch.squeeze(min_coords)

        bboxes = torch.cat([min_coords, max_coords], dim=1)
        return crossed_img, single_img, bboxes

    def resize(self, crossed_img, single_img, bboxes):
        iw, ih = crossed_img.size
        scale = self.input_size / max(iw, ih)
        resize_trans = transforms.Resize(size=(int(ih * scale), int(iw * scale)))
        crossed_img = resize_trans(crossed_img)
        single_img = resize_trans(single_img)
        bboxes *= scale
        return crossed_img, single_img, bboxes

    def pad(self, crossed_img, single_img, bboxes):
        iw, ih = crossed_img.size

        pwl = int(np.floor((self.input_size - iw) / 2))
        pwr = int(np.ceil((self.input_size - iw) / 2))
        pht = int(np.floor((self.input_size - ih) / 2))
        phb = int(np.ceil((self.input_size - ih) / 2))

        pad_trans = transforms.Pad(padding=(pwl, pht, pwr, phb), fill=(128, 128, 128))
        crossed_img = pad_trans(crossed_img)
        single_img = pad_trans(single_img)

        bboxes[:, (0, 2)] += pwl
        bboxes[:, (1, 3)] += pht

        return crossed_img, single_img, bboxes
