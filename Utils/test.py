import os
import shutil

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.yolov3_net import Yolov3Net
from Utils.iou import compute_iou


class Test:
    def __init__(self, dataset, param_file, device=torch.device('cpu')):
        self.data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.device = device
        self.input_size = dataset.input_size

        self.conf_thresh = 0.3
        self.iou_thresh = 0.45

        anchors = [dataset.s_anchors, dataset.m_anchors, dataset.l_anchors]

        self.model = Yolov3Net(anchors)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).to(device=self.device)

        self.model.load_state_dict(torch.load(param_file))
        self.model.eval()

    def run(self):
        process_bar = tqdm(self.data_loader)

        predicted_dir_path = './mAP/predicted'
        ground_truth_dir_path = './mAP/ground-truth'
        result_image_dir_path = './mAP/result-image'

        if os.path.exists(predicted_dir_path):
            shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path):
            shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(result_image_dir_path):
            shutil.rmtree(result_image_dir_path)

        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(result_image_dir_path)

        idx = 0
        for d in process_bar:
            crossed_image, single_image, crossed_image_raw, single_image_raw, gt_boxes_position_raw = d

            with open(os.path.join(ground_truth_dir_path, str(idx) + '.txt'), 'w') as f:
                for gt_box in gt_boxes_position_raw[0]:
                    x_min, y_min, x_max, y_max = gt_box.numpy()
                    f.write('gravel %d %d %d %d\n' % (x_min, y_min, x_max, y_max))

            h, w = crossed_image_raw.shape[2], crossed_image_raw.shape[3]

            s_output, m_output, l_output = self.model(crossed_image, single_image)

            s_output = s_output.reshape((-1, 5))
            m_output = m_output.reshape((-1, 5))
            l_output = l_output.reshape((-1, 5))
            pred_box = torch.cat([s_output, m_output, l_output], dim=0)

            pred_coord, pred_conf = self.postprocess_boxes(pred_box, torch.tensor(w), torch.tensor(h))
            pred_boxes = self.nms(pred_coord, pred_conf)

            with open(os.path.join(predicted_dir_path, str(idx) + '.txt'), 'w') as f:
                for box in pred_boxes:
                    x_min, y_min, x_max, y_max, conf = box
                    f.write('gravel %.4f %d %d %d %d\n' % (conf, x_min, y_min, x_max, y_max))

            idx += 1

    def postprocess_boxes(self, pred_box, w, h):
        pred_coord = pred_box[:, 0:4]
        pred_conf = pred_box[:, 4:]

        # [x, y, w, h] -> [x_min, y_min, x_max, y_max]
        pred_coord = torch.cat([pred_coord[:, :2] - pred_coord[:, 2:] * 0.5, pred_coord[:, :2] + pred_coord[:, 2:] * 0.5], dim=-1)

        # [x_min, y_min, x_max, y_max] -> [x_min_org, y_min_org, x_max_org, y_max_org]
        w = w.float()
        h = h.float()
        input_size = float(self.input_size)
        resize_ratio = min(input_size / w, input_size / h)
        dw = (input_size - resize_ratio * w) / 2
        dh = (input_size - resize_ratio * h) / 2

        if torch.cuda.is_available() and pred_coord.is_cuda:
            dw = dw.to(device=pred_coord.device)
            dh = dh.to(device=pred_coord.device)
            resize_ratio = torch.tensor([resize_ratio]).to(device=pred_coord.device)

        pred_coord[:, 0::2] = (pred_coord[:, 0::2] - dw) / resize_ratio
        pred_coord[:, 1::2] = (pred_coord[:, 1::2] - dh) / resize_ratio

        # clip some boxes which are out of range
        left_up_point = torch.tensor([0, 0]).float()
        right_down_point = torch.tensor([w - 1, h - 1]).float()

        if torch.cuda.is_available() and pred_coord.is_cuda:
            left_up_point = left_up_point.to(device=pred_coord.device)
            right_down_point = right_down_point.to(device=pred_coord.device)

        pred_coord = torch.cat([torch.max(pred_coord[:, :2], left_up_point), torch.min(pred_coord[:, 2:], right_down_point)], dim=-1)
        invalid_mask = torch.gt(pred_coord[:, 0], pred_coord[:, 2]) | torch.gt(pred_coord[:, 1], pred_coord[:, 3])
        pred_coord[invalid_mask] = 0

        # discard some invalid boxes
        box_scale = pred_coord[:, 2:4] - pred_coord[:, 0:2]
        box_scale = torch.sqrt(box_scale[:, :1] * box_scale[:, 1:])
        scale_mask = (box_scale > 0) & (box_scale < np.inf)

        # discard some boxes with low scores
        conf_mask = pred_conf > self.conf_thresh
        mask = torch.squeeze(scale_mask & conf_mask)
        pred_coord = pred_coord[mask]
        pred_conf = pred_conf[mask]

        return pred_coord, pred_conf

    def nms(self, boxes, conf):
        best_boxes = []

        while boxes.shape[0] > 0:
            max_idx = torch.argmax(conf)

            best_box = boxes[max_idx]
            best_conf = conf[max_idx]
            best_boxes.append(torch.cat([best_box, best_conf]).detach().cpu().numpy())

            boxes = torch.cat([boxes[:max_idx], boxes[max_idx + 1:]])
            conf = torch.cat([conf[:max_idx], conf[max_idx + 1:]])

            iou = compute_iou(best_box[np.newaxis, :4], boxes, is_regularize=False)
            iou_mask = iou < self.iou_thresh

            boxes = boxes[iou_mask]
            conf = conf[iou_mask]

        return best_boxes

    def draw_eval_file(self):

        pass

    @staticmethod
    def draw_rectangle(image, pred_coord, color=(0, 255, 0), thickness=5):
        for box in pred_coord:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)

        return image
