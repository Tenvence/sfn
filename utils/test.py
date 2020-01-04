import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.yolov3_net import Yolov3Net
from utils.iou import compute_iou


class Test:
    def __init__(self, dataset, param_file, device):
        print('Initialize test object ...')
        self.data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.device = device
        self.input_size = dataset.input_size

        self.conf_thresh = 0.3
        self.iou_thresh = 0.45

        print('Loading model from %s ...' % param_file)
        anchors = [dataset.s_anchors, dataset.m_anchors, dataset.l_anchors]
        self.model = Yolov3Net(anchors)
        self.model.load_state_dict(torch.load(param_file, map_location=self.device))
        self.model.eval()

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).to(device=self.device)

        print('Model loading succeed.')

    def run(self):
        print('Test running ...')
        process_bar = tqdm(self.data_loader)
        for d in process_bar:
            crossed_image, single_image, s_gt_tensor, m_gt_tensor, l_gt_tensor, s_gt_coords, m_gt_coords, l_gt_coords, w, h = d

            s_output, m_output, l_output = self.model(crossed_image)

            s_output = s_output.reshape((-1, 5))
            m_output = m_output.reshape((-1, 5))
            l_output = l_output.reshape((-1, 5))
            pred_box = torch.cat([s_output, m_output, l_output], dim=0)

            pred_coord, pred_conf = self.postprocess_boxes(pred_box, w, h)

            print(pred_conf.shape, pred_coord.shape)

            pred_boxes = self.nms(pred_coord, pred_conf)

            self.draw_rectangle(pred_boxes, 'result')

            exit(-1)

    def postprocess_boxes(self, pred_box, w, h):
        pred_coord = pred_box[:, 0:4]
        pred_conf = pred_box[:, 4:]

        # [x, y, w, h] -> [x_min, y_min, x_max, y_max]
        pred_coord = torch.cat([pred_coord[:, :2] - pred_coord[:, 2:] * 0.5, pred_coord[:, :2] + pred_coord[:, 2:] * 0.5], dim=-1)

        print(pred_coord.shape)

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

        i = 0
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

            print(len(best_boxes))

            self.draw_rectangle(best_boxes, str(i))
            i += 1

        return best_boxes

    @staticmethod
    def draw_rectangle(pred_coord, x):
        img = cv2.imread('./GravelDataset/Images+/000000.jpg')
        for box in pred_coord:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)

        cv2.imwrite('./output/img_' + x + '.jpg', img)
