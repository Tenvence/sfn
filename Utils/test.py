import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Utils.iou import compute_iou
from .eval import eval


class Test:
    def __init__(self, dataset, model_file, device=torch.device('cpu')):
        self.data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        self.device = device

        self.input_size = dataset.input_size

        self.conf_thresh = 0.3
        self.iou_thresh = 0.45

        self.model = torch.load(model_file)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).to(device=self.device)

        self.model.eval()

    def run(self, output_dict):
        process_bar = tqdm(self.data_loader)

        predicted_dir_path = output_dict + '/predicted'
        ground_truth_dir_path = output_dict + '/ground-truth'

        if os.path.exists(predicted_dir_path):
            shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path):
            shutil.rmtree(ground_truth_dir_path)

        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)

        idx = 0
        for d in process_bar:
            crossed_image, single_image, raw_h, raw_w, raw_bboxes = d

            crossed_image = crossed_image.to(device=self.device)
            single_image = single_image.to(device=self.device)

            raw_bboxes = torch.squeeze(raw_bboxes)
            raw_h = torch.squeeze(raw_h).float().numpy()
            raw_w = torch.squeeze(raw_w).float().numpy()

            with open(os.path.join(ground_truth_dir_path, str(idx) + '.txt'), 'w') as f:
                for gt_box in raw_bboxes:
                    x_min, y_min, x_max, y_max = gt_box.numpy()
                    f.write('gravel %d %d %d %d\n' % (x_min, y_min, x_max, y_max))

            s_output, m_output, l_output = self.model(crossed_image, single_image)

            # remove grad
            s_output = s_output.detach().reshape((-1, 5))
            m_output = m_output.detach().reshape((-1, 5))
            l_output = l_output.detach().reshape((-1, 5))
            pred_box = torch.cat([s_output, m_output, l_output], dim=0)

            pred_coord, pred_conf = self.postprocess_boxes(pred_box, raw_w, raw_h)
            pred_boxes = self.nms(pred_coord, pred_conf)

            with open(os.path.join(predicted_dir_path, str(idx) + '.txt'), 'w') as f:
                for box in pred_boxes:
                    x_min, y_min, x_max, y_max, conf = box
                    f.write('gravel %.4f %d %d %d %d\n' % (conf, x_min, y_min, x_max, y_max))

            process_bar.set_description("    Eval. #gt = %d, #pred = %d" % (len(raw_bboxes), len(pred_boxes)))
            idx += 1

        iou_range = np.arange(0.5, 1.0, 0.05)
        lines = []
        aps = []
        f1s = []
        for iou_thresh in iou_range:
            ap, gt_num, tp, fp, p, r, f1 = eval(ground_truth_dir_path, predicted_dir_path, iou_thresh)
            lines.append('%.2f %.2f %d %d %d %.2f %.2f %.2f\n' % (iou_thresh, ap, gt_num, tp, fp, p, r, f1))
            aps.append(ap)
            f1s.append(f1)

        with open(output_dict + '/record.txt', 'w') as f:
            f.writelines(lines)

        print('    Evaluation Finished! AP50=%.2f; AP70=%.2f; AP90=%.2f; mAP=%.2f; mF1=%.2f\n' % (
            aps[0], aps[4], aps[8], sum(aps) / len(aps), sum(f1s) / len(f1s)))

        with open(output_dict + '/verify_record.txt', 'a+') as f:
            f.writelines('%.2f %.2f %.2f %.2f %.2f\n' % (aps[0], aps[4], aps[8], sum(aps) / len(aps), sum(f1s) / len(f1s)))

    def postprocess_boxes(self, pred_box, w, h):
        pred_coord = pred_box[:, 0:4]
        pred_conf = pred_box[:, 4:]

        # [x, y, w, h] -> [x_min, y_min, x_max, y_max]
        pred_coord = torch.cat([pred_coord[:, :2] - pred_coord[:, 2:] * 0.5, pred_coord[:, :2] + pred_coord[:, 2:] * 0.5], dim=-1)

        # [x_min, y_min, x_max, y_max] -> [x_min_org, y_min_org, x_max_org, y_max_org]
        input_size = float(self.input_size)
        resize_ratio = min(input_size / w, input_size / h)
        dw = (input_size - resize_ratio * w) / 2
        dh = (input_size - resize_ratio * h) / 2

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
