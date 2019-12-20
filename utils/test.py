import torch
import numpy as np
import utils.utils as utils
import utils.config as cfg
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.yolov3_net import Yolov3Net
from utils.gravel_dataset import GravelDataset
from utils.read import get_anchors


class Test:
    def __init__(self, batch_size: int, param_file: str):
        self.data_loader = DataLoader(GravelDataset(is_train=False), batch_size, shuffle=False)
        self.model = Yolov3Net()
        self.model.load_state_dict(torch.load(param_file))
        self.model.eval()

    def run(self):
        print('test running')
        process_bar = tqdm(self.data_loader)
        for d in process_bar:
            crossed_image, single_image, s_tensor, m_tensor, l_tensor, s_coords, m_coords, l_coords, org_w, org_h = d
            s_output, m_output, l_output = self.model(crossed_image)
            # print(s_output.shape, m_output.shape, l_output.shape)

            s_scale, m_scale, l_scale = np.array(cfg.DETECTION_SCALE)
            s_anchors, m_anchors, l_anchors = torch.tensor(get_anchors(cfg.ANCHOR_FILE_PATH))

            s_output = utils.transform_output(s_output, 3, 1)
            m_output = utils.transform_output(m_output, 3, 1)
            l_output = utils.transform_output(l_output, 3, 1)

            s_decode_output = utils.decode_output_to_tensor(s_output, 3, s_scale, s_anchors)
            m_decode_output = utils.decode_output_to_tensor(m_output, 3, m_scale, m_anchors)
            l_decode_output = utils.decode_output_to_tensor(l_output, 3, l_scale, l_anchors)

            s_output = s_decode_output.reshape((-1, 6))
            m_output = m_decode_output.reshape((-1, 6))
            l_output = l_decode_output.reshape((-1, 6))
            pred_bbox = torch.cat([s_output, m_output, l_output], dim=0)
            pred_bbox = np.array(pred_bbox.detach())
            # print(pred_bbox.shape)
            # print(s_output.shape, m_output.shape, l_output.shape, pred_bbox.shape)
            pred_xywh = pred_bbox[:, 0:4]
            pred_conf = pred_bbox[:, 4]
            pred_prob = pred_bbox[:, 5:]
            score_threshold = 0.3

            valid_scale = [0, np.inf]

            org_h = org_h.numpy()[0]
            org_w = org_w.numpy()[0]

            # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
            pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5, pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
            # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
            resize_ratio = min(608 / org_w, 608 / org_h)
            # print(org_h, org_w)

            dw = (608 - resize_ratio * org_w) / 2
            dh = (608 - resize_ratio * org_h) / 2

            # print(dw, dh)

            pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
            pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

            # print(pred_coor[:, 0::2], pred_coor[:, 1::2])

            # # (3) clip some boxes those are out of range
            pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                        np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
            invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
            pred_coor[invalid_mask] = 0

            # # (4) discard some invalid boxes
            bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
            scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

            # # (5) discard some boxes with low scores
            classes = np.argmax(pred_prob, axis=-1)
            scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
            score_mask = scores > score_threshold
            mask = np.logical_and(scale_mask, score_mask)
            coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

            res = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
            process_bar.set_description(str(res.shape))

            # exit(-1)
