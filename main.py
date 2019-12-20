import torch
import utils.config as cfg
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from model.yolov3_net import Yolov3Net
from model.loss import Loss
from model.lr_scheduler import LinearCosineScheduler
from utils.read import get_anchors, get_dataset_list
from utils.test import Test
from utils.gravel_dataset import GravelDataset


class Train:
    def __init__(self, anchors, batch_size, device=torch.device('cpu')):
        self.anchors = anchors
        self.device = device

        self.dataset = GravelDataset(anchors, cfg.CROSSED_IMAGE_PATH, cfg.SINGLE_IMAGE_PATH, cfg.ANNOTATIONS_PATH,
                                     get_dataset_list(cfg.TRAIN_LIST_PATH), 608, 8, 16, 32, 0.3)
        self.data_loader = DataLoader(self.dataset, batch_size, shuffle=True)

        self.model = Yolov3Net(anchors)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).to(device=self.device)
        self.model.train()

        self.optimizer = Adam(self.model.parameters(), weight_decay=0.0005)
        self.criterion = Loss(self.anchors, input_size=608, iou_thresh=0.5)

    def run(self, epoch_num, warm_epoch_num, parameters_save_path):
        steps_per_epoch = len(self.data_loader)
        warm_steps = warm_epoch_num * steps_per_epoch
        total_steps = epoch_num * steps_per_epoch
        lr_scheduler = LinearCosineScheduler(self.optimizer, warm_steps, total_steps)

        step_idx = 0
        for epoch in range(epoch_num):
            process_bar = tqdm(self.data_loader)
            for d in process_bar:
                crossed_image, single_image, s_gt_tensor, m_gt_tensor, l_gt_tensor, s_gt_coords, m_gt_coords, l_gt_coords = d

                if torch.cuda.is_available():
                    crossed_image = crossed_image.to(device=self.device)
                    single_image = single_image.to(device=self.device)
                    s_gt_tensor = s_gt_tensor.to(device=self.device)
                    m_gt_tensor = m_gt_tensor.to(device=self.device)
                    l_gt_tensor = l_gt_tensor.to(device=self.device)
                    s_gt_coords = s_gt_coords.to(device=self.device)
                    m_gt_coords = m_gt_coords.to(device=self.device)
                    l_gt_coords = l_gt_coords.to(device=self.device)

                self.optimizer.zero_grad()
                s_output, m_output, l_output = self.model(crossed_image)
                loss = self.criterion(s_output, m_output, l_output, s_gt_tensor, m_gt_tensor, l_gt_tensor, s_gt_coords, m_gt_coords, l_gt_coords)
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step(step_idx)
                step_idx += 1
                process_bar.set_description("train loss is %.2f" % float(loss))

        torch.save(self.model.state_dict(), parameters_save_path)


if __name__ == '__main__':
    train = Train(anchors=get_anchors(cfg.ANCHOR_FILE_PATH), batch_size=10, device=torch.device('cuda:0'))
    train.run(epoch_num=30, warm_epoch_num=2, parameters_save_path='./saved_model/parameters.pkl')
