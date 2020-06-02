import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.loss import Loss
from Model.lr_scheduler import LinearCosineScheduler
from Model.yolov3_net import Yolov3Net


class Train:
    def __init__(self, dataset, batch_size, device=torch.device('cpu'), iou_thresh=0.5):
        self.anchors = dataset.s_anchors, dataset.m_anchors, dataset.l_anchors
        self.device = device

        self.data_loader = DataLoader(dataset, batch_size, shuffle=True)

        self.model = Yolov3Net(self.anchors)
        self.model.train()

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).to(device=self.device)

        self.optimizer = Adam(self.model.parameters(), weight_decay=0.0005)
        self.criterion = Loss(self.anchors, input_size=dataset.input_size, iou_thresh=iou_thresh)

    def run(self, epoch_num, warm_epoch_num, output_model_path):
        steps_per_epoch = len(self.data_loader)
        warm_steps = warm_epoch_num * steps_per_epoch
        total_steps = epoch_num * steps_per_epoch
        lr_scheduler = LinearCosineScheduler(self.optimizer, warm_steps, total_steps)

        step_idx = 0
        for epoch in range(epoch_num):
            process_bar = tqdm(self.data_loader)
            loss_arr = []
            for d in process_bar:
                crossed_image, single_image, s_gt_tensor, m_gt_tensor, l_gt_tensor, s_gt_coords, m_gt_coords, l_gt_coords = d

                self.optimizer.zero_grad()
                s_output, m_output, l_output = self.model(crossed_image, single_image)
                loss = self.criterion(s_output, m_output, l_output, s_gt_tensor, m_gt_tensor, l_gt_tensor, s_gt_coords, m_gt_coords, l_gt_coords)
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step(step_idx)

                step_idx += 1

                if torch.isnan(loss):
                    return False

                loss_arr.append(float(loss))
                mean_loss = sum(loss_arr) / len(loss_arr)

                process_bar.set_description("  Epoch: %d, mean loss: %.2f, loss: %.2f" % (epoch + 1, mean_loss, float(loss)))
        torch.save(self.model, output_model_path)

        return True
