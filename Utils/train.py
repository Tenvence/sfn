import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.loss import Loss
from Model.lr_scheduler import LinearCosineScheduler
from Model.net import Net


class Train:
    def __init__(self, dataset, batch_size, verify_dataset=None, device=torch.device('cpu'), iou_thresh=0.5, name=None):
        self.anchors = dataset.s_anchors, dataset.m_anchors, dataset.l_anchors
        self.device = device

        self.data_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=16)

        self.is_verify = verify_dataset is not None
        self.verify_dataset = verify_dataset

        self.model = Net(self.anchors)
        self.model.train()

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).to(device=self.device)

        self.optimizer = Adam(self.model.parameters(), weight_decay=0.0005)
        self.criterion = Loss(self.anchors, input_size=dataset.input_size, iou_thresh=iou_thresh)

        self.iou_thresh = iou_thresh
        self.name = name

    def run(self, epoch_num, warm_epoch_num, output_dict):
        steps_per_epoch = len(self.data_loader)
        warm_steps = warm_epoch_num * steps_per_epoch
        total_steps = epoch_num * steps_per_epoch
        lr_scheduler = LinearCosineScheduler(self.optimizer, warm_steps, total_steps, max_lr=0.02)

        verify_record_file = output_dict + '/verify_record.txt'
        if os.path.exists(verify_record_file):
            os.remove(verify_record_file)

        step_idx = 0
        for epoch in range(epoch_num):
            process_bar = tqdm(self.data_loader)
            loss_arr = []
            for d in process_bar:
                crossed_image, single_image, s_gt_tensor, m_gt_tensor, l_gt_tensor, s_gt_coords, m_gt_coords, l_gt_coords = d

                crossed_image = crossed_image.to(device=self.device)
                single_image = single_image.to(device=self.device)
                s_gt_tensor = s_gt_tensor.to(device=self.device)
                m_gt_tensor = m_gt_tensor.to(device=self.device)
                l_gt_tensor = l_gt_tensor.to(device=self.device)
                s_gt_coords = s_gt_coords.to(device=self.device)
                m_gt_coords = m_gt_coords.to(device=self.device)
                l_gt_coords = l_gt_coords.to(device=self.device)

                self.optimizer.zero_grad()
                s_output, m_output, l_output = self.model(crossed_image, single_image)
                loss = self.criterion(s_output, m_output, l_output, s_gt_tensor, m_gt_tensor, l_gt_tensor, s_gt_coords, m_gt_coords,
                                      l_gt_coords)
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step(step_idx)

                step_idx += 1

                if torch.isnan(loss):
                    return False

                loss_arr.append(float(loss))
                mean_loss = sum(loss_arr) / len(loss_arr)

                process_bar.set_description("  %s, Epo: %d/%d, mL: %.3f, L: %.3f" % (self.name, epoch + 1, epoch_num, mean_loss, loss))
            torch.save(self.model, os.path.join(output_dict, 'model.pkl'))

            if self.is_verify:
                pass
                # verify = Test(self.verify_dataset, os.path.join(output_dict, 'model.pkl'), device=self.device)
                # verify.run(output_dict)

        return True
