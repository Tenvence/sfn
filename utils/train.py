import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.loss import Loss
from model.lr_scheduler import LinearCosineScheduler
from model.yolov3_net import Yolov3Net


class Train:
    def __init__(self, anchors, dataset, input_size, batch_size, device=torch.device('cpu')):
        self.anchors = anchors
        self.device = device

        self.data_loader = DataLoader(dataset, batch_size, shuffle=True)

        self.model = Yolov3Net(anchors)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).to(device=self.device)
        self.model.train()

        self.optimizer = Adam(self.model.parameters(), weight_decay=0.0005)
        self.criterion = Loss(self.anchors, input_size=input_size)

    def run(self, epoch_num, warm_epoch_num, output_path):
        steps_per_epoch = len(self.data_loader)
        warm_steps = warm_epoch_num * steps_per_epoch
        total_steps = epoch_num * steps_per_epoch
        lr_scheduler = LinearCosineScheduler(self.optimizer, warm_steps, total_steps)

        step_idx = 0
        for epoch in range(epoch_num):
            process_bar = tqdm(self.data_loader)
            for d in process_bar:
                crossed_image, single_image, s_gt_tensor, m_gt_tensor, l_gt_tensor, s_gt_coords, m_gt_coords, l_gt_coords = d

                self.optimizer.zero_grad()
                s_output, m_output, l_output = self.model(crossed_image)
                loss = self.criterion(s_output, m_output, l_output, s_gt_tensor, m_gt_tensor, l_gt_tensor, s_gt_coords, m_gt_coords, l_gt_coords)
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step(step_idx)

                step_idx += 1

                process_bar.set_description("epoch: %d, loss: %.2f" % (epoch + 1, float(loss)))

        torch.save(self.model.to(torch.device('cpu')).state_dict(), output_path)
