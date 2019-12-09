import numpy as np
import utils.config as cfg
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from net.yolov3_net import Yolov3Net
from utils.read_file_utils import get_class_list, get_anchors
from utils.gravel_dataset import GravelDataset
from utils.loss import Loss

cla_num = 1 if cfg.IS_ONE_CLASSIFICATION else len(get_class_list(cfg.CLASSES_FILE_PATH))
detection_scale = np.array(cfg.DETECTION_SCALE)
anchors = get_anchors(cfg.ANCHOR_FILE_PATH)
iou_threshold = cfg.IOU_THRESHOLD

gravel_dataset = GravelDataset(is_train=True)
train_dataset_loader = DataLoader(dataset=gravel_dataset, batch_size=2, shuffle=True, num_workers=0)
net = Yolov3Net(num_class=1)
optimizer = Adam(net.parameters())
criterion = Loss(cfg.ANCHOR_PER_SCALE, cla_num, detection_scale, anchors, iou_threshold)

net.train()

for epoch in range(30):
    for crossed_image, single_image, s_tensor, m_tensor, l_tensor, s_coords, m_coords, l_coords in train_dataset_loader:
        optimizer.zero_grad()
        s_output, m_output, l_output = net(crossed_image)

        loss = criterion(s_output, m_output, l_output, s_tensor, m_tensor, l_tensor, s_coords, m_coords, l_coords)
        print('loss:', loss)

        loss.backward()
        optimizer.step()

torch.save(net.state_dict(), './parameter.pkl')
