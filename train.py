from torch.optim import Adam
from torch.utils.data import DataLoader
from net.yolov3_net import Yolov3Net
from utils.gravel_dataset import GravelDataset
import torch
import torch.nn as nn
import numpy as np

gravel_dataset = GravelDataset(is_train=True)
train_dataset_loader = DataLoader(dataset=gravel_dataset, batch_size=4, shuffle=True, num_workers=0)
net = Yolov3Net(num_class=1)
optimizer = Adam(net.parameters())
criterion = nn.CrossEntropyLoss()

net.train()

for crossed_image, single_image, s_tensor, m_tensor, l_tensor, s_coords, m_coords, l_coords in train_dataset_loader:
    print(crossed_image.shape)
    print(single_image.shape)
    print(s_tensor.shape)
    print(m_tensor.shape)
    print(l_tensor.shape)
    print(s_coords.shape)
    print(m_coords.shape)
    print(l_coords.shape)
    print()

for epoch in range(30):
    pass
# for i in train_dataset_loader:
#     print(i)
# for [crossed_image, single_image], [label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes] in train_dataset_loader:
#     print(crossed_image)
# optimizer.zero_grad()
# lobj_branch_output, mobj_branch_output, sobj_branch_output = net()(crossed_image)
# loss = criterion(crossed_image, single_image)
# optimizer.step()
