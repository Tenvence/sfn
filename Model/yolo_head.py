import torch
import torch.nn as nn


class YoloHead(nn.Module):
    def __init__(self):
        super(YoloHead, self).__init__()

    def forward(self, output: torch.Tensor):
        pass
