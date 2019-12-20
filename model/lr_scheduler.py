import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


class LinearCosineScheduler(LambdaLR):
    def __init__(self, optimizer, warm_steps, total_steps, max_lr=1e-3):
        self.warm_steps = warm_steps
        self.total_steps = total_steps
        self.max_lr = max_lr

        super(LinearCosineScheduler, self).__init__(optimizer, self.compute_lr)

    def compute_lr(self, step):
        if step < self.warm_steps:
            return step / self.warm_steps
        else:
            return self.max_lr + 0.5 * (1 - self.max_lr) * (1 + torch.cos((step - self.warm_steps) / (self.total_steps - self.warm_steps) * np.pi))
