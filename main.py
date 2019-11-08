import torch
from net.backbone_net import Darknet53

x = torch.randn(1, 3, 416, 416)

r1, r2, r3 = Darknet53()(x)

print('r1:', r1.shape)
print('r2:', r2.shape)
print('r3:', r3.shape)
