import torch
import net.backbone as backbone
import net.yolov3_net as yolov3_net

darknet53 = backbone.Darknet53()
yolov3_net = yolov3_net.Yolov3Net(num_class=6)

x = torch.randn(1, 3, 608, 608)

r1, r2, r3 = darknet53(x)

print('r1:', r1.shape)
print('r2:', r2.shape)
print('r3:', r3.shape)

lobj_out, mobj_out, sobj_out = yolov3_net(x)

print('lobj:', lobj_out.shape)
print('mobj:', mobj_out.shape)
print('sobj:', sobj_out.shape)
