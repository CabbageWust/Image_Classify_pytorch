# coding:utf-8
import torch

model = torch.load('/home/yinliang/works/pytorch_learn/Image_Classify_pytorch/weights/best_resnet.pkl', map_location="cuda:0")
model.cuda()
img_size = 224

# 使用 torch.jit.trace 生成 torch.jit.ScriptModule 来跟踪

x = torch.rand(1, 3, 224, 224)
x = x.cuda()  # very important
traced_script_module = torch.jit.trace(model, x)
traced_script_module.save("resnet.pt")
