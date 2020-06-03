import torch
import netron
import torchvision
import torch
from torch.autograd import Variable
import onnx
print(torch.__version__)

# torch  -->  onnx
input_name = ['input']
output_name = ['output']
input = Variable(torch.randn(1, 3, 224, 224)).cuda()
# model = torchvision.models.resnet18(pretrained=True).cuda()
model = torch.load('', map_location="cuda:0")
torch.onnx.export(model, input, 'psenet.onnx', input_names=input_name, output_names=output_name, verbose=True)
# 模型可视化
# netron.start('resnet18.onnx')


# onnx  -->  trt
def onnx2trt():
    '''
    使用TensorRT/bin/下的　trtexec　来转换模型,
    trtexec --onnx= --saveEngine=
    :return:
    '''
    pass