import torch
import netron

model = torch.load('../weights/best_resnet.pth', map_location="cuda:0")
dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
input_names = ['input']
output_names = ['output']
onnx_path = 'resnet50.onnx'
torch.onnx.export(model, dummy_input, onnx_path, verbose=True, input_names=input_names, output_names=output_names)

#　模型可视化
netron.start(onnx_path)

def onnx2trt():
    '''
    使用TensorRT/bin/下的　trtexec　来转换模型,
    :return:
    '''
    pass