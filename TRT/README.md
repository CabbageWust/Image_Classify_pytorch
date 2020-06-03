Version:

Pytorch==1.2.0

CUDA 10.2

TensorRT 7.0


1.Convert your trained torch model(.pth) to onnx 

2.Convert onnx model to TensorRT

3. Do inference with trt model


Experiments results:
 
 Using Resnet50 model to computing 1000 images,
 
 With Pytorch Python API: 18 seconds
 
 With Pytorch C++ API: 15 seconds
 
 With TensorRT python API: 10 seconds
 
 With TensorRT C++ API: TODO
    
