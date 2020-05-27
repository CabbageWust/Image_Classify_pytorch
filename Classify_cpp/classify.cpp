#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <stdio.h>
#include <map>
#include <algorithm>
#include<ctime>
#define CLK_TCK 18.2

using namespace std;
using namespace cv;


int main()
{

   // load model
    torch::DeviceType device_type;
    device_type = torch::kCPU;
    if (torch::cuda::is_available())
    {
        device_type = torch::kCUDA;
    }
    else
    {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    torch::jit::script::Module module = torch::jit::load("/home/yinliang/works/pytorch_learn/Image_Classify_pytorch/resnet.pt");
    module.to(device);
    std::cout<<"load model success"<<std::endl;

    double time0=static_cast<double>(getTickCount());
    for (int k=0; k<1000; k++){

        Mat img = imread("/home/yinliang/works/video_down/wangzhe/240.jpg");
        int img_size = 224;
        Mat img_resized = img.clone();
        resize(img, img_resized,Size(img_size, img_size));

        Mat img_float;
        img_resized.convertTo(img_float, CV_32F, 1.0f / 255.0f);   //归一化到[0,1]区间
        auto tensor_image = torch::from_blob(img_float.data, {1, img_size, img_size, 3}, torch::kFloat32);  //对于一张图而言可使用此函数将nhwc格式转换成tensor
        tensor_image = tensor_image.permute({0, 3, 1, 2});//调整通道顺序,将nhwc转换成nchw

        tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
        tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
        tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);

        tensor_image = tensor_image.to(at::kCUDA);   //将tensor放进GPU中处理

        torch::Tensor out_tensor = module.forward({tensor_image}).toTensor();  //前向计算
        auto results = out_tensor.sort(-1, true);

        auto softmaxs = std::get<0>(results)[0].softmax(0);
        auto indexs = std::get<1>(results)[0];
        auto idx = indexs[0].item<int>();
        string labels[2] = {"normal", "pk"};
        string label = labels[idx];
        float confidence = softmaxs[0].item<float>() * 100.0f;
        cout<<"label:"<<label<<"   confidence:"<<confidence<<endl;
        }
    time0=((double)getTickCount()-time0)/getTickFrequency();
    cout << "time consume: " << time0 << endl;
    return 0;



}
