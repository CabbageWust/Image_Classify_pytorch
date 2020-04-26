'''
Reference: https://github.com/jaxony/ShuffleNet/blob/master/model.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import init
from torchsummary import summary

def conv3x3(in_planes, out_planes, stride=1, padding = 1,
            bias = True, groups = 1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups
    )

def conv1x1(in_planes, out_planes, groups=1):

    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=1,
                     groups=groups)

def channel_shuffle(x, groups):
    '''
    假定将输入层分为 g 组，总通道数为 g x n ,
    首先将通道那个维度拆分为 (g,n) 两个维度,
    然后将这两个维度转置变成 (n,g),
    最后重新reshape成一个维度g x n
    :param x:
    :param groups:
    :return:
    '''

    batchsize, num_channels, height, width = x.data.size()
    n = num_channels // groups

    x = x.view(batchsize, groups, n, height, width)

    # transpose
    # contiguous is required when transpose used before view()
    x = torch.transpose(x, 1, 2).contiguous()

    #flatten
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleUnit(nn.Module):

    def __init__(self, in_planes, out_planes, groups=3,
                 grouped_conv = True, combine = 'add'):
        super(ShuffleUnit, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.groups_conv = grouped_conv
        self.combine = combine
        self.bottleneck_channels = self.out_planes // 4

        #define the type of ShuffleUnit
        if self.combine == 'add':
            self.depthwise_stride = 1
            self._combile_func = self._add
        elif self.combine == 'concat':
            self.depthwise_stride = 2
            self._combile_func = self._concat
            self.out_planes -= self.in_planes

        else:
            raise ValueError('Can not combine tensor ')

        self.first_1x1_groups = self.groups if grouped_conv else 1

        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            self.in_planes,
            self.bottleneck_channels,
            self.first_1x1_groups,
            batch_norm = True,
            relu = True
        )

        # 3x3 depthwise convolution followed by batch normalization
        self.depthwise_conv3x3 = conv3x3(
            self.bottleneck_channels,
            self.bottleneck_channels,
            stride=self.depthwise_stride,
            groups=self.bottleneck_channels
        )
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(

            self.bottleneck_channels,
            self.out_planes,
            self.groups,
            batch_norm = True,
            relu = False
        )

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    @staticmethod
    def _add(x, out):
        return x + out

    def _make_grouped_conv1x1(self, in_planes, out_planes, groups, batch_norm=True, relu=False):

        modules = OrderedDict()

        conv = conv1x1(in_planes, out_planes, groups=groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_planes)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        residual = x

        if self.combine == 'cancat':

            residual = F.avg_pool2d(residual,kernel_size=3, stride = 2, padding = 1)
            out = self.g_conv_1x1_compress(x)
            out = channel_shuffle(out, self.groups)
            out = self.depthwise_conv3x3(out)
            out = self.bn_after_depthwise(out)
            out = self.g_conv_1x1_expand(out)

            out = self._combile_func(residual, out)

            return F.relu(out)

class ShuffleNet(nn.Module):

    def __init__(self, groups, in_planes, num_class = 1000):
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.stage_repeats = [3,7,3]
        self.in_planes = in_planes
        self.num_classes = num_class

        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError('')


        self.conv1 = conv3x3(self.in_planes, self.stage_out_channels[1], stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #stage2
        self.stage2 = self._make_stage(2)
        self.stage3 = self._make_stage(3)
        self.stage4 = self._make_stage(4)

        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, self.num_classes)
        self.init_params()




    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = 'ShuffleUnit_Stage{}'.format(stage)

        grouped_conv = stage > 2

        first_module = ShuffleUnit(
            self.stage_out_channels[stage-1],
            self.stage_out_channels[stage],
            groups=self.groups,
            grouped_conv=grouped_conv,
            combine='concat'
        )
        modules[stage_name + '_0'] = first_module

        for i in range(self.stage_repeats[stage-2]):
            name = stage_name + '_{}'.format(i+1)
            module = ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups = self.groups,
                grouped_conv=True,
                combine = 'add'
            )
            modules[name] = module
        return nn.Sequential(modules)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.avg_pool2d(x, x.data.size()[-2:])

        x = x.view(x.size(), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)




if __name__ == '__main__':
    model = ShuffleNet()
    model.cuda()
    print(model, (3,224,224))



