import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(in_planes, out_planes, stride):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True)
            )

        def conv_dw(in_planes, out_planes, stride):
            return nn.Sequential(
                nn.Conv2d(in_planes, in_planes, 3, stride, 1, groups=in_planes, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)


    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x



