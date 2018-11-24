import torch
import torch.nn as nn
import torch.nn.functional as F

# def ResNet18():
#     return ResNet(BasicBlock, [2,2,2,2])

# def ResNet34():
#     return ResNet(BasicBlock, [3,4,6,3])

# def ResNet50():
#     return ResNet(BasicBlock, [3,4,6,3])

# def ResNet101():
#     return ResNet(BasicBlock, [3,4,23,3])

# def ResNet152():
#     return ResNet(BasicBlock, [3,8,36,3])


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) #
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), #avgPooling?
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=2350, resnet_layer=56):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.n = 9

        # 64 32 32
        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_0', BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None))
        for i in range(1,self.n):
            self.layer1.add_module('layer1_%d' % (i), BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None))

        # 64 32 32
        self.layer2 = nn.Sequential()
        self.layer2.add_module('layer2_0', BasicBlock(in_channels=64, out_channels=128, stride=2, downsample=True))
        for i in range(1,self.n):
            self.layer2.add_module('layer2_%d' % (i), BasicBlock(in_channels=128, out_channels=128, stride=1, downsample=None))

        # 128 16 16
        self.layer3 = nn.Sequential()
        self.layer3.add_module('layer3_0', BasicBlock(in_channels=128, out_channels=256, stride=2, downsample=True))
        for i in range(1,self.n):
            self.layer3.add_module('layer3_%d' % (i), BasicBlock(in_channels=256, out_channels=256, stride=1, downsample=None))

        # 256 8 8

        self.MSEavgpool = nn.AvgPool2d(kernel_size=2, stride=1)

        self.Class_avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(256, num_classes)


    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        MSE_out = self.MSEavgpool(x)
        MSE_out = MSE_out.view(MSE_out.size(0), 1, 64, 64)

        x = self.Class_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return [x, MSE_out]


        
import torch
import torch.nn as nn
from torch.autograd import Variable

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            
            # 1-1 conv layer
            # batch_size * 3*64*64
            nn.Conv2d(9, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 1-2 conv layer
            # batch_size * 64*64*64
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 1 Pooling layer
            # batch_size * 64*64*64
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            
            # 2-1 conv layer
            # batch_size * 64*32*32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 2-2 conv layer
            # batch_size * 128*32*32
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 2 Pooling lyaer
            # batch_size * 128*32*32
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            
            # 3-1 conv layer
            # batch_size * 128*16*16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 3-2 conv layer
            # batch_size * 256*16*16
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 3 Pooling layer
            # batch_size * 256*16*16
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = nn.Sequential(
            
            # 4-1 conv layer
            # batch_size * 512*8*8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 4-2 conv layer
            # batch_size * 512*8*8
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 4 Pooling layer
            # batch_size * 512*8*8
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = nn.Sequential(
            
            # 5-1 conv layer
            # batch_size * 512*4*4
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 5-2 conv layer
            # batch_size * 512*4*4
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        self.layer6 = nn.Sequential(
            
            # 6 Transpose
            # batch_size * 256*4*4
            nn.ConvTranspose2d(256, 64, kernel_size = 4, stride = 4, padding = 0),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer7 = nn.Sequential(
            
            # 7 Transpose
            # batch_size * 64*16*16
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        
        self.layer8 = nn.Sequential(
            
            # 8 Transpose
            # batch_size * 16*32*32
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh())
            # batch_size * 1*64*64
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
            
        return out
