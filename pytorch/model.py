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
            
            # 5 Pooling layer
            # batch_size * 4*4*512
            # nn.MaxPool2d(kernel_size=2, stride=2))
        
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
