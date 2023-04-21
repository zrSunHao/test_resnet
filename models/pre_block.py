from torch import nn
import torch as t
from torch.nn import functional as F

'''
ResNet34 前几层的图像转换
'''
class PreBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3,
                                 stride=2,
                                 padding=1)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        return out