from torch import nn
from torch.nn import functional as F

'''
实现子module: ResidualBlock
'''
class ResidualBlock34(nn.Module):

    def __init__(self, inchannels, outchannels, stride=1, shortcut=None):
        super().__init__()
        self.conv1 = nn.Conv2d( in_channels=inchannels,
                                out_channels=outchannels,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d( in_channels=outchannels,
                                out_channels=outchannels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.bn2 = nn.BatchNorm2d(outchannels)
        self.right = shortcut


    def forward(self, x):
        left = self.conv1(x)
        left = self.bn1(left)
        left = self.relu(left)
        left = self.conv2(left)
        left = self.bn2(left)

        '''
        残差，
            只有每一层的第一个 ResidualBlock 需要统一通道数，
            其他 ResidualBlock 不需要处理，直接相加即可
        ''' 
        residual = x if self.right is None else self.right(x)
        out = left + residual
        out = F.relu(out)
        return out

