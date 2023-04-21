from torch import nn
from torch.nn import functional as F

from models.pre_block import PreBlock
from models.residual_block_34 import ResidualBlock34

'''
实现主module: ResNet34
ResNet34 包含多个 layer，每个 layer 又包含多个 Residual Block
用子 module 来实现 Residual Block，用 _make_layer 函数实现 layer
'''
class ResNet34(nn.Module):
    
    def __init__(self, num_classes=1000):
        super().__init__()
        self.pre = PreBlock()
        # 重复的 layer，分别有 3、4、6、3 个 Residual Block
        self.layer1 = self._make_layer(inchannel=64,
                                       outchannel=64,
                                       block_num=3,
                                       stride=1,
                                       is_shortcut=False)
        self.layer2 = self._make_layer(inchannel=64,
                                       outchannel=128,
                                       block_num=4,
                                       stride=2,
                                       is_shortcut=True)
        self.layer3 = self._make_layer(inchannel=128,
                                       outchannel=256,
                                       block_num=6,
                                       stride=2,
                                       is_shortcut=True)
        self.layer4 = self._make_layer(inchannel=256,
                                       outchannel=512,
                                       block_num=3,
                                       stride=2,
                                       is_shortcut=True)
        # 分类用的全链接
        self.classifier = nn.Linear(512,num_classes)


    def forward(self, x):
        out = self.pre(x)
        #print('pre out', out.size())
        out = self.layer1(out)
        #print('layer1 out', out.size())
        out = self.layer2(out)
        #print('layer2 out',out.size())
        out = self.layer3(out)
        #print('layer3 out',out.size())
        out = self.layer4(out)
        #print('layer4 out',out.size())
        out = F.avg_pool2d(out, 7)
        #print('pool out', out.size())
        out = out.view(out.size(0), -1)
        #print('view out',out.size())
        out = self.classifier(out)
        return out



    '''
    构建 layer，包含多个 Residual Block
        block_num：表示残差块的数量
    '''
    def _make_layer(self, inchannel, outchannel, block_num, stride, is_shortcut=True):
        '''
        维度变换，不同的层之间维度可能不同，通过 1*1 的卷积核进行 升为或降维，使得维度统一
        stride 与 layer 的第一个 residual block 的 stride 保持一致，这样特征图的尺寸会保持一致
        '''
        if is_shortcut:
            conv = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(outchannel)
            shortcut = nn.Sequential(conv, bn)
        else:
            shortcut = None

        layers = []
        '''
        每一层的，第一个 residual block 
            需要shortcut，进行维度变换，方便之后相加
            指定步长
        '''
        b1 = ResidualBlock34(inchannel, outchannel, stride, shortcut)
        layers.append(b1)

        for i in range(1, block_num):
            b2 = ResidualBlock34(outchannel, outchannel, 1, None)
            layers.append(b2)
        
        seq = nn.Sequential(*layers)
        return seq


