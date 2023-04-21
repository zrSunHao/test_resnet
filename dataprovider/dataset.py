import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

op_train ='train'
op_val = 'val'
op_test = 'test'

class DogCat(data.Dataset):

    # root：数据集存储路径 transformd：数据增强 mode：数据集用途
    def __init__(self, root, mode=None):
        super().__init__()

        assert mode in [op_train, op_val, op_test]
        self.mode = mode
        # 获取所有图像的地址
        imgs = [f'{root}/{img}' for img in os.listdir(root)]
        # 按图像名称进行排序
        if self.mode == op_test:         # 1.jpg
            imgs = sorted(imgs, key=lambda x:
                          int(x.split('.')[-2].split('/')[-1])
                          )
        else:                           # cat.0.jpg
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)

        # 划分训练集、验证集、测试集，验证:训练 = 2:8
        if self.mode == op_test:
            self.imgs = imgs
        if self.mode == op_train:
            self.imgs = imgs[:int(0.8*imgs_num)]
        if self.mode == op_val:
            self.imgs = imgs[int(0.8*imgs_num):]

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        # 测试集和验证集不需要数据增强
        if self.mode == op_test or self.mode == op_val:
            self.transforms = T.Compose([
                T.Resize(360),          # 按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
                T.CenterCrop(224),      # CenterCrop 把图像按照中心随机切割成224正方形大小的图片
                T.ToTensor(),           # 转换为tensor格式
                normalize               # 对像素值进行归一化处理
            ])
        else:  # 训练集需要数据增强
            self.transforms = T.Compose([
                T.Resize(360),
                T.RandomResizedCrop(224),       # 随机长宽比裁剪
                T.RandomHorizontalFlip(),       # 依概率p水平翻转
                T.ToTensor(),
                normalize
            ])

    # 返回一张图像的数据和标签

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.mode == op_test:   # 1.jpg
            label = int(img_path.split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    # 返回数据集中所有图像的数量

    def __len__(self):
        return len(self.imgs)
