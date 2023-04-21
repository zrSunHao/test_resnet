import os

import torch as t
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from configs import DefaultCfg
from models import ResNet34
from dataprovider import DogCat

cfg = DefaultCfg()

# 1、预处理数据，加载数据
train_dataset = DogCat(root=cfg.data_root+cfg.train_dir, mode=cfg.op_train)
trainloader = DataLoader(train_dataset,
                       batch_size=cfg.batch_size,
                       shuffle=True,
                       num_workers=cfg.num_workers)
val_dataset = DogCat(root=cfg.data_root+cfg.train_dir, mode=cfg.op_val)
valloader = DataLoader(val_dataset,
                       batch_size=cfg.batch_size,
                       shuffle=True,
                       num_workers=cfg.num_workers
                       )

# 2、实例化模型
net = ResNet34(num_classes=len(cfg.classes))
model_path = cfg.models_root + cfg.net_path
if os.path.exists(model_path):
    state_dict = t.load(model_path)
    net.load_state_dict(state_dict)
net.to(cfg.device)

# 3、声明优化器，随机梯度下降，学习率为0.1，动量0.9
optimizer = t.optim.SGD(params=net.parameters(), lr=cfg.lr, momentum=0.9)
# 4、声明损失函数，交叉熵损失函数
criterion = t.nn.CrossEntropyLoss()

def train(net, trainloader, epoch, total):
    running_loss = 0.0
    for i, (imgs, labels) in enumerate(trainloader):                                                       # 1、输入数据
        imgs = imgs.to(cfg.device)   # 数据转移到GPU上
        labels = labels.to(cfg.device)
        optimizer.zero_grad()                                                       # 2、梯度清零
        outputs = net(imgs)                                                       # 3、前向传播
        loss = criterion(outputs, labels)                                       # 4、计算损失
        loss.backward()                                                                # 5、反向传播
        optimizer.step()                                                                # 6、更新网络参数
 
        # 打印 log 信息
        running_loss += loss.item()
        if (i+1) % 10 == 0:                                                       # 每 2000 个batch 打印一次训练状态
            msg = '[%d, %5d/%5d] loss: %.3f'\
                % (epoch, (i+1)*cfg.batch_size, total, running_loss/10)
            running_loss = 0.0
            print(msg)

def val(net, valloader, epoch):
    net.eval()
    acc_sum = 0
    loss_sum = 0
    num = len(valloader)
    with t.no_grad():
        for idx, (data, labels) in enumerate(valloader):
            input = data.to(cfg.device)
            target = labels.to(cfg.device)
            output = net(input)

            loss = criterion(output, target)                        # 计算误差
            loss_sum += loss

            predicted = t.max(output.data, 1).indices               # 每行最大值的索引
            acc_pred = t.sum(predicted == target).data              # 预测正确的个数
            acc_sum += acc_pred.tolist()
    acc_ratio = val_print(epoch, num, loss_sum, acc_sum)
    return acc_ratio

def val_print(epoch, num, loss_sum, acc_sum):
    loss_ave = loss_sum/num
    loss_print = format(loss_ave, '.5f')
    acc_ratio = acc_sum / (num*cfg.batch_size)
    acc_print = format(acc_ratio*100, '.2f')
    epo = str(epoch).ljust(2, " ")
    msg = f'epoch:{epo}/{cfg.max_epoch}  ==========>  loss:{loss_print}  accuracy:{acc_print}%'
    print(msg)
    return acc_ratio


# 5、训练
total = len(train_dataset)
epochs = range(cfg.max_epoch)
for epoch in iter(epochs):
    if epoch+1 < cfg.cur_epoch:
        continue
    train(net, trainloader, epoch+1, total)
    val(net, valloader, epoch+1)

    if (epoch+1) % cfg.save_every == 0:
        model_path = '%s/resnet34_%s.pth'% (cfg.models_root, str(epoch+1))
        t.save(net.state_dict(), model_path)
    


print('Finished Traning')