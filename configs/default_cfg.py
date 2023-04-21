class DefaultConfig(object):
    # 基本的配置
    data_root = 'D:/WorkSpace/DataSet/dogs-vs-cats-2/'                   # 数据集的存放路径
    models_root = './checkpoints/'           # 模型存放路径
    train_dir= 'train'                     # 训练集路径
    test_dir= 'test'                        # 验证集路径
    img_size = 512                          # 图像尺寸
    device = 'cuda'                         # 使用的设备 cuda/cpu

    # 训练相关的配置
    max_epoch = 100                         # 最大训练轮次
    cur_epoch = 1                           # 当前训练的轮次，用于中途停止后再次训练时使用
    save_every = 1                          # 每训练多少个 epoch，保存一次模型
    num_workers = 0                         # 多进程加载数据所用的进程数，默认为0，表示不使用多进程
    batch_size = 32                         # 每批次加载图像的数量
    lr = 1e-3                               # 学习率
    net_path = 'resnet34_1.pth'                         # 预训练的判别器模型路径
    classes = ['cat','dog']

    op_train = 'train'                      # 训练操作
    op_val = 'val'                          # 验证操作
    op_test = 'test'                       # 测试操作
    