"""卷积神经网络，数据库为MNIST"""
# 使用三层CNN

# 第三方库
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import random
import time

# torch.manual_seed(1)    # reproducible

'''超参数'''
EPOCH = 1  # 训练整批数据多少次，为节约时间设为1，即只训练一次
BATCH_SIZE = 50  # 每批次图像的多少
LR = 0.001  # 学习率
DOWNLOAD_MNIST = False  # 不需要下载MNIST数据集，如果需要下载则设为True

'''处理数据集为可被使用的格式'''
# 处理训练数据
train_data = torchvision.datasets.MNIST(
    root='./data/',  # 训练数据集的存放目录
    train=True,  # 表明这是训练数据
    transform=torchvision.transforms.ToTensor(),  # 把 PIL.Image 或者 numpy.array 转化成
    # torch.FloatTensor of shape (C x H x W)
    # 训练时normalrize在 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,  # 判断是否需要下载MNIST数据集
)

# 分批训练，一次输入五十个样本，每个样本有一个通道，大小为28*28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 处理测试数据，注：为节约时间，只提取前两千个进行测试
test_data = torchvision.datasets.MNIST(
    root='./data/',  # 测试数据集的存放目录
    train=False  # 表明这是测试数据
)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255.
# 把测试数据规格从 (2000, 28, 28) 变成 (2000, 1, 28, 28), 取值在(0,1)范围内
test_y = test_data.test_labels[:2000]   # 测试数据对应的标签，因为是是个数字所以为0~9

'''定义CNN类'''


# 定义CNN类
class CNN(nn.Module):  
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层
        # # 输入数据的形状为一个通道，边长为28 (1, 28, 28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 输入的通道数为1
                out_channels=16,  # 输出的通道数为16
                kernel_size=5,  # 卷积核的边长
                stride=1,  # 步长
                padding=2,  # 每一个边补零的行数
                # 如果想让图片在卷积前后的长和宽不变，当步长为1时补零的行数padding=(kernel_size-1)/2
            ),  # 输出数据的形状为16个通道，边长为28 (16, 28, 28)
            nn.ReLU(),  # 激活，这里使用ReLU作为激活函数
            nn.MaxPool2d(kernel_size=2),  # 最大池化（取区域内的最大值），输出数据的格式为16个通道，边长为14 (16, 14, 14)
        )
        # 第二个卷积层
        # 输入数据的形状与上一层的输出一样，为 (16, 14, 14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # 输出数据的形状为 (32, 14, 14)
            nn.ReLU(),  # 激活
            nn.MaxPool2d(2),  # 最大池化，输出数据的形状为 (32, 7, 7)
        )
         # 第三个卷积层
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU()
        )
        # 全连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输入数据形状为 (32 * 7 * 7)，输出10个类别（十种数字）

    # 定义前向传播
    def forward(self, x):
        x = self.conv1(x)  # 第一层卷积
        x = self.conv2(x)  # 第二层卷积
        x = self.conv3(x)  # 第三层卷积
        x = x.view(
            x.size(0),  # 保留batch数
            -1  # 把其他三个维度的数据展平
        )  # 展平，把第二个卷积层输出的数据的形状变成 (batch_size, 32 * 7 * 7)
        output = self.out(x)  # 全连接
        return output, x


'''创建一个CNN实例'''
cnn = CNN() 
print(cnn) # 打印出网络结构，不是必需步骤，可以去掉

'''定义用于优化神经网络的函数'''
# 优化函数，用于优化参数，使用Adam作为优化器
optimizer = torch.optim.Adam(
    cnn.parameters(),  # 待优化参数的dict
    lr=LR  # 学习率
)
# 损失函数，这里选用交叉熵损失函数（自带softmax）
loss_func = nn.CrossEntropyLoss()  

'''训练与测试'''
# 训练网络
for epoch in range(EPOCH):
    timeStart = time.time()
    for step, (b_x, b_y) in enumerate(train_loader):  # 加载数据
        # 这里的train_loader是一个索引序列，包括下标step和数据对象(b_x, b_y)

        output = cnn(b_x)[0]  # 放入神经网络处理
        loss = loss_func(output, b_y)  # 计算损失
        optimizer.zero_grad()  # 优化器将梯度归零
        loss.backward()  # 反向传播, 计算梯度
        optimizer.step()  # 将计算出来的梯度应用

        if step % 50 == 0:  # 每50步打印一次训练效果
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()  # 将预测的数字值放入pred_y，形式为一个array
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))  # 计算精确度
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.3f' % accuracy)
    timeEnd = time.time()
    print('run time:',timeEnd-timeStart)

# 测试网络
random_num = random.randint(1, 1000)  # 生成随机数，用于确定测试数据的区间
test_output, _ = cnn(test_x[random_num: random_num + 10])  # 用生成的随机数从测试数据中选出十个进行预测
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[random_num: random_num + 10].numpy(), 'real number')
