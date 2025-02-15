import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 多元线性回归
# 设置为 TkAgg 否则在 mac 上面可能不能显示出来
# matplotlib.use('TkAgg')


startTime = time.time()
is_cuda = torch.cuda.is_available()

print("is support cuda:{}".format(is_cuda))

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
x_train = np.c_[x_train, x_train]
x_train = np.c_[x_train, x_train]

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)

y_train = torch.from_numpy(y_train)


# Linear Regression Model
# 定义线性回归模型
# 所有的模型都要继承这个类 nn.Module
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        # 这里的nn.Linear表示的是 z=w1*x1+w2*x2+b，1，表示的是x是2维，z是1维
        # 默认bias为True，可以设置为False
        self.linear = nn.Linear(4, 1)

    # 定义前向传播
    def forward(self, x):
        out = self.linear(x)
        return out


# 创建 model 对象
model = LinearRegression()

# 定义 loss/误差/损失函数/loss 函数 反正都一个意思
criterion = nn.MSELoss()  # 均方差/最小二次方
# 定义优化函数/优化器
# 注意需要将model的参数model.parameters()传进去让这个函数知道他要优化的参数是那些
# lr --> learning rate 学习率  1e-4 --> 0.00001
optimizer = optim.SGD(model.parameters(), lr=1e-4)  # 随机梯度下降

# 开始训练
num_epochs = 10000  # 训练次数

for epoch in range(num_epochs):
    # 输入参数
    inputs = Variable(x_train)
    # 目标参数
    target = Variable(y_train)

    # forward
    out = model(inputs)  # 前向传播
    loss = criterion(out, target)  # 计算损失函数/ loss/误差

    # backward 反向传播
    optimizer.zero_grad()  # 梯度归零 注: 在每次反向传播的时候需要将参数的梯度归零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if (epoch + 1) % 20 == 0:  # 每20次打印一下
        print('Eopch[{}/{}],loss:{:.6f}'.format(epoch + 1, num_epochs, loss.item()))

# 结束训练

# 开始验证
model.eval()  # 让model变成测试模式，这主要是对dropout和batch normalization的操作在训练和测试的时候是不一样的
predict = model(Variable(x_train))  # 开始预测
predict = predict.data.numpy()  # 转化为 numpy 格式

endTime = time.time()

print('consuming time:{}'.format(endTime - startTime))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x_train.numpy()[:, 0], x_train.numpy()[:, 1], y_train.numpy(), 'ro', label='Original data')
ax.scatter(x_train.numpy()[:, 0], x_train.numpy()[:, 1], predict, label='Fitting Line')
# 显示图例
plt.legend()
plt.show()

torch.save(model.state_dict(), './mul_regression_model.pkl')

# 参数查看
params = list(model.named_parameters())
print(params)



