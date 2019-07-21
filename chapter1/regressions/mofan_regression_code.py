import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 200), dim=1)  # x data (tensor), shape=(100, 1)
y = 0.2 * x.pow(3) - 0.4 * x.pow(2) + 0.3*x + 0.01 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
# y = x

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


# 继承torch.nn.Module，需要重写init和forward，forward为前向传播
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        # 继承Net的init
        super(Net, self).__init__()
        # n_feature表示输入维度，n_hidden表示隐藏层输出维度
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        # self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer2

        # 预测神经层，输入为上一层输出，然后输出结果
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer
    # [计算+激活]为一组固定操作，n组操作后进行最后一次计算作为输出
    def forward(self, x):
        # x通过self.hidden后输出隐藏层值，通过relu激活函数
        x = self.hidden(x)
        x = F.relu(x)  # activation function for hidden layer
        # x = self.hidden2(x)
        # x = F.relu(x)

        # 将上一次的输出作为输出层输入，由于前面已经使用激励函数，所以输出值必定已经给予一定截断，所以在此不需要截断效果
        x = self.predict(x)  # linear output
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)  # define the network
print(net)  # net architecture

# optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.2)
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.9)

loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()  # something about plotting

for t in range(200):
    prediction = net(x)  # input x and predict based on x
    print(prediction)
    loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

    # 三个优化步骤，梯度清除，loss反向传播，optimizer以学习效率lr优化梯度
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.numpy(), y.numpy())
        plt.plot(x.numpy(), prediction.data.numpy(), 'r-', lw=1)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
