import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

file_path = "iris.csv"
df_iris = pd.read_csv(file_path, sep=",", header="infer")
np_iris = df_iris.values
np.random.shuffle(np_iris)
species = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
for i in range(np_iris.shape[0]):
    np_iris[i, -1] = species[np_iris[i, -1]]
np_iris = np_iris.astype('float64')


# min-max归一化更易懂的形式
def autoNorm2(dataSet):
    # 标0表示取每列最小/大，标1表示取每行最小/大，这里实质上是min-max归一化
    minVals = dataSet.min(0)
    maxVlas = dataSet.max(0)
    ranges = maxVlas - minVals
    normDataSet = (dataSet - minVals) / ranges
    return torch.from_numpy(normDataSet).float()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)  # activation function for hidden layer
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.out(x)
        return x

# 训练集，测试集划分
train_percent = 0.8
train_num = int(train_percent * len(np_iris))
train_xy = np_iris[0:train_num]
test_xy = np_iris[train_num:]
x = autoNorm2(train_xy[:, 0:-1])
y = torch.from_numpy(train_xy[:, -1]).long()
test_x = autoNorm2(test_xy[:, 0:-1])
test_y = torch.from_numpy(test_xy[:, -1]).long()

# n_output表示输出，有5类就会输出[x1,x2,x3,x4,x5]，其中只有一个为1，表示分到该类，实际上是为独热编码
net = Net(n_feature=4, n_hidden=20, n_output=3)  # define the network
print(net)  # net architecture

# optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.9)

# 分类问题用交叉熵，里面自带softmax概率计算，例如有5类输出，则他输出的5个概率之和为1，其中最大的即表示他分类结果
# 通过预测结果和真实结果的独热编码进行交叉熵计算
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

# plt.ion()   # something about plotting

for t in range(10000):
    out = net(x)  # input x and predict based on x
    loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if t % 5 == 0:
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.numpy()
        target_y = y.numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        print(accuracy)


        # 测试集准确率输出
        net.eval()  # 让model变成测试模式，这主要是对dropout和batch normalization的操作在训练和测试的时候是不一样的
        predict = net(test_x)  # 开始预测
        predict = torch.max(predict, 1)[1]
        predict = predict.numpy()  # 转化为 numpy 格式
        target_test_y = test_y.numpy()
        real_acc = float((predict == target_test_y).astype(int).sum()) / float(target_test_y.size)
        print('real acc is', real_acc)
        net.train()


