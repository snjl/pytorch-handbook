import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



def get_data():
    frTrain = open('horseColicTraining.txt')  # 打开训练集
    frTest = open('horseColicTest.txt')  # 打开测试集
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    return trainingSet, trainingLabels,testSet, testLabels


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


# 读取数据
trainingSet, trainingLabels,testSet, testLabels = get_data()
x = torch.tensor(trainingSet)
y = torch.tensor(trainingLabels).long()
test_x = torch.tensor(testSet)
test_y = torch.tensor(testLabels).long()
# n_output表示输出，有5类就会输出[x1,x2,x3,x4,x5]，其中只有一个为1，表示分到该类，实际上是为独热编码
net = Net(n_feature=21, n_hidden=20, n_output=2)  # define the network
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
# torch.save(net.state_dict(), './mofan_classification.pth')


