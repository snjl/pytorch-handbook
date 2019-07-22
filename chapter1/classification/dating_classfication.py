import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# min-max归一化更易懂的形式
def autoNorm2(dataSet):
    # 标0表示取每列最小/大，标1表示取每行最小/大，这里实质上是min-max归一化
    minVals = dataSet.min(0)
    maxVlas = dataSet.max(0)
    ranges = maxVlas - minVals
    normDataSet = (dataSet - minVals) / ranges

    return torch.from_numpy(normDataSet).float()

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, np.array(classLabelVector)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)      # activation function for hidden layer
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.out(x)
        return x


# 读取数据
datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
x = autoNorm2(datingDataMat)
y = torch.from_numpy(datingLabels).long()
# 分类要从0开始，否则会报错RuntimeError: Assertion `cur_target >= 0 && cur_target < n_classes' failed.
y = y - 1
# n_output表示输出，有5类就会输出[x1,x2,x3,x4,x5]，其中只有一个为1，表示分到该类，实际上是为独热编码
net = Net(n_feature=3, n_hidden=20, n_output=3)     # define the network
print(net)  # net architecture

# optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-4, alpha=0.9)

# 分类问题用交叉熵，里面自带softmax概率计算，例如有5类输出，则他输出的5个概率之和为1，其中最大的即表示他分类结果
# 通过预测结果和真实结果的独热编码进行交叉熵计算
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

# plt.ion()   # something about plotting

for t in range(100000):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        # plt.cla()
        # torch.max(a,0) 返回每一列中最大值的那个元素，且返回其索引（返回最大元素在这一列的行索引）
        # torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.numpy()
        target_y = y.numpy()
        # plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        print(accuracy)
        # plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        # plt.pause(0.1)

# plt.ioff()
# plt.show()


# torch.save(net.state_dict(), './mofan_classification.pth')
