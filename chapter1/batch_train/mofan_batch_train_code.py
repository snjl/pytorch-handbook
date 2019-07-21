import torch
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible
# 表示每个batch有5个数据
BATCH_SIZE = 8
# BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)


# x表示data_tensor,y表示target_tensor
torch_dataset = Data.TensorDataset(x, y)
#
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)


def show_batch():
    # 训练3个epoch
    for epoch in range(3):   # train entire dataset 3 times
        # 每次训练把所有数据拆分,每批数量为BATCH_SIZE，如果一共有n个数据，则传入n/BATCH_SIZE后结束该轮训练
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data,输出第几轮，第几步（即一个batch传入了几次数据），传入数据的x和y
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()

