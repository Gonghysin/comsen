import os
import csv
import torch
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import preprocessing
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
from ff_gnn import net

# torch.manual_seed(0)
dim = 32
path = "***.pth"
dataset = torch.load(path)

std = 0
mean = 0


# 划分数据集
train_dataset, test_dataset = train_test_split(
    dataset, test_size=0.2, train_size=0.8, shuffle=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset[0:200], batch_size=32, shuffle=False)
val_loader = DataLoader(test_dataset[200:], batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net05(9, dim, True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)


def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y_mess)  # 和计算值求loss
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
        return loss_all / len(train_dataset)


def test(loader):
    model.eval()
    errorexp = 0
    errorcomp = 0
    a, b = 0.5
    for data in loader:
        # MAE 和实验值求误差
        errorexp += ((model(data) - a*data.y_exp)/b).abs().sum().item()
        # MAE 和计算值求误差
        errorcomp += ((model(data) - data.y)/b).abs().sum().item()
    return errorexp / len(loader.dataset), errorcomp / len(loader.dataset)


best_val_error = None
test_errorexplist = []
test_errorcomplist = []
for epoch in range(1, 120):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error, val_error_exp = test(val_loader)
    scheduler.step(val_error_exp)

    # if best_val_error is None or val_error <= best_val_error:
    test_errorexp, test_errorcomp = test(test_loader)
    best_val_error = val_error
    test_errorexplist.append(test_errorexp)
    test_errorcomplist.append(test_errorcomp)
    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
          'Testcomp MAE: {:.7f},Testexp MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_errorcomp, test_errorexp))
# print(test_errorcomp)
# torch.save(test_errorlist, "./para/Net04_特征金字塔_mean.pth")
plt.plot(range(1, len(test_errorcomplist) + 1), test_errorcomplist)
plt.xlabel("epoches")
plt.ylabel("MAE")
plt.show()