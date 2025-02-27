import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set, GATConv, GatedGraphConv
from torch_geometric.nn import global_mean_pool

class net(torch.nn.Module):
    def __init__(self, num_features, dim):
        # super(Net05, self).__init__()
        # num_features 节点特征维度
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(3, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)
        self.gat = GATConv(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(dim, dim)

        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data,add_des):
        out = F.relu(self.lin0(data.x))
        # h = out.unsqueeze(0)
        for i in range(1):
            m1 = F.relu(self.conv(out, data.edge_index, data.edge_attr))
        # for i in range(4):
        #     m2 = F.relu(self.conv(out, data.edge_index, data.edge_attr))
        m1 = F.relu(self.gat(m1, data.edge_index))
        for i in range(2):
            m3 = F.relu(self.conv(m1, data.edge_index, data.edge_attr))
        # for i in range(2):
        #     m4 = F.relu(self.conv(out, data.edge_index, data.edge_attr))
        m3 = F.relu(self.gat(m3, data.edge_index))
        for i in range(3):
            m5 = F.relu(self.conv(m3, data.edge_index, data.edge_attr))
        m5 = F.relu(self.gat(m5, data.edge_index))
        out = torch.mean(torch.stack([m1, m3, m5]), 0)
        out = global_mean_pool(out, data.batch)
        if add_des:
            out_des = data.des
        # print(out_des.shape)
            out = torch.cat((out, out_des), 1)
        # print(out.shape)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        out = out.view(-1)  # 转化为一维
        return out