"""
Created : 2019/10/04
Author  : Philip Gao

Beijing Normal University
"""

# This script aims to implement the structure2vec framework used in S2V-DQN.
# Refer to Learning Combinatorial Optimization Algorithms over Graph for more 
# details.

import torch
import torch.nn as nn
import torch.optim as optim
from  torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

from torch_scatter import scatter_add
from functools import partial

class S2V(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(S2V, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        Linear = partial(nn.Linear, bias=False)
        self.lin1 = Linear(1, out_dim)
        self.lin2 = Linear(in_dim, out_dim)
        self.lin3 = Linear(out_dim, out_dim)
        self.lin4 = Linear(1, out_dim)

    def forward(self, mu, x, edge_index, edge_w):
        #first part of eq. 3
        x = self.lin1(x)        

        # second part of eq. 3
        mu_j = mu[edge_index[1, :], :]
        mu_aggr = scatter_add(mu_j, edge_index[1, :], dim=0)
        mu_aggr = self.lin2(mu_aggr) 

        # third part of eq.3
        edge_w = F.relu(self.lin4(edge_w))
        edge_w_aggr = scatter_add(edge_w, edge_index[1, :], dim=0)
        edge_w_aggr = self.lin3(edge_w_aggr)

        return F.relu(x + mu_aggr + edge_w_aggr) 

# Q function
class Q_Fun(nn.Module):
    def __init__(self, in_dim, hid_dim, T, lr, lr_gamma=0.95):
        super(Q_Fun, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        Linear = partial(nn.Linear, bias=False)
        self.lin5 = Linear(2*hid_dim, 1)
        self.lin6 = Linear(hid_dim, hid_dim)
        self.lin7 = Linear(hid_dim, hid_dim)

        self.S2Vs = nn.ModuleList([S2V(in_dim=in_dim, out_dim=hid_dim)])
        for i in range(T - 1):
            self.S2Vs.append(S2V(hid_dim, hid_dim))

        self.loss = nn.MSELoss

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_gamma)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, graph):
        mu = graph.mu
        x  = graph.node_tag
        edge_w = graph.edge_w
        edge_index = graph.edge_index

        for i in range(self.T):
            mu = self.S2Vs[i](mu, x, edge_index, edge_w)
        nodes_vec = self.lin7(mu)

        if "batch" in graph.keys:
            graph_pool = scatter_add(mu, graph.batch, dim=0)[graph.batch]
        else:
            num_nodes = graph.num_nodes
            graph_pool = torch.sum(mu, dim=0, keepdim=True)
            graph_pool = graph_pool.repeat(num_nodes,1)

        Cat = torch.cat((graph_pool, nodes_vec), dim=1)
        return self.lin5(F.relu(Cat)).squeeze()

