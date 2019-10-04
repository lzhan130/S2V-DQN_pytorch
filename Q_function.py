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

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter_

from functools import partial

# Implement Structure2Vec using pytorch geometric
class S2V(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super(S2V, self).__init__(aggr="add")
        Linear = partial(nn.Linear, bias=False)
        self.lin1 = Linear(1, out_dim)
        self.lin2 = Linear(in_dim, out_dim)
        self.lin3 = Linear(in_dim, out_dim)
        self.lin4 = Linear(1, out_dim)

    def forward(self, x, x_tag, edge_index, edge_attr):
        # x has shape [N, in_dim]
        # x_tag has shape [N, 1]
        # edge_index has shape [E, 2]
        # edge_attr has shape [E, 1]
        x_tag = self.lin1(x_tag)
        edge_attr = nn.ReLU(self.lin4(edge_attr))
        return self.propagate(edge_index, x=x, x_tag=x_tag, edge_attr=edge_attr)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x_tag, edge_attr):
        p1 = x_tag
        p2 = self.lin2(aggr_out)
        p3 = edge_attr_agg = scatter_("add", edge_attr, edge_index[0], dim=0,
                dim_size=x.shape[0])
        return nn.ReLU(p1+p2+p3)

# Q function
class Q_Fun(nn.Module):
    def __init__(self, in_dim, hid_dim, T):
        super(Q_Fun, self).__init__()
        Linear = partial(nn.Linear, bias=False)
        self.lin5 = Linear(2*hid_dim, 1)
        self.lin6 = Linear(hid_dim, hid_dim)
        self.lin7 = Linear(hid_dim, hid_dim)
        self.S2Vs = [S2V(in_dim=in_dim, out_dim=hid_dim)]
        self.T    = T
        for i in range(T-1):
            self.S2Vs.append(S2V(hid_dim, hid_dim))

    def forward(self, x, x_tag, edge_index, edge_attr):
        # x has shape [N, in_dim]
        # x_tag has shape [N, 1] bool value
        # edge_index has shape [E, 2]
        # edge_attr has shape [E, 1]

        for i in range(self.T):
            x = self.S2Vs[i][x, x_tag, edge_index, edge_attr)
        Cat = torch.cat((self.lin6(torch.sum(x, dim=0)), self.lin7(x)), dim=1)
        Q   = self.lin5(nn.ReLU(Cat)) # has shape [N, 1]

        return Q
