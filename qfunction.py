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
import torch.nn.functional as F

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
        x_tag = x_tag.unsqueeze(-1)
        x_tag = self.lin1(x_tag)
        edge_attr = F.relu(self.lin4(edge_attr))
        return self.propagate(edge_index=edge_index, x=x, x_tag=x_tag, edge_attr=edge_attr)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x_tag, edge_attr, x, edge_index):
        p1 = x_tag
        p2 = self.lin2(aggr_out)
        p3 = scatter_("add", edge_attr, edge_index[0], dim_size=x.shape[0])
        return F.relu(p1+p2+p3)

# Q function
class Q_Fun(nn.Module):
    def __init__(self, in_dim, hid_dim, T, ALPHA):
        super(Q_Fun, self).__init__()
        Linear = partial(nn.Linear, bias=False)
        self.lin5 = Linear(2*hid_dim, 1)
        self.lin6 = Linear(hid_dim, hid_dim)
        self.lin7 = Linear(hid_dim, hid_dim)
        self.S2Vs = [S2V(in_dim=in_dim, out_dim=hid_dim)]
        self.T    = T
        for i in range(T-1):
            self.S2Vs.append(S2V(hid_dim, hid_dim))

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)

        self.device = torch.device("cpu" if torch.cuda.is_available() 
                                   else "cpu")
        self.to(self.device)

    def forward(self, x, edge_index, edge_attr, x_tag):
        # x has shape [N, in_dim]
        # x_tag has shape [N, 1] bool value
        # edge_index has shape [E, 2]
        # edge_attr has shape [E, 1]

        print(x_tag.shape)
        for i in range(self.T):
            x = self.S2Vs[i](x, x_tag, edge_index, edge_attr)

        graph_pool = self.lin6(torch.sum(x, dim=0, keepdim=True))
        nodes_vec = self.lin7(x)
        Cat = torch.cat((graph_pool.repeat(nodes_vec.shape[0], 1), nodes_vec), 
                        dim=1)
        Q   = self.lin5(F.relu(Cat)) # has shape [N, 1]

        return Q
