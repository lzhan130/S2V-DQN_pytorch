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
        # mu has shape [batch_size, N, in_dim]
        # x has shape [batch_size, N, 1]
        # edge_index has shape [batch_size, E, 2]
        # edge_w has shape [batch_size, E, 1]
        batch_size, N, in_dim = mu.shape
        
        #first part of eq. 3
        x = self.lin1(x)        

        # second part of eq. 3
        mu_j =  [mu[b, edge_index[b][:, 1], :] for b in range(batch_size)]
        ## |_ [batch_size, N, in_dim]
        mu_aggr = torch.stack([scatter_add(mu_j[b], edge_index[b][:, 1], dim=0, out=mu_j[b].new_zeros(N, self.in_dim)) for b in range(batch_size)])
        mu_aggr = self.lin2(mu_aggr) 

        # third part of eq.3
        edge_w = [F.relu(self.lin4(item)) for item in edge_w] 
        ## |_ [batch_size, E, out_dim]
        edge_w_aggr = torch.stack([scatter_add(edge_w[b], edge_index[b][:, 1], dim=0,  out=edge_w[b].new_zeros(N, self.out_dim)) for b in range(batch_size)])
        ## |_[batch_size, N, out_dim]
        edge_w_aggr = self.lin3(edge_w_aggr)

        return F.relu(x + mu_aggr + edge_w_aggr) 

# Q function
class Q_Fun(nn.Module):
    def __init__(self, in_dim, hid_dim, T, ALPHA):
        super(Q_Fun, self).__init__()
        Linear = partial(nn.Linear, bias=False)
        self.lin5 = Linear(2*hid_dim, 1)
        self.lin6 = Linear(hid_dim, hid_dim)
        self.lin7 = Linear(hid_dim, hid_dim)
        self.T    = T
        self.S2Vs = nn.ModuleList([S2V(in_dim=in_dim, out_dim=hid_dim)])
        for i in range(self.T - 1):
            self.S2Vs.append(S2V(hid_dim, hid_dim))

        self.loss = nn.MSELoss
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)

        self.device = torch.device("cuda:1" if torch.cuda.is_available() 
                                   else "cpu")
        self.to(self.device)

    def forward(self, mu, x, edge_index, edge_w):
        # mu has shape [batch_size, N, in_dim]
        # x has shape [batch_size, N, 1]
        # edge_index has shape [batch_size, E, 2]
        # edge_w has shape [batch_size, E, 1]

        for i in range(self.T):
            mu = self.S2Vs[i](mu, x, edge_index, edge_w)

        graph_pool = self.lin6(torch.sum(mu, dim=1, keepdim=True))
        nodes_vec = self.lin7(mu)
        Cat = torch.cat((graph_pool.repeat(1, nodes_vec.shape[1], 1), nodes_vec), 
                        dim=2)
        return self.lin5(F.relu(Cat)).squeeze() #[batch_size, N]

