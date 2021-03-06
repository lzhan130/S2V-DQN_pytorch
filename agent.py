"""
Created: 2019/10/07
Author : Philip Gao

Beijing Normal University
"""

from qfunction import Q_Fun
from replay_buffer import ReplayBuffer

import torch
from torch_scatter import scatter_max
import numpy as np

class Agent(object):
    def __init__(self, 
                 epsilon=0.99, gamma=0.99, batch_size=128, ALPHA=0.1,
                 in_dim=3, hid_dim=64, T=5,
                 max_size=100000, num_nodes=50):
        
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        self.Q = Q_Fun(in_dim, hid_dim, T, ALPHA)
        self.memory = ReplayBuffer(max_size)

    def choose_action(self, mu, edge_index, edge_w, state):
        """
        edge_index: [E, 2]
        state: [N, 1]
        """
        mu = torch.tensor(mu).to(self.Q.device)
        edge_index = torch.tensor(edge_index).to(self.Q.device)
        edge_w = torch.tensor(edge_w).to(self.Q.device)
        state = torch.tensor(state).to(self.Q.device)

        Q = self.Q(mu, state, edge_index, edge_w)
        # make sure select new nodes
        if np.random.rand() > self.epsilon:
            while True:
                action = np.random.choice(self.num_nodes, size=1)
                if state[int(action)] == 0:
                    break
        else:
            q_value, q_action = torch.sort(Q, descending=True)
            for action in q_action:
                if state[action] == 0:
                    break
            action = action.item()
        return [int(action)]
    
    def remember(self, *args):
        self.memory.store_transition(*args)

    def learn(self):
        #print("**********")
        if self.memory.mem_cntr < self.batch_size:
            return
        graph_batch = self.memory.sample_buffer(self.batch_size)
        graph_batch.to(self.Q.device)

        for i in range(1):
            self.Q.optimizer.zero_grad()
            
            mu          = graph_batch.x_attr
            state       = graph_batch.state
            new_state   = graph_batch.new_state
            edge_index  = graph_batch.edge_index
            edge_w      = graph_batch.edge_w
            action      = graph_batch.action
            done        = graph_batch.done
            reward_sum  = graph_batch.reward
            batch_index = graph_batch.batch

            num_nodes   = self.num_nodes 
            batch_size  = self.batch_size

            y_target = reward_sum + self.gamma * done * self._max_Q(mu, new_state, edge_index, edge_w, batch_index, num_nodes, batch_size)
            y_pred   = self.Q(mu, state, edge_index, edge_w, batch_index, num_nodes, batch_size)[action]

            loss = torch.mean(torch.pow(y_target-y_pred, 2))
            loss.backward()
            self.Q.optimizer.step()


    def _max_Q(self, *args):
        Q = self.Q(*args) #[batch_size*N]
        batch_index = args[4]
        return scatter_max(Q, batch_index)[0].detach()
