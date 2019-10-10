"""
Created: 2019/10/07
Author : Philip Gao

Beijing Normal University
"""

from qfunction import Q_Fun
from replay_buffer import ReplayBuffer

import torch
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
        self.memory = ReplayBuffer(max_size, num_nodes)

    def choose_action(self, edge_index, edge_w, state):
        """
        edge_index: [E, 2]
        state: [N, 1]
        """
        mu = torch.zeros(self.num_nodes, self.in_dim).unsqueeze(0).to(self.Q.device)
        edge_index = torch.tensor(edge_index).unsqueeze(0).to(self.Q.device)
        edge_w = torch.tensor(edge_w).unsqueeze(0).to(self.Q.device)
        state = torch.tensor(state).unsqueeze(0).to(self.Q.device)

        Q = self.Q(mu, state, edge_index, edge_w)

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_nodes, size=1)
        else:
            action = torch.argmax(Q).item()
        return int(action)
    
    def remember(self, edge_index, edge_w, state, action, reward_sum, new_state, done):
        self.memory.store_transition(edge_index, edge_w, state, action, reward_sum, 
                                     new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        edge_index, edge_w, state, action, reward_sum, new_state, done = self.memory.sample_buffer(self.batch_size)
        device = self.Q.device

        mu = torch.zeros(self.batch_size, self.num_nodes, self.in_dim).to(device)
        edge_index = [torch.tensor(item).to(device) for item in edge_index]
        edge_w = [torch.tensor(item).to(device) for item in edge_w]
        state = torch.tensor(state).to(device)
        action = torch.tensor(action).to(device)
        reward_sum = torch.tensor(reward_sum).to(device)
        new_state = torch.tensor(new_state).to(device)
        done = torch.tensor(done).to(device)
        
        for i in range(10):
            self.Q.optimizer.zero_grad()
            y_target = reward_sum + self.gamma * self._max_Q(mu, new_state, edge_index, edge_w)
            y_pred   = self.Q(mu, state, edge_index, edge_w)[:, action]

            loss = torch.mean(torch.pow(y_target-y_pred, 2))
            loss.backward()
            self.Q.optimizer.step()


    def _max_Q(self, mu, new_state, edge_index, edge_w):
        Q = self.Q(mu, new_state, edge_index, edge_w)
        # Q has shape [batch_size, N]
        return torch.max(Q, dim=1).values.detach()
