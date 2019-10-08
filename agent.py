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

    def choose_action(self, edges_index, state):
        x_attr = torch.zeros(self.num_nodes, self.in_dim).to(self.Q.device)
        edges_index = torch.tensor(edges_index).to(self.Q.device)
        edges_attr = torch.ones(edges_index.shape[1], 1).to(self.Q.device)
        state = torch.tensor(state).to(self.Q.device)

        Q = self.Q(x_attr, edges_index, edges_attr, state)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_nodes, size=1)
        else:
            action = torch.argmax(Q).item()

        return action
    
    def remember(self, edges_list, state, action, reward_sum, new_state, done):
        self.memory.store_transition(edges_list, state, action, reward_sum, 
                                     new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        edges_list, state, action, reward_sum, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        self.Q.optimizer.zero_grad()
        device = self.Q.device
        edges_list = torch.tensor(edges_list).to(device)
        state = torch.tensor(state).to(device)
        action = torch.tensor(action).to(device)
        reward_sum = torch.tensor(reward_sum).to(device)
        new_state = torch.tensor(new_state).to(device)
        done = torch.tensor(done).to(device)

        y_target = reward_sum + self.gamma * self._max_Q(edges_list, new_state)
        y_pred   = self.Q(edges_list, new_state)[action]

        loss = torch.pow(y_target - y_pred, 2)
        loss.backward()

        self.Q.optimizer.step()


    def _max_Q(self, edges_index, state):
        x_attr = torch.zeros(self.num_nodes, self.in_dim).to(self.Q.device)
        edges_index = torch.tensor(edges_index).to(self.Q.device)
        edges_attr = torch.ones(edges_index.shape[1], 1).to(self.Q.device)
        state = torch.tensor(state).to(self.Q.device)

        Q = self.Q(x_attr, edges_index, edges_attr, state)
        return torch.max(Q).detach() 
