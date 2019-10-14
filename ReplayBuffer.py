import numpy as np
import torch as T

from torch_geometric.data import Batch

class replayBuffer():
    def __init__(self, mem_size=100000):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.graphs_pre= [None] * mem_size
        self.graphs_later = [None] * mem_size
        self.actions = [None] * mem_size
        self.rewards = [None] * mem_size

    def store(self, graph_pre, action, reward, graph_later):
        idx = self.mem_cntr % self.mem_size
        sef.graphs_pre[idx] = graph_pre
        sef.graphs_later[idx] = graph_later
        sef.actions[idx] = action
        sef.rewards[idx] = reward

        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        graphs_pre_batch   = Batch.from_data_list([self.graphs_pre[b] for b in batch])
        graphs_later_batch = Batch.from_data_list([self.graphs_later[b] for b in batch])
        actions_batch      = T.tensor([self.actions[b] for b in batch])
        rewards_batch      = T.tensor([self.rewards[b] for b in batch])

        return graphs_pre_batch, graphs_later_batch, actions_batch, rewards_batch

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)
