import numpy as np
import torch
from torch_geometric.data import Data, Batch
"""
Store trajectories using Data class
mini-Batch using Batch
"""
class ReplayBuffer:
    def __init__(self, max_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.graph_memory = [None] * self.mem_size

    def store_transition(self, mu, edge_index, edge_w, state, action, reward, state_, done):
        graph = Data(edge_index=torch.tensor(edge_index))
        graph.x_attr = torch.tensor(mu)
        graph.edge_w = torch.tensor(edge_w)
        graph.state = torch.tensor(state)
        graph.action = torch.tensor(action)
        graph.reward = torch.tensor(reward)
        graph.new_state = torch.tensor(state_)
        graph.done = torch.tensor(done)

        index = self.mem_cntr % self.mem_size
        self.graph_memory[index] = graph
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        graph_list = [self.graph_memory[b] for b in batch]
        keys = graph_list[0].keys
        
        return Batch.from_data_list(graph_list)

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)

