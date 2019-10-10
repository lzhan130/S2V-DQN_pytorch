import numpy as np
from torch_geometric.data import Data, Batch
"""
Store trajectories using Data class
mini-Batch using Batch
"""

class ReplayBuffer:
    def __init__(self, max_size, num_nodes):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.graph_memory = [None] * self.mem_size

    def store_transition(self, edge_index, edge_w, state, action, reward, state_, done):
        graph = Data(edge_index=edge_index)
        graph.edge_w = edge_w
        graph.state = state
        graph.action = action
        graph.reward = reward
        graph.new_state = state_
        graph.done = done

        index = self.mem_cntr % self.mem_size
        self.graph_memory[index] = graph
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        graph_list = [self.graph_memory[b] for b in batch]
        
        return Batch(graph_list)

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)

