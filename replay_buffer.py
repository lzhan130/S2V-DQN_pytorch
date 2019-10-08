import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, num_nodes):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.graph_memory = [None]*self.mem_size #variational size
        self.state_memory = np.zeros((self.mem_size, num_nodes))
        self.new_state_memory = np.zeros((self.mem_size, num_nodes))
        self.action_memory = np.zeros((self.mem_size, 1))
        self.reward_memory = np.zeros((self.mem_size, 1))
        self.terminal_memory = np.zeros((self.mem_size, 1))
 

    def store_transition(self, graph, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.graph_memory[index] = graph
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)
        graph = [self.graph_memory[i] for i in batch]
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return graph, states, actions, rewards, states_, terminal

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)

