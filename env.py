"""
Created : 2019/10/04
Author  : Philip Gao

Beijing Normal University
"""

import numpy as np
import networkx as nx

# build env for MVC

class env():
    """
    The following methods are needed:
    - reset: regenerate a instance(graph) and initialize records
    - step : take action and return state_next, reward, done 
    """
    def __init__(self, graph_size=50):
        self.graph_size = graph_size
        self.name = "MVC"
        self.reset()

    def reset(self):
        self.edges_index = self._BA(self.graph_size) 
        self.nodes_tag = np.zeros(self.graph_size,  dtype=np.float32)
        self.cost = 0
        self.done = False

        return self.edges_index, self.nodes_tag

    def step(self, action):
        assert action in range(self.graph_size)
        state = self.nodes_tag
        self.nodes_tag[action] = 1
        new_cost = self._MVC_cost()
        reward = new_cost - self.cost
        self.cost = new_cost
        self.done = self._done()
        state_ = self.nodes_tag

        return self.edges_index, state, reward, state_, self.done

    def _BA(self, size):
        G = nx.random_graphs.barabasi_albert_graph(n=size, m=3)
        edges_index = np.array(G.edges()).T
        return edges_index

    def _MVC_cost(self):
        return sum(self.nodes_tag)

    def _done(self):
        # check if all nodes are covered
        half_remain_edges = [False if self.nodes_tag[node] == 1 else True 
                             for node in self.edges_index[0]]
        remain_nodes = self.edges_index[1][half_remain_edges] 

        if self.nodes_tag[remain_nodes].sum() == len(remain_nodes):
            done_flag = True 
        else:
            done_flag = False
        return done_flag
        

        
