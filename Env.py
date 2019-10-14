import torch as T
import numpy as np
import torch_geometric as TG
from torch_geometric.data import Data
from copy import deepcopy

class env():
    """
    Enviroment for MVC
    """
    def __init__(self, train_size):
        """
        train_size: the number of traning graph
        """
        self.name = "MVC"
        self.train_size = train_size
        self.train_pool = None
        self.graph = None
        self._train_pool()
        self.reset()
    
    def _train_pool(self):
        """
        Generate traning instances for MVC.
        Graph size ranges from 50 to 100.
        ER: edge probability is 0.15
        BA: average degree 4
        """
        self.train_pool = []
        for i in range(self.train_size):
            N = np.random.choice(range(50, 100), size=1)[0]
            if np.random.rand() > 0.5:
                edge_index = TG.utils.barabasi_albert_graph(num_nodes=N, num_edges=4)
            else:
                edge_index = TG.utils.erdos_renyi_graph(num_nodes=N, edge_prob=0.15)
                # so how, for ER,  the edge_index's date type above is not int
                # BUG
                edge_index = edge_index.long()

            graph = Data(num_nodes=N, edge_index=edge_index)
            graph.edge_w = T.ones((N, 1)) 
            graph.mu = T.cat((TG.utils.degree(edge_index[0]).unsqueeze(1),
                                  T.ones((N, 2))), dim=1)
            self.train_pool.append(graph)
            
        print("Train pool generated! Size={}".format(self.train_size))

    def reset(self):
        """
        Randomly choose a graph instance from train pool
        """
        idx = np.random.choice(range(100), size=1)[0]
        self.graph = deepcopy(self.train_pool[idx])
        self.graph.node_tag = T.zeros((self.graph.num_nodes, 1))
        self.graph.cover = self._cover()
        self.graph.done = self._done()

        return self.graph

    def step(self, action):
        """
        take action and update
        """
        cover_pre = deepcopy(self.graph.cover)
        self.graph.node_tag[action] = 1
        self.graph.cover = self._cover()
        self.graph.done = self._done

        reward = self.graph.cover - cover_pre

        return reward, self.graph


    def _done(self):
        num_edges = self.graph.edge_index.shape[1]
        if num_edges == self.graph.cover:
            return 1.0
        else:
            return 0.0

    def _cover(self):
        """
        The number of edges been covered
        """
        edge_index = self.graph.edge_index
        tag = self.graph.node_tag
        cover_edge = [1.0 if tag[n1] == 1 or tag[n2] == 1 else 0.0 for n1, n2 in
                      zip(edge_index[0], edge_index[1])]
        return T.tensor(sum(cover_edge))
