from env import env
from agent import Agent

import numpy as np

def Greedy(edge_index):
    r"""Greedy for MVC"""
    import networkx as nx
    G = nx.from_edgelist(np.array(edge_index).T)
    cover_set = []
    while G.number_of_edges() > 0:
        d = dict(G.degree())
        sort_d = sorted(list(zip(d.keys(), d.values())), key=lambda x:x[1], reverse=True)
        del_node = sort_d[0][0]
        cover_set.append(del_node)
        G.remove_node(del_node)
    print("***** Greedy:{} *****".format(len(cover_set)))

if __name__ == "__main__":
    env = env(graph_size=50) 
    num_eposides = 100000
    n_step = 5
    agent = Agent(num_nodes=env.graph_size)

    scores = []
    
    for i in range(num_eposides):
        score = 0
        num_nodes, mu, edge_index, edge_w, state, done = env.reset()

        state_steps = [state]
        reward_steps = []
        action_steps = []
        steps_cntr = 0

        while not done[0]:
            action = agent.choose_action(mu, edge_index, edge_w, state)
            _, _, _, reward, new_state, done = env.step(action)

            state_steps.append(new_state)
            reward_steps.append(reward)
            action_steps.append(action)
            steps_cntr += 1
            
            if steps_cntr > n_step+1:
                agent.remember(num_nodes,
                               mu,
                               edge_index,
                               edge_w,
                               state_steps[-(n_step+1)], 
                               action_steps[-n_step], 
                               [sum(reward_steps[-n_step:])], 
                               state_steps[-1], 
                               done)
                agent.learn()

            state = new_state
        score = len(set([a[0] for a in action_steps]))
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        print("Episode: {:<4}, score: {:<4}, avg_score: {:.2f}".
                format(i, score, avg_score))
        if i%10 == 0:
            # Greedy
            Greedy(edge_index)
            print("Saving scores ...")
            np.save("scores_log.npy", scores)
            agent.Q.scheduler.step() 

        if i%100 == 0:
            print("Saving model ...")
            agent.save("check_point/model_state_dict_{}".format(i))
            

