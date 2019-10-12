from env import env
from agent import Agent

import numpy as np

if __name__ == "__main__":
    env = env(graph_size=50) 
    num_eposides = 100000
    n_step = 5
    agent = Agent(num_nodes=env.graph_size)

    scores = []
    
    for i in range(num_eposides):
        score = 0
        mu, edge_index, edge_w, state, done = env.reset()

        state_steps = [state]
        reward_steps = []
        action_steps = []
        steps_cntr = 0

        while not done[0]:
            action = agent.choose_action(mu, edge_index, edge_w, state)
            _, _, _, reward, new_state, done = env.step(action)
            score += reward

            state_steps.append(new_state)
            reward_steps.append(reward)
            action_steps.append(action)
            steps_cntr += 1
            
            if steps_cntr > n_step+1:
                agent.remember(mu,
                               edge_index,
                               edge_w,
                               state_steps[-(n_step+1)], 
                               action_steps[-n_step], 
                               [sum(reward_steps[-n_step:])], 
                               state_steps[-1], 
                               done)
                agent.learn()
            state = new_state

        scores.append(score)
        avg_score = np.mean(scores[-10:])

        print("Episode: {:<4}, score: {:<4}, avg_score: {:.2f}".
                format(i, score, avg_score))
        if i%10 == 0:
            print("... Saving scores ...")
            np.save("scores_log.npy", scores)







