from env import env
from agent import Agent

import numpy as np

if __name__ == "__main__":
    env = env(graph_size=50) 
    num_eposides = 1000
    n_step = 5
    agent = Agent(num_nodes=env.graph_size)

    scores = []
    
    for i in range(num_eposides):
        score = 0
        edge_index, edge_w, state, done = env.reset()

        state_steps = [state]
        reward_steps = []
        action_steps = []
        steps_cntr = 0

        while not done:
            action = agent.choose_action(edge_index, edge_w, state)
            _, _, state, reward, new_state, done = env.step(action)
            score += reward

            state_steps.append(state)
            reward_steps.append(reward)
            action_steps.append(action)
            steps_cntr += 1
            
            if steps_cntr > n_step:
                agent.remember(edge_index,
                               edge_w,
                               state = state_steps[-(n_step+1)], 
                               action = action_steps[-n_step], 
                               reward_sum = sum(reward_steps[-n_step:]), 
                               new_state = state_steps[-1], 
                               done = done)
            agent.learn()
            state = new_state

        scores.append(score)
        avg_score = np.mean(scores[-10:])

        print("Episode: {}, score: {}, avg_score: {}".
                format(i, score, avg_score))







