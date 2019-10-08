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
        done = False
        score = 0
        edges_index, state = env.reset()

        state_n_step = [state]
        reward_n_step = []
        action_n_step = []
        n_step_cntr = 0

        while not done:
            action = agent.choose_action(edges_index, state)
            _, state, reward, new_state, done = env.step(action)
            score += reward

            state_n_step.append(state)
            reward_n_step.append(reward)
            action_n_step.append(action)
            n_step_cntr += 1
            
            if n_step_cntr > n_step:
                agent.remember(edges_index, 
                               state = state_n_step[-(n_step+1)], 
                               action = action_n_step[-n_step], 
                               reward_sum = sum(reward_n_step[-n_step:]), 
                               new_state = state_n_step[-1], 
                               done = done)
            agent.learn()
            state = new_state

        scores.append(score)
        avg_score = np.mean(scores[-10:])

        print("Episode: {}, score: {}, avg_score: {}".
                format(i, score, avg_score))







