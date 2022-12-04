import numpy as np
from random import randint
from EnvRL import EnvRL_v0
import random
import matplotlib.pyplot as plt
from scipy.stats import norm


def train(step,nids):

    attack_set = [randint(0,3) for i in range(step)]
    env = EnvRL_v0(attack_set)
    env.reset()
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    record = []
    for i in range(1, 1000):
        state = env.attack_set[env.reset()]

        epochs, reward, = 0.8, 0
        done = False
        
        #print(q_table)
        total_reward = []
        total=0
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            nids_success_rate = random.randrange(0, 100)
            if nids_success_rate < nids:

                next_state, reward, done, info = env.step(action) 
                next_state = env.attack_set[next_state]
                
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])
                
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state, action] = new_value
            else:
                env.current_attack+=1
                if env.current_attack == len(env.attack_set)-1:
                    done = True
                next_state = env.attack_set[env.current_attack]

            state = next_state
            epochs += 1
            total+=reward
            total_reward.append(total)
        record.append(total_reward)

    print('Training Finished')
    
    dis = np.zeros((4,4))

    for n in range(4):
        dis[n] = [i/sum(q_table[n]) for i in q_table[n]]

    return dis,record


