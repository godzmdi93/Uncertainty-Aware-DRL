import numpy as np
from EnvRL import EnvRL_v0
import random


import matplotlib.pyplot as plt

env = EnvRL_v0()
env.reset()
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []

for i in range(1, 5000):
    state = env.attack_set[env.reset()]

    epochs, reward, = 0, 0
    done = False
    
    #print(q_table)
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        next_state = env.attack_set[next_state]
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state
        epochs += 1

print('Training Finished')
print(q_table)

