from os import stat
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
    epsilon = 0.8

    record = []
    attacks = []
    acts = []
    for i in range(1, 1000):
        state = env.attack_set[env.reset()]

        epochs, reward, = 0.8, 0
        done = False
        
        #print(q_table)
        total_reward = []
        t_a = []
        t_d = []
        total=0
        if i < 500:
            epsilon -=1/randint(1000,3000)
        else:
            epsilon -=0.003

        if epsilon < 0.005:
            epsilon = 0.002

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            t_a.append(env.attack_set[env.current_attack])
            nids_success_rate = random.randrange(0, 100)
            if nids_success_rate < nids:

                next_state, reward, done, info = env.step(action) 
                next_state = env.attack_set[next_state]
                
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])
                
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state, action] = new_value
                t_d.append(action)
            else:
                env.current_attack+=1
                if env.current_attack == len(env.attack_set)-1:
                    done = True
                next_state = env.attack_set[env.current_attack]
                t_d.append(-1)

            state = next_state
            epochs += 1
            total+=reward
            total_reward.append(total)
        acts.append(t_d)
        attacks.append(t_a)
        record.append(total_reward)

    print('Training Finished')
    pdf = []

    s1,s2,s3,s4,s_u = 0,0,0,0,0
    r = []
    for j in range(len(acts)):
        for i in range(len(acts[j])):
            if attacks[j][i] == 0:
                if acts[j][i] == 0:
                    s1+=1
                if acts[j][i] == 1:
                    s2+=1
                if acts[j][i] == 2:
                    s3+=1
                if acts[j][i] == 3:
                    s4+=1
                if acts[j][i] == -1:
                    s_u+=1
                
        r.append([s1,s2,s3,s4,s_u])

    pdf.append(r)

    s1,s2,s3,s4,s_u = 0,0,0,0,0
    r = []
    for j in range(len(acts)):
        for i in range(len(acts[j])):
            if attacks[j][i] == 1:
                if acts[j][i] == 0:
                    s1+=1
                if acts[j][i] == 1:
                    s2+=1
                if acts[j][i] == 2:
                    s3+=1
                if acts[j][i] == 3:
                    s4+=1
                if acts[j][i] == -1:
                    s_u+=1
                
        r.append([s1,s2,s3,s4,s_u])

    pdf.append(r)

    s1,s2,s3,s4,s_u = 0,0,0,0,0
    r = []
    for j in range(len(acts)):
        for i in range(len(acts[j])):
            if attacks[j][i] == 2:
                if acts[j][i] == 0:
                    s1+=1
                if acts[j][i] == 1:
                    s2+=1
                if acts[j][i] == 2:
                    s3+=1
                if acts[j][i] == 3:
                    s4+=1
                if acts[j][i] == -1:
                    s_u+=1
                
        r.append([s1,s2,s3,s4,s_u])

    pdf.append(r)

    s1,s2,s3,s4,s_u = 0,0,0,0,0
    r = []
    for j in range(len(acts)):
        for i in range(len(acts[j])):
            if attacks[j][i] == 3:
                if acts[j][i] == 0:
                    s1+=1
                if acts[j][i] == 1:
                    s2+=1
                if acts[j][i] == 2:
                    s3+=1
                if acts[j][i] == 3:
                    s4+=1
                if acts[j][i] == -1:
                    s_u+=1
                
        r.append([s1,s2,s3,s4,s_u])

    pdf.append(r)

    

    dis = np.zeros((4,4))

    for n in range(4):
        dis[n] = [i/sum(q_table[n]) for i in q_table[n]]

    return dis,record,pdf


