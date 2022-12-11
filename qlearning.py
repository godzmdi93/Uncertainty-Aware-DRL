from curses import echo
from os import stat
import numpy as np
from random import randint

from pyparsing import actions
from EnvRL import EnvRL_v0
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from u_helper import *


def train(step,nids,type):

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
    for i in range(1, 2000): #2000 / 200
        state = env.attack_set[env.reset()]

        epochs, reward, = 0.8, 0
        done = False
        
        #print(q_table)
        total_reward = []
        t_a = []
        t_d = []
        total=0
        if type == 0: #vacuity
            epsilon -=0.0005 #0.0005

            if epsilon < 0.005: #0.005
                epsilon = 0.005 #0.005

        elif type == 1:  #dissonance
            epsilon -=0.00025 #0.0005

            if epsilon < 0.35: #0.005
                epsilon = 0.35 #0.005
        elif type == 2:  #dissonance
            epsilon -=0.0003 #0.0005

            if epsilon < 0.3: #0.005
                epsilon = 0.3 #0.005
        elif type == 3:  #epsilon-greedy
            epsilon -=0.00035 #0.0005

            if epsilon < 0.1: #0.005
                epsilon = 0.1 #0.005



        # if len(attacks)!=0 and len(acts) !=0:
        #     a = attacks[0]
        #     d = acts[0]
        #     B1 = draw_u(0,a,d) #belief for attack_set1
        #     B2 = draw_u(1,a,d)
        #     B3 = draw_u(2,a,d)
        #     B4 = draw_u(3,a,d)
        #     H1,H2,H3,H4 = diss(B1,B2,B3,B4)
        # else:
        #     H1,H2,H3,H4 = [1],[1],[1],[1]

        # epsilon = epsilon*(1-(H1[-1]+H2[-1]+H3[-1]+H4[-1]))

        # if len(acts)!=0 and len(attacks)!=0:
        #     d = acts[0]
        #     a = attacks[0]
        #     B1 = draw_u(0,a,d) #belief for attack_set1
        #     B2 = draw_u(1,a,d)
        #     B3 = draw_u(2,a,d)
        #     B4 = draw_u(3,a,d)
        #     u = [translateU(i[0:4]) for i in B1][-1] #0.03
        #     d1,d2,d3,d4 = diss(B1,B2,B3,B4)
        #     d = np.average([d1,d2,d3,d4],axis=0)[-1] #0.1
        #     epsilon = 0.2
        # else:
        #     epsilon = 0.8

        
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
        #total_reward.append(total)
        acts.append(t_d)
        attacks.append(t_a)
        record.append(total/epochs)

    print('Training Finished')

    pdf = cal(acts,attacks)

    dis = np.zeros((4,4))

    for n in range(4):
        dis[n] = [i/sum(q_table[n]) for i in q_table[n]]

    return dis,record,pdf,acts,attacks


def cal(acts,attacks):
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

    return pdf
