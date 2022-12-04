import random

import numpy as np
import gym
from gym.spaces import Discrete
from Action import Actions
from attack1 import dp, ddos, attack3, attack4


class EnvRL_v0(gym.Env):

    def __init__(self,attack_set = []):
        super().__init__()

        self.current_accuracy = 0
        self.current_attack = 0  # 0 means no attack
        self.attack_set = attack_set
        self.action_space = Discrete(4)
        self.observation_space = Discrete(4) # [attack type, accuracy]

    def take_action(self, action):
        
        # if action == self.attack_set[self.current_attack]:
        #     self.current_accuracy = 1
        # else:
        #     self.current_accuracy = -1


        if self.attack_set[self.current_attack] == 0:
            self.current_accuracy = dp(0.8,0.2,action)
        elif self.attack_set[self.current_attack] == 1:
            self.current_accuracy = ddos(0.8,0.2,action)
        elif self.attack_set[self.current_attack] == 2:
            self.current_accuracy = attack3(0.8,0.2,action)
        elif self.attack_set[self.current_attack] == 3:
            self.current_accuracy = attack4(0.8,0.2,action)

        
    def step(self, action):

        self.take_action(action)
        done = False
        if self.current_attack == len(self.attack_set)-2:
            done = True
        reward = self.current_accuracy
        info = {}
        self.current_attack+=1

        return self.current_attack, reward, done, info

    def reset(self):
        self.current_accuracy = 0
        self.current_attack = 0
        return self.current_attack

    def render(self, mode='human', action=0, reward=0):
        if mode == 'human':
            print(f"{Actions(action): <4}: ({self.current_attack}, {self.current_accuracy}) reward = {reward}")
        else:
            super().render(mode=mode)  # just raise an exception
