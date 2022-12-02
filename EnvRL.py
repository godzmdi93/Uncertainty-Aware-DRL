import random

import numpy as np
import gym
from gym.spaces import Discrete, Box
from Action import Actions
from attack1 import dp, ddos, attack3, attack4
import pickle
X_train = pickle.load(open("xtrain", "rb"))
y_train = pickle.load(open("ytrain", "rb"))
X_test = pickle.load(open("xtest", "rb"))
y_test = pickle.load(open("ytest", "rb"))

class EnvRL_v0(gym.Env):

    def __init__(self):
        super().__init__()

        self.current_accuracy = 1
        self.current_attack = 0  # 0 means no attack
        self.attack_type = 1
        self.action_space = Discrete(4)
        self.observation_space = Box(low=np.array([0., 0.]), high=np.array([4., 1.])) # [attack type, accuracy]

    def take_action(self, action):
        # nids_success_rate = 2
        # attack happen
        # attack_type = random.randint(1, 5)
        # attack_type = 3
        self.attack_type += 1
        if self.attack_type > 4:
            self.attack_type = 1
        self.current_attack = self.attack_type

        if self.attack_type == 1:
            self.current_accuracy = dp(X_train, 0.8, 0.2, action)
        elif self.attack_type == 2:
            self.current_accuracy = ddos(X_train, 0.8, 0.2, action)
        elif self.attack_type == 3:
            self.current_accuracy = attack3(X_train, 0.8, 0.2, action)
        elif self.attack_type == 4:
            self.current_accuracy = attack4(X_train, 0.8, 0.2, action)



    def step(self, action):

        self.take_action(action)
        state = np.array([self.current_attack, self.current_accuracy])
        done = 1
        reward = self.current_accuracy
        info = {}

        return state, reward, done, info

    def reset(self):
        self.current_accuracy = 0.65
        self.current_attack = 0
        return np.array([self.current_attack, self.current_accuracy])

    def render(self, mode='human', action=0, reward=0):
        if mode == 'human':
            print(f"{Actions(action): <4}: ({self.current_attack}, {self.current_accuracy}) reward = {reward}")
        else:
            super().render(mode=mode)  # just raise an exception
