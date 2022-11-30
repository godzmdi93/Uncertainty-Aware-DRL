import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete
from Action import Actions


class EnvRL_v0(gym.Env):

    def __init__(self):
        super().__init__()

        self.current_accuracy = 0
        self.previous_accuracy = 0
        self.action_space = Discrete(4)
        self.observation_space = MultiDiscrete([self.previous_accuracy, self.current_accuracy])

    def take_action(self, action):
        # attack happen

        if action == Actions.Defence1:
            pass
        elif action == Actions.Defence2:
            pass
        elif action == Actions.Defence3:
            pass
        elif action == Actions.Defence4:
            pass

    def step(self, action):

        self.take_action(action)
        state = np.array([self.previous_accuracy, self.current_accuracy])
        done = 1
        reward = self.current_accuracy - self.previous_accuracy
        info = {}

        return state, reward, done, info

    def reset(self):
        self.current_accuracy = 0
        self.previous_accuracy = 0
        return np.array([self.previous_accuracy, self.current_accuracy])

    def render(self, mode='human', action=0, reward=0):
        if mode == 'human':
            print(f"{Actions(action): <4}: ({self.previous_accuracy},{self.current_accuracy}) reward = {reward}")
        else:
            super().render(mode=mode)  # just raise an exception
