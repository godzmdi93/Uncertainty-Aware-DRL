import random

import numpy as np
from cnn import trans_data
import gym
from gym.spaces import Discrete, Box
from Action import Actions
from attack1 import dp, ddos, attack3, attack4
import pickle
from keras.models import model_from_json

X_train = pickle.load(open("xtrain", "rb"))
y_train = pickle.load(open("ytrain", "rb"))
X_test = pickle.load(open("xtest", "rb"))
y_test = pickle.load(open("ytest", "rb"))

def get_acc(attack,sa,sd,action):
    
    if attack == action:
        return 0.8
    else:
        if attack == 2:
            xtest,ytest = trans_data()
            json_file = open('original_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("original_weight.h5")
            print("Loaded model from disk")

            weights = np.random.rand(5,5,1,32)
            bias = np.random.rand(32)
            model.layers[0].set_weights([weights,bias])
            model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

            score = model.evaluate(xtest,ytest)
            return score[0]
        else:
            return 0.2


class EnvRL_v0(gym.Env):

    def __init__(self):
        super().__init__()

        self.current_accuracy = 0
        self.current_attack = 0  # 0 means no attack
        self.attack_type = 1
        self.attack_set = [0,1,2,3,2,0,1,2,0,3,3,2,1,1,0,3,2,1,0,2]
        self.action_space = Discrete(4)
        self.observation_space = Discrete(4) # [attack type, accuracy]

    def take_action(self, action):
        
        # if action == self.attack_set[self.current_attack]:
        #     self.current_accuracy = 1
        # else:
        #     self.current_accuracy = -1


        if self.attack_set[self.current_attack] == 0:
            self.current_accuracy = dp(X_train,0.8,0.2,action)
        elif self.attack_set[self.current_attack] == 1:
            self.current_accuracy = ddos(X_train,0.8,0.2,action)
        elif self.attack_set[self.current_attack] == 2:
            self.current_accuracy = attack3(X_train,0.8,0.2,action)
        elif self.attack_set[self.current_attack] == 3:
            self.current_accuracy = attack4(X_train,0.8,0.2,action)


        #self.current_accuracy = get_acc(self.attack_set[self.current_attack],0.6,0.4,action)
        
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
