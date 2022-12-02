from random import randint
from keras import backend as k
import pickle
import random
import numpy as np

from Action import Actions
from cnn import train
import json
import keras
from keras.models import Sequential, model_from_json

X_train = pickle.load(open("xtrain", "rb"))
y_train = pickle.load(open("ytrain", "rb"))
X_test = pickle.load(open("xtest", "rb"))
y_test = pickle.load(open("ytest", "rb"))

print('X_train shape:', X_train.shape)


def dp(X_train, sa, sd, type):  # {dp, ddos}   dp()
    if type == Actions.Defence1:
        if sa == 0.8 and sd == 0.2:
            cnn_accuracy = 0.8
    else:
        cnn_accuracy = 0.65

    return cnn_accuracy


def ddos(X_train, sa, sd, type):
    if type == Actions.Defence2:
        if sa==0.8 and sd==0.2:
            cnn_accuracy = 0.8
    else:
        cnn_accuracy = 0.65

    return cnn_accuracy

def attack3(X_train, sa, sd, type):
    if type == Actions.Defence3:
        if sa==0.8 and sd==0.2:
            cnn_accuracy = 0.8

    else:
        cnn_accuracy = 0.65

    return cnn_accuracy

def attack4(X_train, sa, sd, type):
    if type == Actions.Defence4:
        if sa==0.8 and sd==0.2:
            cnn_accuracy = 0.8

    else:
        cnn_accuracy = 0.65

    return cnn_accuracy