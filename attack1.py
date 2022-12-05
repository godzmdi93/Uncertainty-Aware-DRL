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


def dp(sa, sd, type):  # {dp, ddos}   dp()
    if type == Actions.Defence1:
        if random.random() > (sa - sd):
        #if sa == 0.8 and sd == 0.2:
            cnn_accuracy = random.uniform(0.7, 1.0)
        else:
            cnn_accuracy = random.uniform(0.5, 0.7)
    else:
        cnn_accuracy = 0

    return cnn_accuracy


def ddos(sa, sd, type):
    if type == Actions.Defence2:
        if random.random() > (sa - sd):
        #if sa == 0.8 and sd == 0.2:
            cnn_accuracy = random.uniform(0.6, 0.8)
        else:
            cnn_accuracy = random.uniform(0.5, 0.7)
    else:
        cnn_accuracy = 0

    return cnn_accuracy

def attack3(sa, sd, type):
    if type == Actions.Defence3:
        if random.random() >sa:
        #if sa == 0.8 and sd == 0.2:
            cnn_accuracy = random.uniform(0.6, 1.0)
        else:
            cnn_accuracy = random.uniform(0.5, 0.7)
    else:
        cnn_accuracy = 0
    return cnn_accuracy

def attack4(sa, sd, type):
    if type == Actions.Defence4:
        if random.random() > (sa - sd):
        #if sa == 0.8 and sd == 0.2:
            cnn_accuracy = random.uniform(0.3, 0.9)
        else:
            cnn_accuracy = random.uniform(0.3, 0.7)
    else:
        cnn_accuracy = 0

    return cnn_accuracy
