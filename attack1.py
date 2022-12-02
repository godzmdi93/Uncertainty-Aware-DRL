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
            cnn_accuracy = 0.9
        else:
            cnn_accuracy = 0.8
            # random.uniform(a,b)

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


# for attack C&W and code processing attack, we don't consider converage area
# just use probability of a model being attacked
def cw_cpa(ap, a_type, d_type):
    json_file = open('original_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("original_weight.h5")
    print("Loaded model from disk")

    if random.random() < ap:
        print('Attack performed')
        if a_type == d_type:
            print('Successfully defended')
        else:
            weights = np.random.rand(3, 3, 1, 32)
            bias = np.random.rand(32)
            loaded_model.layers[0].set_weights([weights, bias])
            weights = np.random.rand(3, 3, 32, 64)
            bias = np.random.rand(64)
            loaded_model.layers[1].set_weights([weights, bias])

    loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                         optimizer=keras.optimizers.Adadelta(),
                         metrics=['accuracy'])

    img_rows, img_cols = 28, 28
    if k.image_data_format() == 'channels_first':
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    else:
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    X_test = X_test.astype('float32')
    X_test /= 255
    num_category = 10
    y_test = keras.utils.to_categorical(y_test, num_category)
    model = cw_cpa(0.5, 'cw', 'cw')
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score[1]


# Test data poisoning attack
# dp(X_train, 0.8, 0.2, 'ddos')  # last parameter is the defense mechanism
# train(X_train, y_train, X_test, y_test)

# Test DDoS attack
# ddos(y_train, 0.8, 0.2, 'dp')
# train(X_train, y_train, X_test, y_test)

'''
# Test C&W and code processing attack
img_rows, img_cols = 28, 28
if k.image_data_format() == 'channels_first':
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
else:
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_test = X_test.astype('float32')
X_test /= 255
num_category = 10
y_test = keras.utils.to_categorical(y_test, num_category)
model = cw_cpa(0.5, 'cw', 'cw')
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
