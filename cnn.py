from keras.datasets import mnist
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train(xtrain,ytrain,xtest,ytest):
    #loading data
    #reshaping data as needed by the model
    xtrain=np.reshape(xtrain,(-1,28,28,1))
    xtest=np.reshape(xtest,(-1,28,28,1))

    #normalising
    xtrain=xtrain/255
    xtest=xtest/255

    #implementing one hot encoding
    from keras.utils.np_utils import to_categorical
    y_train = to_categorical(ytrain, num_classes=10)
    y_test = to_categorical(ytest, num_classes=10)

    #importing the model
    from keras.models import Sequential

    #creating model object
    model=Sequential()

    #importing layers
    from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

    #adding layers and forming the model
    model.add(Conv2D(32,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(28,28,1)))
    model.add(MaxPooling2D(padding="same"))

    model.add(Conv2D(64,kernel_size=5,strides=1,padding="same",activation="relu"))
    model.add(MaxPooling2D(padding="same"))

    model.add(Flatten())

    model.add(Dense(1024,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation="sigmoid"))

    #compiling
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

    #training the model
    model.fit(xtrain,y_train,batch_size=500,epochs=1)

    return model,xtest,y_test

# (xtrain,ytrain),(xtest,ytest)=mnist.load_data()
# xtrain = xtrain[:1000]
# ytrain = ytrain[:1000]
# #ytrain = [i if i != 1 else 9 for i in ytrain]
# #ytrain = [i if i != 7 else 9 for i in ytrain]
# model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
# score = model.evaluate(xtest,ytest)
# print('Test loss:', score[0]) #Test loss: 0.0296396646054
# print('Test accuracy:', score[1]) #Test accuracy: 0.9904

