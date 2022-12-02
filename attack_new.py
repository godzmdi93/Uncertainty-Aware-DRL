from cmath import inf
from random import randint
import random
import numpy as np
from cnn import train
from keras.datasets import mnist
import sys


def dp(X_train,y_train,sa,sd):
    print('Perform DP attack')
    case = randint(1,3)
    print('case: ' + str(case))
    if case == 1:
        #attackers and defenders have different converage area
        #impact = sa
        impact = random.sample(range(0, 59999), int((sa+(1-sa)*sa)*60000))
        for i in impact:
            X_train[i] = np.random.randint(sys.maxsize, size=(28, 28))
        
    if case == 2:
        #attackers and defenders have same converage area
        #impact = sa - sd
        impact = random.sample(range(0, 59999), int(((sa+(1-sa)*sa)-sd)*60000))
        for i in impact:
            X_train[i] = np.random.randint(sys.maxsize, size=(28, 28))
        
    if case == 3:
        #attackers and defenders have some overlapped converage area
        a_c = random.sample(range(0, 59999), int(sa*60000))
        d_c = random.sample(range(0, 59999), int(sd*60000))
        for i in a_c:
            if i not in d_c:
                X_train[i] = np.random.randint(sys.maxsize, size=(28, 28))
    y_train = [i if (i != 1 and random.random() > (sa-sd)) else 9 for i in y_train]
        

    return X_train,y_train

def ddos(X_train,y_train,sa):

    print('Perform DDoS attack')
    impact = random.sample(range(0, 59999), int((sa+(1-sa)*sa)*60000))
    print('remove: ' + str(len(impact)) + ' instances')
    X_train = np.delete(X_train,impact,0)
    y_train = np.delete(y_train,impact,0)
    
    return X_train,y_train

#sa is coverage rate for attackers, sd is converage rate for defender
def attack_model(sa,sd,df_set):

    print('df_set: ')
    print(df_set)

    #load datasets
    (xtrain,ytrain),(xtest,ytest)=mnist.load_data()

    #random a set of attacks
    #0 = {phish, dp,mp}, 1 = {phish, dp,ddos}, 2 = {phish,mp,ddos}, 3 = {dp, mp, ddos}
    set = randint(0,3)

    #attacks
    #0 - phish, 1 - dp, 2 - mp, 3 - ddos
    #defences
    #0 - ut, 1 - en, 2 - rds, 3 - nf
    if set == 0:
        print('attack_set:',[0,1,2])
        #attack_set = [0,1,2] 
        #dp and mp will only be successfully performed when phishing succeeds 
        if 0 not in df_set:
            if 2 not in df_set:
                if 1 not in df_set:
                    #perform 1 - dp
                    xtrain,ytrain = dp(xtrain,ytrain,sa,0)
                    model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
                    #perform 2 - mp
                    print('Perform MP attack')
                    weights = np.random.rand(5,5,1,32)
                    bias = np.random.rand(32)
                    model.layers[0].set_weights([weights,bias])
                    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
                    return model,xtest,ytest
                else:
                    #perform 1 - dp
                    xtrain,ytrain = dp(xtrain,ytrain,sa,sd)
                    model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
                    #perform 2 - mp
                    print('Perform MP attack')
                    weights = np.random.rand(5,5,1,32)
                    bias = np.random.rand(32)
                    model.layers[0].set_weights([weights,bias])
                    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
                    return model,xtest,ytest
            else:
                if 1 not in df_set:
                    #perform 1 - dp
                    xtrain,ytrain = dp(xtrain,ytrain,sa,0)
                    model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
                    return model,xtest,ytest
                else:
                    #perform 1 - dp
                    xtrain,ytrain = dp(xtrain,ytrain,sa,sd)
                    model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
                    return model,xtest,ytest
        else:
            #if phishing not is successfully defended
            if random.random() < sa:
                if 2 not in df_set:
                    print('Defence failed for 2')
                    if 1 not in df_set:
                        print('Defence failed for 1')
                        #perform 1 - dp
                        xtrain,ytrain = dp(xtrain,ytrain,sa,0)
                        model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
                        #perform 2 - mp
                        print('Perform MP attack')
                        weights = np.random.rand(5,5,1,32)
                        bias = np.random.rand(32)
                        model.layers[0].set_weights([weights,bias])
                        model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
                        return model,xtest,ytest
                    else:
                        #perform 1 - dp
                        xtrain,ytrain = dp(xtrain,ytrain,sa,sd)
                        model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
                        #perform 2 - mp
                        print('Perform MP attack')
                        weights = np.random.rand(5,5,1,32)
                        bias = np.random.rand(32)
                        model.layers[0].set_weights([weights,bias])
                        model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
                        return model,xtest,ytest
                else:
                    if 1 not in df_set:
                        print('Defence failed for 1')
                        #perform 1 - dp
                        xtrain,ytrain = dp(xtrain,ytrain,sa,0)
                        model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
                        return model,xtest,ytest
                    else:
                        #perform 1 - dp
                        xtrain,ytrain = dp(xtrain,ytrain,sa,sd)
                        model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
                        return model,xtest,ytest
            #successfully defend phishing, so dp and mp cannot be performed
            else:
                model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
                return model,xtest,ytest
    elif set == 1:
        print('attack_set:',[0,1,3])
        #attack_set = [0,1,3]
        if 0 not in df_set:
            if 1 not in df_set:
                print('Defence failed for 1')
                #perform 1 - dp
                xtrain,ytrain = dp(xtrain,ytrain,sa,0)
            else:
                xtrain,ytrain = dp(xtrain,ytrain,sa,sd)
        else:
            #if phishing not is successfully defended
            if random.random() < sa:
                print('Defence failed for 0')
                if 1 not in df_set:
                    #perform 1 - dp
                    xtrain,ytrain = dp(xtrain,ytrain,sa,0)
                else:
                    xtrain,ytrain = dp(xtrain,ytrain,sa,sd)
        if 3 not in df_set:
            print('Defence failed for 3')
            #perform 1 - dp
            xtrain,ytrain = ddos(xtrain,ytrain,sa)
        model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
        return model,xtest,ytest
    elif set == 2:
        print('attack_set:',[0,2,3])
        #attack_set = [0,2,3]
        if 3 not in df_set:
            print('Defence failed for 3')
            #perform 1 - dp
            xtrain,ytrain = ddos(xtrain,ytrain,sa)
        model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
        if 0 not in df_set:
            print('Defence failed for 0')
            if 2 not in df_set:
                print('Defence failed for 2')
                print('Perform MP attack')
                #perform 2 - mp
                weights = np.random.rand(5,5,1,32)
                bias = np.random.rand(32)
                model.layers[0].set_weights([weights,bias])
                model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
                return model,xtest,ytest
            else:
                return model,xtest,ytest
        else:
            #if phishing not is successfully defended
            if random.random() < sa:
                print('Defence failed for 0')
                if 2 not in df_set:
                    print('Defence failed for 2')
                    print('Perform MP attack')
                    #perform 2 - mp
                    weights = np.random.rand(5,5,1,32)
                    bias = np.random.rand(32)
                    model.layers[0].set_weights([weights,bias])
                    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
                    return model,xtest,ytest
                else:
                    return model,xtest,ytest
            else:
                return model,xtest,ytest

    elif set == 3:
        print('attack_set:',[1,2,3])
        #attack_set = [1,2,3]
        if 3 not in df_set:
            print('Defence failed for 3')
            #perform 3 - ddos
            xtrain,ytrain = ddos(xtrain,ytrain,sa)
        model,xtest,ytest = train(xtrain,ytrain,xtest,ytest)
        return model,xtest,ytest

def get_acc(sa,sd,action):
    model,xtest,ytest = attack_model(sa,sd,action)
    score = model.evaluate(xtest,ytest)
    print('Test loss:', score[0]) #Test loss: 0.0296396646054
    print('Test accuracy:', score[1]) #Test accuracy: 0.9904
    return score[1]


#test attack_model
df_set = random.sample(range(4), 3)
accuracy = get_acc(0.6,0.4,df_set)

#test original cnn
# (xtrain,ytrain),(xtest,ytest)=mnist.load_data()
# model, x,y = train(xtrain,ytrain,xtest,ytest)
# score = model.evaluate(x,y)
# print('Test loss:', score[0]) #Test loss: 0.0296396646054
# print('Test accuracy:', score[1]) #Test accuracy: 0.9904










