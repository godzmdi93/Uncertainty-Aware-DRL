from bijective_mapping import *

from uncertainty_estimates import *

import matplotlib.pyplot as plt

from qlearning import *

import numpy as np


#_,_,pdf,acts,attacks = train(5000,80,1) #500,80


#calculate B U for observation 1 (which is attack set 1)
B1,B2,B3,B4 = [],[],[],[] #belief mass for each action

U1 = []

def draw_u(action,a,d):

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    t = 0
    f = []
    for i in range(len(a)):
        if a[i] == action:
            t+=1
            if d[i] == 0:
                count1+=1
            if d[i] == 1:
                count2+=1
            if d[i] == 2:
                count3+=1
            if d[i] == 3:
                count4+=1
            if d[i] == -1:
                count5+=1
            f.append([count1,count2,count3,count4,count5])
    return f

#B, U
def b_u(B1,B2,B3,B4):
    x = [i for i in range(len(B1))]
    plt.plot(x,[translateB(i[0],i[0:4]) for i in B1],label = 'Action1')
    plt.plot(x,[translateB(i[1],i[0:4]) for i in B1],label = 'Action2')
    plt.plot(x,[translateB(i[2],i[0:4]) for i in B1],label = 'Action3')
    plt.plot(x,[translateB(i[3],i[0:4])for i in B1],label = 'Action4')
    plt.plot(x,[translateU(i[0:4]) for i in B1],label = 'Uncertainty')
    plt.xlabel('Number of steps')
    plt.ylabel('Belief mass b_X1(x)')
    plt.legend()
    plt.show()

#Dissonance
def diss(B1,B2,B3,B4):
    b1 = [translateB(i[0],i[0:4]) for i in B1]
    b2 = [translateB(i[1],i[0:4]) for i in B1]
    b3 = [translateB(i[2],i[0:4]) for i in B1]
    b4 = [translateB(i[3],i[0:4]) for i in B1]


    D1,D2,D3,D4 = [],[],[],[]
    for i in range(len(b1)):
        D1.append(Diss([b1[i],b2[i],b3[i],b4[i]]))
    b1 = [translateB(i[0],i[0:4]) for i in B2]
    b2 = [translateB(i[1],i[0:4]) for i in B2]
    b3 = [translateB(i[2],i[0:4]) for i in B2]
    b4 = [translateB(i[3],i[0:4]) for i in B2]
    for i in range(len(b1)):
        D2.append(Diss([b1[i],b2[i],b3[i],b4[i]]))

    b1 = [translateB(i[0],i[0:4]) for i in B3]
    b2 = [translateB(i[1],i[0:4]) for i in B3]
    b3 = [translateB(i[2],i[0:4]) for i in B3]
    b4 = [translateB(i[3],i[0:4]) for i in B3]
    for i in range(len(b1)):
        D3.append(Diss([b1[i],b2[i],b3[i],b4[i]]))

    b1 = [translateB(i[0],i[0:4]) for i in B4]
    b2 = [translateB(i[1],i[0:4]) for i in B4]
    b3 = [translateB(i[2],i[0:4]) for i in B4]
    b4 = [translateB(i[3],i[0:4]) for i in B4]
    for i in range(len(b1)):
        D4.append(Diss([b1[i],b2[i],b3[i],b4[i]]))

    # plt.xlabel('Number of steps')
    # plt.ylabel('Dissonance')
    # plt.plot([i for i in range(len(D1))],D1,label = 'attack_set1')
    # plt.plot([i for i in range(len(D2))],D2,label = 'attack_set2')
    # plt.plot([i for i in range(len(D3))],D3,label = 'attack_set3')
    # plt.plot([i for i in range(len(D4))],D4,label = 'attack_set4')
    # plt.legend()
    # plt.show()
    return D1


def entropy(B1,B2,B3,B4):
    #attack_set1
    b1 = [translateB(i[0],i[0:4]) for i in B1]
    b2 = [translateB(i[1],i[0:4]) for i in B1]
    b3 = [translateB(i[2],i[0:4]) for i in B1]
    b4 = [translateB(i[3],i[0:4]) for i in B1]
    u = [translateU(i[0:4]) for i in B1]

    p1,p2,p3,p4 =[],[],[],[]
    for i in range(len(b1)):
        p1.append(b1[i]+0.25*u[i])
    for i in range(len(b2)):
        p2.append(b2[i]+0.25*u[i])
    for i in range(len(b3)):
        p3.append(b3[i]+0.25*u[i])
    for i in range(len(b4)):
        p4.append(b4[i]+0.25*u[i])

    H1,H2,H3,H4 = [],[],[],[]
    for i in range(len(p1)):
        H1.append(p1[i]*math.log(p1[i],4)+p2[i]*math.log(p2[i],4)+p3[i]*math.log(p3[i],4)+p4[i]*math.log(p4[i],4))


     #attack_set2
    b1 = [translateB(i[0],i[0:4]) for i in B2]
    b2 = [translateB(i[1],i[0:4]) for i in B2]
    b3 = [translateB(i[2],i[0:4]) for i in B2]
    b4 = [translateB(i[3],i[0:4]) for i in B2]
    u = [translateU(i[0:4]) for i in B2]

    p1,p2,p3,p4 =[],[],[],[]
    for i in range(len(b1)):
        p1.append(b1[i]+0.25*u[i])
    for i in range(len(b2)):
        p2.append(b2[i]+0.25*u[i])
    for i in range(len(b3)):
        p3.append(b3[i]+0.25*u[i])
    for i in range(len(b4)):
        p4.append(b4[i]+0.25*u[i])

    for i in range(len(p1)):
        H2.append(p1[i]*math.log(p1[i],4)+p2[i]*math.log(p2[i],4)+p3[i]*math.log(p3[i],4)+p4[i]*math.log(p4[i],4))

      #attack_set3
    b1 = [translateB(i[0],i[0:4]) for i in B3]
    b2 = [translateB(i[1],i[0:4]) for i in B3]
    b3 = [translateB(i[2],i[0:4]) for i in B3]
    b4 = [translateB(i[3],i[0:4]) for i in B3]
    u = [translateU(i[0:4]) for i in B3]

    p1,p2,p3,p4 =[],[],[],[]
    for i in range(len(b1)):
        p1.append(b1[i]+0.25*u[i])
    for i in range(len(b2)):
        p2.append(b2[i]+0.25*u[i])
    for i in range(len(b3)):
        p3.append(b3[i]+0.25*u[i])
    for i in range(len(b4)):
        p4.append(b4[i]+0.25*u[i])

    for i in range(len(p1)):
        H3.append(p1[i]*math.log(p1[i],4)+p2[i]*math.log(p2[i],4)+p3[i]*math.log(p3[i],4)+p4[i]*math.log(p4[i],4))

      #attack_set4
    b1 = [translateB(i[0],i[0:4]) for i in B4]
    b2 = [translateB(i[1],i[0:4]) for i in B4]
    b3 = [translateB(i[2],i[0:4]) for i in B4]
    b4 = [translateB(i[3],i[0:4]) for i in B4]
    u = [translateU(i[0:4]) for i in B4]

    p1,p2,p3,p4 =[],[],[],[]
    for i in range(len(b1)):
        p1.append(b1[i]+0.25*u[i])
    for i in range(len(b2)):
        p2.append(b2[i]+0.25*u[i])
    for i in range(len(b3)):
        p3.append(b3[i]+0.25*u[i])
    for i in range(len(b4)):
        p4.append(b4[i]+0.25*u[i])

    for i in range(len(p1)):
        H4.append(p1[i]*math.log(p1[i],4)+p2[i]*math.log(p2[i],4)+p3[i]*math.log(p3[i],4)+p4[i]*math.log(p4[i],4))

    y = np.average([H1,H2,H3,H4],axis = 0)
    # plt.plot([i for i in range(len(H1))],H1,label = 'attack_set1')
    # plt.plot([i for i in range(len(H2))],H2,label = 'attack_set2')
    # plt.plot([i for i in range(len(H3))],H3,label = 'attack_set3')
    # plt.plot([i for i in range(len(H4))],H4,label = 'attack_set4')
    # plt.xlabel('Number of steps')
    # plt.ylabel('Shannon Entropy H(X)')
    # plt.legend()
    # plt.show()
    return H1

def draw_e():
    #b_u(B1,B2,B3,B4)
    #diss(B1,B2,B3,B4)
    _,_,pdf,acts,attacks = train(5000,80,0)


    d = acts[0]
    a = attacks[0]

    B1 = draw_u(0,a,d) #belief for attack_set1
    B2 = draw_u(1,a,d)
    B3 = draw_u(2,a,d)
    B4 = draw_u(3,a,d)


    h1 = entropy(B1,B2,B3,B4)

    _,_,pdf,acts,attacks = train(5000,80,1)


    d = acts[0]
    a = attacks[0]

    B1 = draw_u(0,a,d) #belief for attack_set1
    B2 = draw_u(1,a,d)
    B3 = draw_u(2,a,d)
    B4 = draw_u(3,a,d)


    h2 = entropy(B1,B2,B3,B4)

    _,_,pdf,acts,attacks = train(5000,80,2)


    d = acts[0]
    a = attacks[0]

    B1 = draw_u(0,a,d) #belief for attack_set1
    B2 = draw_u(1,a,d)
    B3 = draw_u(2,a,d)
    B4 = draw_u(3,a,d)


    h3 = entropy(B1,B2,B3,B4)

    _,_,pdf,acts,attacks = train(5000,80,3)


    d = acts[0]
    a = attacks[0]

    B1 = draw_u(0,a,d) #belief for attack_set1
    B2 = draw_u(1,a,d)
    B3 = draw_u(2,a,d)
    B4 = draw_u(3,a,d)


    h4 = entropy(B1,B2,B3,B4)

    plt.plot([i for i in range(len(h1))],h1,label = 'Epsilon-Vacuity')
    plt.plot([i for i in range(len(h2))],h2,label = 'Epsilon-Dissonance')
    plt.plot([i for i in range(len(h3))],h3,label = 'Epsilon-Entropy')
    plt.plot([i for i in range(len(h4))],h4,label = 'Epsilon-Greedy')
    plt.xlabel('Number of episodes')
    plt.ylabel('Shannon Entropy H(X)')
    plt.legend()
    plt.show()

def draw_d():
    #b_u(B1,B2,B3,B4)
    #diss(B1,B2,B3,B4)
    _,_,pdf,acts,attacks = train(5000,80,0)


    d = acts[0]
    a = attacks[0]

    B1 = draw_u(0,a,d) #belief for attack_set1
    B2 = draw_u(1,a,d)
    B3 = draw_u(2,a,d)
    B4 = draw_u(3,a,d)


    h1 = diss(B1,B2,B3,B4)

    _,_,pdf,acts,attacks = train(5000,80,1)


    d = acts[0]
    a = attacks[0]

    B1 = draw_u(0,a,d) #belief for attack_set1
    B2 = draw_u(1,a,d)
    B3 = draw_u(2,a,d)
    B4 = draw_u(3,a,d)


    h2 = diss(B1,B2,B3,B4)

    _,_,pdf,acts,attacks = train(5000,80,2)


    d = acts[0]
    a = attacks[0]

    B1 = draw_u(0,a,d) #belief for attack_set1
    B2 = draw_u(1,a,d)
    B3 = draw_u(2,a,d)
    B4 = draw_u(3,a,d)


    h3 = diss(B1,B2,B3,B4)

    _,_,pdf,acts,attacks = train(5000,80,3)


    d = acts[0]
    a = attacks[0]

    B1 = draw_u(0,a,d) #belief for attack_set1
    B2 = draw_u(1,a,d)
    B3 = draw_u(2,a,d)
    B4 = draw_u(3,a,d)


    h4 = diss(B1,B2,B3,B4)

    plt.plot([i for i in range(len(h1))],h1,label = 'Epsilon-Vacuity')
    plt.plot([i for i in range(len(h2))],h2,label = 'Epsilon-Dissonance')
    plt.plot([i for i in range(len(h3))],h3,label = 'Epsilon-Entropy')
    plt.plot([i for i in range(len(h4))],h4,label = 'Epsilon-Greedy')
    plt.xlabel('Number of episodes')
    plt.ylabel('Dissonance')
    plt.legend()
    plt.show()


def uncertain():

    _,_,pdf,acts,attacks = train(5000,80,0)


    d = acts[0]
    a = attacks[0]

    B1 = draw_u(0,a,d) #belief for attack_set1
    B2 = draw_u(1,a,d)
    B3 = draw_u(2,a,d)
    B4 = draw_u(3,a,d)


    u1 = [translateU(i[0:4]) for i in B1]

    _,_,pdf,acts,attacks = train(5000,80,1)


    d = acts[0]
    a = attacks[0]

    B1 = draw_u(0,a,d) #belief for attack_set1
    B2 = draw_u(1,a,d)
    B3 = draw_u(2,a,d)
    B4 = draw_u(3,a,d)


    u2 = [translateU(i[0:4]) for i in B1]

    _,_,pdf,acts,attacks = train(5000,80,2)


    d = acts[0]
    a = attacks[0]

    B1 = draw_u(0,a,d) #belief for attack_set1
    B2 = draw_u(1,a,d)
    B3 = draw_u(2,a,d)
    B4 = draw_u(3,a,d)


    u3 = [translateU(i[0:4]) for i in B1]

    _,_,pdf,acts,attacks = train(5000,80,3)


    d = acts[0]
    a = attacks[0]

    B1 = draw_u(0,a,d) #belief for attack_set1
    B2 = draw_u(1,a,d)
    B3 = draw_u(2,a,d)
    B4 = draw_u(3,a,d)

    u4 = [translateU(i[0:4]) for i in B1]

    plt.plot([i for i in range(len(u1))],u1,label = 'Epsilon-Vacuity')
    plt.plot([i for i in range(len(u2))],u2,label = 'Epsilon-Dissonance')
    plt.plot([i for i in range(len(u3))],u3,label = 'Epsilon-Entropy')
    plt.plot([i for i in range(len(u4))],u4,label = 'Epsilon-Greedy')
    plt.xlabel('Number of episodes')
    plt.ylabel('Vacuity')
    plt.legend()
    plt.show()


def com():
    #plt.plot([0,1,2,3],[1500,1600,1600,2000])
    plt.plot([0,1,2,3],[0.8494933066433129, 0.7629845929227341, 0.7245789526273134, 0.7182866968371004])
    plt.annotate('Epsilon-Vacuity', # this is the text
                    (0,0.8494933066433129), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
    plt.annotate('Epsilon-Dissonance', # this is the text
                    (1,0.7629845929227341), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
    plt.annotate('Epsilon-Entropy', # this is the text
                    (2,0.7245789526273134), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
    plt.annotate('Epsilon-Greedy', # this is the text
                    (3, 0.7182866968371004), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    
_,_,pdf,acts,attacks = train(5000,80,0)
d = acts[0]
a = attacks[0]

B1 = draw_u(0,a,d)
B2 = draw_u(1,a,d)
B3 = draw_u(2,a,d)
B4 = draw_u(3,a,d)
D1 = entropy(B1,B2,B3,B4) #entropy
#D1 = diss(B1,B2,B3,B4) #dissonance

plt.xlabel('Number of episodes')
plt.ylabel('Shannon entropy H(x)')
plt.plot([i for i in range(len(D1))],D1)
plt.legend()
plt.show()

#draw_e()
#draw_d()
#uncertain()
#com()


