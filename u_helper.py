import math
from uncertainty_estimates import *
from bijective_mapping import *

import matplotlib.pyplot as plt

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
    return D1,D2,D3,D4


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

    return H1,H2,H3,H4
  
