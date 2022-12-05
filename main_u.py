from re import U
from termios import B50
from qlearning import *

from bijective_mapping import *

from uncertainty_estimates import *

import matplotlib.pyplot as plt


_,_,pdf = train(500,50)


#calculate B U for observation 1 (which is attack set 1)
B1,B2,B3,B4 = [],[],[],[] #belief mass for each action
U1 = [] 
for i in pdf[0]:
    for j in range(len(i)):
        if j == 0:
            r = i[j]
            b_1 = translateB(r,i[0:4])
            B1.append(b_1)
        if j == 1:
            r = i[j]
            b_2 = translateB(r,i[0:4])
            B2.append(b_2)
        if j == 2:
            r = i[j]
            b_3 = translateB(r,i[0:4])
            B3.append(b_3)
        if j == 3:
            r = i[j]
            b_4 = translateB(r,i[0:4])
            B4.append(b_4)
    u = translateU(i[0:4])
    U1.append(u)

R1 = []
for i in range(len(B1)):
    R1.append(translateR(B1[i],U1[i]))

#B U R
# x = [i for i in range(len(U1))]
# #plt.plot(x,B1)
# plt.plot(x,U1)
# #plt.plot(x,R1)

#DPDF
# # x = [1,2,3,4]
# # y = np.random.uniform(0, 1, size=4)
# #plt.plot(x, [U1[0]+y[0]*B1[0],U1[0]+y[0]*B2[0],U1[0]+y[0]*B3[0],U1[0]+y[0]*B4[0]])

# plt.show()

#Shannon entropy
H = []
for i in pdf[0]:
    freqList = [i[0]/sum(i),i[1]/sum(i),i[2]/sum(i),i[3]/sum(i)]
    H.append(shannon([],freqList))

x = [i for i in range(len(H))]
plt.plot(x,H)
plt.show()

#Dissonance





