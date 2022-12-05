#Uncertainty estimates (Shannon, Vacuity, Disonance)
import math
import numpy as np

#input
#action_sequence:string of actions peformed
def shannon(action_sequence,freqList = []):
    if len(freqList) == 0:
        action_sequence_List = list(action_sequence)
        action_sequence_set = list(set(action_sequence_List)) # list of symbols in the string
        print(action_sequence_set)

        # calculate the frequency of each symbol in the string
        freqList = []
        for action in action_sequence_set:
            count = 0
            for act in action_sequence_List:
                if act == action:
                    count += 1
            freqList.append(float(count) / len(action_sequence_List))

        print(freqList)

    # Shannon entropy
    entropy = 0.0
    for freq in freqList:
        entropy += freq * math.log(freq, 4)
    entropy = -entropy
    return entropy


def Bal(b_i, b_j):
    output = 1 - np.abs(b_i - b_j) / (b_i + b_j)
    return output

def Diss(b_list):
    b_diss = 0

    for i in b_list:
        up = 0
        down = 0
        for j in b_list:
            if i != j:
                up+=j * Bal(i,j)
                down+=j
        if down !=0:
            b_diss+=(i*up)/down

    return 1-b_diss
        
