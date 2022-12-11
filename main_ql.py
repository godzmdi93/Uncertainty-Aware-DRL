from cProfile import label
import random
from qlearning import *
from EnvRL import EnvRL_v0
import matplotlib.pyplot as plt

def evaluate_QL(step, nids,type):
    print('Evalution for Q Learning:')
    total_reward = 0
    episodes = 100

    attack_set = [random.randint(0,3) for i in range(step)]
    env  = EnvRL_v0(attack_set)

    q_table,record,pdf,_,_ = train(step,nids,type)

    actions = []
    attacks = []

    for _ in range(episodes):

        state = env.attack_set[env.reset()]
        epochs, reward = 0, 0
        
        done = False

        action_list = []
        attack_list = []
        
        while not done:
            nids_success_rate = random.randrange(0, 100)
            attack_list.append(env.attack_set[env.current_attack])
            if nids_success_rate < nids:
                action = np.argmax(q_table[state])
                state, reward, done, info = env.step(action)
                state = env.attack_set[state]
                action_list.append(action)
            else:
                if env.current_attack+1 == len(env.attack_set)-1:
                    done = True
                env.current_attack+=1
                action_list.append(-1)

            epochs += 1
        actions.append(action_list)
        attacks.append(attack_list)
        total_reward+=reward

    print(f"Results after {episodes} episodes:")
    print(f"Average reward per episode: {total_reward / episodes}")

    return q_table,record,actions,attacks

def evaluate_random():

    print('Evalution for Random:')
    total_reward = 0
    episodes = 100

    attack_set = [randint(0,3) for i in range(500)]
    env  = EnvRL_v0(attack_set)

    actions = []
    attacks = []

    for _ in range(episodes):

        state = env.attack_set[env.reset()]
        epochs, reward = 0, 0
        
        done = False

        action_list = []
        attack_list = []
        
        while not done:
            nids_success_rate = random.randrange(0, 100)
            attack_list.append(env.attack_set[env.current_attack])
            if nids_success_rate < 80:
                action = randint(0,3)
                action_list.append(action)
                state, reward, done, info = env.step(action)
                state = env.attack_set[state]
            epochs += 1

        actions.append(action_list)
        attacks.append(attack_list)
        total_reward+=reward

    print(f"Results after {episodes} episodes:")
    print(f"Average reward per episode: {total_reward / episodes}")

    return actions,attacks


q_table,r0,act1,atk1 = evaluate_QL(1000,80,0) #1000 is number of steps in one eps, 80 is the nids success rate
q_table,r1,act2,atk2 = evaluate_QL(1000,50,1)
q_table,r2,act3,atk3 = evaluate_QL(1000,50,2)
q_table,r3,act4,atk4 = evaluate_QL(1000,60,3)
evaluate_random()


record = []
#Accuracy
count = 0
total = 0

for i,j in zip(act1,atk1):
    for x in range(len(i)):
        if i[x]==j[x]:
            count+=1
        if i[x]!=-1:
            total+=1
record.append(count/total)

for i,j in zip(act2,atk2):
    for x in range(len(i)):
        if i[x]==j[x]:
            count+=1
        if i[x]!=-1:
            total+=1
record.append(count/total)

for i,j in zip(act3,atk3):
    for x in range(len(i)):
        if i[x]==j[x]:
            count+=1
        if i[x]!=-1:
            total+=1
record.append(count/total)

for i,j in zip(act4,atk4):
    for x in range(len(i)):
        if i[x]==j[x]:
            count+=1
        if i[x]!=-1:
            total+=1
record.append(count/total)

print('Accuracy for each setting:')
print(record)

# print('Q table:')
# print(q_table)

# filter_length = 50
# avg0 = np.convolve(r0,np.ones((filter_length)),mode = 'same')
# avg0 /= filter_length
# avg1 = np.convolve(r1,np.ones((filter_length)),mode = 'same')
# avg1 /= filter_length
# avg2 = np.convolve(r2,np.ones((filter_length)),mode = 'same')
# avg2 /= filter_length
# avg3 = np.convolve(r3,np.ones((filter_length)),mode = 'same')
# avg3 /= filter_length

x = [i for i in range(len(r0))]
# # plt.plot(x[50:-50],avg0[50:-50],label = 'Epsilon-Vacuity')
# # plt.plot(x[50:-50],avg1[50:-50],label = 'Epsilon-Dissonance')
# # plt.plot(x[50:-50],avg2[50:-50],label = 'Epsilon-Entropy')
# # plt.plot(x[50:-50],avg3[50:-50],label = 'Epsilon-Greedy')
plt.plot(x,r0,label = 'Epsilon-Vacuity')
plt.plot(x,r1,label = 'Epsilon-Dissonance')
plt.plot(x,r2,label = 'Epsilon-Entropy')
plt.plot(x,r3,label = 'Epsilon-Greedy')
plt.xlabel('Number of episodes')
plt.ylabel('Average reward per episode')
plt.legend()
plt.show()


