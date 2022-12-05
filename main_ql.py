import random
from qlearning import *
from EnvRL import EnvRL_v0
import matplotlib.pyplot as plt

def evaluate_QL(step, nids):
    print('Evalution for Q Learning:')
    total_reward = 0
    episodes = 100

    attack_set = [random.randint(0,3) for i in range(step)]
    env  = EnvRL_v0(attack_set)

    q_table,record,pdf,_,_ = train(step,nids)

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


q_table,rewards,actions,attacks = evaluate_QL(500,80) #500 is number of steps in one eps, 80 is the nids success rate
evaluate_random()


#Accuracy
# count = 0
# total = 0

# for i,j in zip(actions,attacks):
#     for x in range(len(i)):
#         if i[x]==j[x]:
#             count+=1
#         if i[x]!=-1:
#             total+=1
# print('total: ',total)
# print('same: ', count)
# print('accuracy: ',count/total)

# print('Q table:')
# print(q_table)

x = [i for i in range(len(rewards))]
plt.plot(x,rewards)
plt.xlabel('Number of episodes')
plt.ylabel('Average reward per episode')
plt.title('Epsilon-Greedy')
plt.show()


