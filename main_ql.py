import random
from qlearning import *
from EnvRL import EnvRL_v0
import matplotlib.pyplot as plt

def plot(dis):
    # set width of bar
    barWidth = 0.2
    fig = plt.subplots(figsize =(12, 8))

    # set height of bar
    s0 = dis[0]
    s1 = dis[1]
    s2 = dis[2]
    s3 = dis[3]

    # Set position of bar on X axis
    br1 = np.arange(len(s0))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]


    plt.bar(br1, s0, color ='r', width = barWidth,edgecolor ='grey', label ='Action1')
    plt.bar(br2, s1, color ='g', width = barWidth,edgecolor ='grey', label ='Action2')
    plt.bar(br3, s2, color ='b', width = barWidth,edgecolor ='grey', label ='ACtion3')
    plt.bar(br4, s3, color ='y', width = barWidth,edgecolor ='grey', label ='Action4')

    # Adding Xticks
    plt.xlabel('States', fontweight ='bold', fontsize = 15)
    plt.ylabel('Probability', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(s0))],['Attack1', 'Attack2', 'Attack3', 'Attack4'])

    plt.legend()
    plt.show()


def evaluate_QL(step, nids):
    print('Evalution for Q Learning:')
    total_reward = 0
    episodes = 100

    attack_set = [random.randint(0,3) for i in range(step)]
    env  = EnvRL_v0(attack_set)

    q_table,record = train(step,nids)

    actions = []

    for _ in range(episodes):

        state = env.attack_set[env.reset()]
        epochs, reward = 0, 0
        
        done = False

        action_list = []
        
        while not done:
            nids_success_rate = random.randrange(0, 100)
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
        total_reward+=reward

    print(f"Results after {episodes} episodes:")
    print(f"Average reward per episode: {total_reward / episodes}")

    return q_table,record,actions

def evaluate_random():

    print('Evalution for Random:')
    total_reward = 0
    episodes = 100

    attack_set = [randint(0,3) for i in range(50)]
    env  = EnvRL_v0(attack_set)

    for _ in range(episodes):

        state = env.attack_set[env.reset()]
        epochs, reward = 0, 0
        
        done = False
        
        while not done:
            nids_success_rate = random.randrange(0, 100)
            if nids_success_rate < 80:
                action = randint(0,3)
                state, reward, done, info = env.step(action)
                state = env.attack_set[state]
            epochs += 1

        total_reward+=reward

    print(f"Results after {episodes} episodes:")
    print(f"Average reward per episode: {total_reward / episodes}")


q_table,rewards,actions = evaluate_QL(500,20) #500 is number of steps in one eps, 80 is the nids success rate
evaluate_random()

#this will give you an action sequence, [0,1,2,3,2,1,...]; 
# if -1 appears, it means we don't take any action since nids failed
print('Actions for eps 1:')
print(actions[0])

print('Q table:')
print(q_table)

print('Plot action distributions for each state')
plot(q_table)

print('Comparison between Q-Learning and Random algorithm')
x = [i for i in range(len(rewards[0]))]
plt.plot(x,rewards[0])
plt.xlabel('Number of steps')
plt.ylabel('Accumulated Rewards')
plt.show()


