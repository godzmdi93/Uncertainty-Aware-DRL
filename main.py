from EnvRL import EnvRL_v0
from stable_baselines3 import DQN
import random

env = EnvRL_v0()
env.reset()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100, log_interval=4)
model.save("dqn_opinion")

# del model # remove to demonstrate saving and loading

model = DQN.load("dqn_opinion")

episodes = 10000
obs = env.reset()
for ep in range(episodes):
    action, _state = model.predict(obs, deterministic=True)

    nids_success_rate = random.randrange(0, 100)
    if nids_success_rate < 80:
        env.take_action(action)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    else:
        continue

