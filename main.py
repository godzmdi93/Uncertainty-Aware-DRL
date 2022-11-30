from EnvRL import EnvRL_v0
from stable_baselines3 import DQN

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
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()