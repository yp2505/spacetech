import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.step_cnt = 0
    def reset(self, seed=None, options=None):
        self.step_cnt = 0
        return np.zeros(4), {}
    def step(self, action):
        self.step_cnt += 1
        return np.zeros(4), 1.0, False, self.step_cnt >= 10, {}

def make_env():
    return Monitor(DummyEnv())

venv = DummyVecEnv([make_env])
model = PPO("MlpPolicy", venv, n_steps=20, verbose=0)
model.learn(total_timesteps=30)
print("Buffer len:", len(model.ep_info_buffer))
if len(model.ep_info_buffer) > 0:
    print("Max R:", np.max([ep["r"] for ep in model.ep_info_buffer]))
