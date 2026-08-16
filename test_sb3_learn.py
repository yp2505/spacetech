import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return Monitor(gym.make("CartPole-v1"))

venv = DummyVecEnv([make_env])
model = PPO("MlpPolicy", venv, n_steps=256, verbose=0)
model.learn(total_timesteps=300)
print("Buffer len after 300:", len(model.ep_info_buffer))
model.learn(total_timesteps=300, reset_num_timesteps=False)
print("Buffer len after second 300:", len(model.ep_info_buffer))
