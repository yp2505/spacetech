from rl_training.memory import EpisodicMemory
from simulation.sat_config import PRESETS
from simulation.satellite_env import SingleAgentWrapper
from stable_baselines3 import PPO

config = PRESETS["starlink_leo"]
config.num_satellites = 2
mem = EpisodicMemory(capacity=10)
env = SingleAgentWrapper(config, agent_idx=0, memory=mem)
model = PPO("MultiInputPolicy", env, n_steps=20, device="cpu")

obs, _ = env.reset()
print("Main model predict:")
a1, _ = model.predict(obs)
print(a1)

print("Policy predict:")
a2, _ = model.policy.predict(obs)
print(a2)

env.set_other_model(model.policy)
print("Stepping env:")
obs, rew, term, trunc, info = env.step(a1)
print("Step done!")
