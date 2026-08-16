import unittest
import os
import shutil
import warnings

from simulation.sat_config import PRESETS
from simulation.satellite_env import SingleAgentWrapper
from rl_training.memory import EpisodicMemory
from rl_training import train as train_module
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

class TestFSWTrainBuffer(unittest.TestCase):
    def test_rollout_guard_and_episode_buffer_population(self):
        """Guard against short PPO rollouts and ensure episode buffers stay populated."""
        config = PRESETS["starlink_leo"]
        config.num_satellites = 2

        original_n_envs = train_module.N_ENVS
        try:
            train_module.N_ENVS = 8
            env = SingleAgentWrapper(config=config, max_steps=600, agent_idx=0, memory=EpisodicMemory(capacity=10))
            with self.assertRaisesRegex(ValueError, "n_steps.*max_steps|rollout length"):
                train_module.build_or_load_model(env)
        finally:
            train_module.N_ENVS = original_n_envs

        primary_memory = EpisodicMemory(capacity=10)

        def _make_env():
            _env = SingleAgentWrapper(config=config, max_steps=10, agent_idx=0, memory=primary_memory)
            return Monitor(_env)

        train_env = DummyVecEnv([_make_env for _ in range(2)])
        model = PPO(
            "MultiInputPolicy", train_env,
            n_steps=20,
            batch_size=10,
            n_epochs=1,
            device="cpu"
        )
        train_env.env_method("set_other_model", model)

        for cycle in range(2):
            model.learn(total_timesteps=40, reset_num_timesteps=False)
            self.assertGreater(len(model.ep_info_buffer), 0, f"ep_info_buffer empty after cycle {cycle + 1}")
            train_env.env_method("set_other_model", model)

if __name__ == '__main__':
    unittest.main()
