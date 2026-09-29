import copy
import unittest

import numpy as np

from rl_training.memory import EpisodicMemory
from simulation.sat_config import PRESETS
from simulation.satellite_env import SingleAgentWrapper
from utils.orbital_decay import compute_beta


class TestOSGPolicyIntegration(unittest.TestCase):
    def _memory_with_orbital_episode(self) -> EpisodicMemory:
        memory = EpisodicMemory(capacity=10, filepath="/tmp/test_osg_policy_memory.pkl")
        memory.record(total_reward=12.0, collisions=0, fuel_outs=0, steps=20)
        episode = memory.episodes[-1]
        episode.orbital_state = {
            "altitude_km": 550.0,
            "true_anomaly": 0.0,
            "eclipse_fraction": 0.0,
        }
        episode.timestamp_step = 0
        return memory

    def test_osg_context_is_bounded_and_retrieval_derived(self):
        memory = self._memory_with_orbital_episode()
        beta, _, _ = compute_beta(550.0, 1.0 / 15.9)
        context = memory.osg_context(
            {"altitude_km": 550.0, "true_anomaly": 0.0, "eclipse_fraction": 0.0},
            current_timestep=1,
            beta=beta,
        )

        self.assertEqual(context.shape, (4,))
        self.assertEqual(context.dtype, np.float32)
        self.assertTrue(np.all(context >= 0.0))
        self.assertTrue(np.all(context <= 1.0))
        self.assertGreater(context[3], 0.0, "OSG confidence should come from a retrieved episode")

    def test_wrapper_appends_live_osg_context_and_records_episode(self):
        config = copy.deepcopy(PRESETS["starlink_leo"])
        config.num_satellites = 2
        memory = self._memory_with_orbital_episode()
        wrapper = SingleAgentWrapper(config=config, max_steps=2, agent_idx=0, memory=memory)

        observation, _ = wrapper.reset(seed=7)
        expected = memory.osg_context(
            wrapper._orbital_state(0), wrapper._osg_timestep, wrapper._osg_beta
        )
        np.testing.assert_allclose(observation["local"][-4:], expected)

        action = np.zeros(8, dtype=np.float32)
        wrapper.step(action)
        _, _, _, truncated, _ = wrapper.step(action)

        self.assertTrue(truncated)
        self.assertEqual(len(memory.episodes), 2)
        recorded = memory.episodes[-1]
        self.assertTrue(recorded.orbital_state)
        self.assertEqual(recorded.timestamp_step, 2)


if __name__ == "__main__":
    unittest.main()
