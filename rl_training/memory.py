"""
memory.py
---------
Long-lived Episodic Memory for the multi-satellite AI.

Survives across training runs (persisted to disk as episodic_memory.pkl).
Stores the top-K most *salient* episodes so the agent can adapt its strategy
based on past experience — both successes and rare dangerous events.

Memory context vector (4 floats, all in [0, 1]):
  [0] mean_reward_norm  — overall historical performance level
  [1] collision_rate    — how often collisions happened  (lower = safer history)
  [2] fuel_out_rate     — how often fuel ran out         (lower = better history)
  [3] best_quality      — normalised quality of the best episode ever seen

This vector is appended to every local observation so the Actor can condition
its policy on its own long-term experience.
"""

import os
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import List

MEMORY_FILE     = "episodic_memory.pkl"
MEMORY_CAPACITY = 500   # keep the top-K most salient episodes


@dataclass
class EpisodeRecord:
    """One remembered episode."""
    episode_id:       int
    total_reward:     float
    collisions:       int
    fuel_outs:        int
    steps:            int
    was_eclipse:      bool   = False
    was_weather:      bool   = False
    was_fault:        bool   = False
    salience:         float  = field(default=0.0, compare=False)
    start_conditions: dict   = field(default_factory=dict)
    
    # Part 5: New traces
    satellite_id:        int = -1
    orbital_state:       dict = field(default_factory=dict)
    commander_goal:      list = field(default_factory=list)
    action_sequence:     list = field(default_factory=list)
    maneuver_performed:  str = ""
    fuel_cost:           float = 0.0
    data_routed_via_isl: bool = False
    timestamp_step:      int = 0


class EpisodicMemory:
    """
    Persistent episodic memory that survives across python train.py runs.

    Salience scoring
    ----------------
    Both very-good and very-bad episodes are worth keeping.
    Rare-event episodes (solar storm, collision, fuel-out) get a bonus so
    the agent never forgets how to handle edge cases.
    """

    def __init__(self, capacity: int = MEMORY_CAPACITY,
                 filepath: str = MEMORY_FILE):
        self.capacity     = capacity
        self.filepath     = filepath
        self.episodes:    List[EpisodeRecord] = []
        self.total_seen:  int   = 0
        self.best_reward: float = -1e9
        self.worst_reward:float =  1e9
        self._load()

    # ── recording ─────────────────────────────────────────────────────────────
    def record(self, total_reward: float, collisions: int, fuel_outs: int,
               steps: int, was_eclipse: bool = False,
               was_weather: bool = False, was_fault: bool = False,
               start_conditions: dict = None) -> None:
        """Add one episode to the memory bank."""
        self.total_seen   += 1
        self.best_reward   = max(self.best_reward,  total_reward)
        self.worst_reward  = min(self.worst_reward, total_reward)

        rec = EpisodeRecord(
            episode_id=self.total_seen,
            total_reward=total_reward,
            collisions=collisions,
            fuel_outs=fuel_outs,
            steps=steps,
            was_eclipse=was_eclipse,
            was_weather=was_weather,
            was_fault=was_fault,
            start_conditions=start_conditions or {},
        )
        rec.salience = self._salience(rec)
        self.episodes.append(rec)

        # Keep only the top-K by salience (prune oldest / least salient)
        if len(self.episodes) > self.capacity:
            self.episodes.sort(key=lambda e: e.salience, reverse=True)
            self.episodes = self.episodes[:self.capacity]

    def _salience(self, rec: EpisodeRecord) -> float:
        """
        Salience score — higher = more worth remembering.
        Large absolute reward (good or bad) + rare-event bonus.
        """
        base = abs(rec.total_reward)
        rare = (
            (2.0 if rec.was_weather  else 0.0) +
            (2.0 if rec.was_fault    else 0.0) +   # Phase E fault episodes are very salient
            (1.5 if rec.fuel_outs > 0 else 0.0) +
            (1.0 if rec.collisions > 0 else 0.0)
        )
        return base + rare

    # ── Prioritized Experience Replay ─────────────────────────────────────────
    def sample_high_reward(self, k: int) -> list:
        """
        Prioritized sampling: P(episode_i) ∝ (reward_i - min_reward + ε)^α

        High-reward episodes are selected much more often than average ones.
        This implements Prioritized Experience Replay (Schaul et al. 2015).

        Args:
            k: number of episodes to sample (without replacement where possible)

        Returns:
            List of EpisodeRecord objects, skewed toward high-reward episodes.
        """
        if not self.episodes:
            return []

        rewards = np.array([e.total_reward for e in self.episodes], dtype=np.float64)
        min_r   = rewards.min()

        # Priority: (reward - min + ε)^α
        # α = 0.6 is from Schaul et al. 2015 — balances uniform vs greedy
        priorities = (rewards - min_r + 1e-6) ** 0.6
        probs      = priorities / priorities.sum()

        k   = min(k, len(self.episodes))
        idx = np.random.choice(len(self.episodes), size=k, replace=False, p=probs)
        return [self.episodes[i] for i in idx]

    def top_k_episodes(self, k: int) -> list:
        """Return the k episodes with highest total reward (for inspection)."""
        return sorted(self.episodes, key=lambda e: e.total_reward, reverse=True)[:k]

    # ── context vector (appended to Actor's local obs) ─────────────────────────
    def get_context(self) -> np.ndarray:
        """
        Returns a 4-dim float32 vector in [0, 1].
        Safe to call before any episodes are recorded (returns neutral defaults).
        """
        if not self.episodes:
            # Neutral defaults — agent starts with no prior assumptions
            return np.array([0.5, 0.0, 0.0, 0.5], dtype=np.float32)

        rewards    = [e.total_reward for e in self.episodes]
        col_rates  = [e.collisions  / max(e.steps, 1) for e in self.episodes]
        fuel_rates = [e.fuel_outs   / max(e.steps, 1) for e in self.episodes]

        rng = (self.best_reward - self.worst_reward) + 1e-8

        # Normalise mean reward to [0, 1]
        mean_rew_01 = float(np.clip(
            (np.mean(rewards) - self.worst_reward) / rng, 0.0, 1.0
        ))
        mean_col    = float(np.clip(np.mean(col_rates),  0.0, 1.0))
        mean_fuel   = float(np.clip(np.mean(fuel_rates), 0.0, 1.0))
        best_qual   = float(np.clip(
            (self.best_reward - self.worst_reward) /
            (abs(self.best_reward) + abs(self.worst_reward) + 1e-8),
            0.0, 1.0
        ))

        return np.array([mean_rew_01, mean_col, mean_fuel, best_qual],
                        dtype=np.float32)

    # ── persistence ────────────────────────────────────────────────────────────
    def save(self) -> None:
        """Persist memory to disk."""
        with open(self.filepath, "wb") as f:
            pickle.dump({
                "episodes":     self.episodes,
                "total_seen":   self.total_seen,
                "best_reward":  self.best_reward,
                "worst_reward": self.worst_reward,
            }, f)

    def _load(self) -> None:
        """Restore memory from disk if it exists."""
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "rb") as f:
                data = pickle.load(f)
            self.episodes     = data["episodes"]
            self.total_seen   = data["total_seen"]
            self.best_reward  = data["best_reward"]
            self.worst_reward = data["worst_reward"]

            # ── Backward-compatibility migration ──────────────────────────────
            # Old pickles won't have start_conditions on each EpisodeRecord.
            # Patch them in-place so sample_high_reward() works safely.
            migrated = 0
            for ep in self.episodes:
                if not hasattr(ep, "start_conditions"):
                    ep.start_conditions = {}
                    migrated += 1
                if not hasattr(ep, "was_fault"):
                    ep.was_fault = False
                    migrated += 1
            if migrated:
                print(f"[EpisodicMemory] [INFO] Migrated {migrated} records "
                      f"(added start_conditions={{}})")

            print(
                f"[EpisodicMemory] [OK] Restored {len(self.episodes)} episodes "
                f"({self.total_seen} total seen | "
                f"Best: {self.best_reward:+.1f} | "
                f"Worst: {self.worst_reward:+.1f})"
            )
        except Exception as exc:
            print(f"[EpisodicMemory] [WARN] Could not load '{self.filepath}': {exc}. Starting fresh.")

    def query_similar_episode(self, current_orbital_state: dict, threshold=0.8):
        """
        Cross-satellite memory query: finds a past episode with a highly similar
        orbital state (true anomaly, altitude, eclipse).
        Uses cosine similarity.
        """
        if not self.episodes:
            return None
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return None
            
        def to_vec(state):
            return np.array([[
                state.get("true_anomaly", 0.0),
                state.get("altitude_km", 500.0),
                state.get("eclipse_fraction", 0.0)
            ]])
            
        current_vec = to_vec(current_orbital_state)
        
        best_ep = None
        best_sim = -1.0
        
        for ep in self.episodes:
            if not ep.orbital_state:
                continue
            ep_vec = to_vec(ep.orbital_state)
            sim = cosine_similarity(current_vec, ep_vec)[0][0]
            if sim > best_sim and sim > threshold:
                best_sim = sim
                best_ep = ep
                
        return best_ep

    # ── summary ────────────────────────────────────────────────────────────────
    def _recompute_stats(self) -> None:
        """Recompute best/worst/total_seen from current episodes."""
        if not self.episodes:
            self.best_reward = -1e9
            self.worst_reward = 1e9
            self.total_seen = 0
        else:
            rewards = [e.total_reward for e in self.episodes]
            self.best_reward = max(rewards)
            self.worst_reward = min(rewards)
            self.total_seen = max(e.episode_id for e in self.episodes)

    @property
    def stats(self) -> str:
        if not self.episodes:
            return "Memory empty — no episodes recorded yet."
        rewards = [e.total_reward for e in self.episodes]
        return (
            f"{len(self.episodes)} episodes stored / {self.total_seen} total seen | "
            f"Best: {self.best_reward:+.1f} | "
            f"Worst: {self.worst_reward:+.1f} | "
            f"Avg(stored): {np.mean(rewards):+.1f}"
        )
