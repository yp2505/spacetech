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

    # ── Orbital Salience Gating (OSG) ─────────────────────────────────────────

    @staticmethod
    def _orbital_state_to_vec(orbital_state: dict) -> np.ndarray:
        """
        Convert an orbital_state dict to a 1-D numpy float64 array.

        Keys are sorted alphabetically so the same set of keys always
        produces the same ordering.  Only numeric (int / float) values
        are included; non-numeric values are silently skipped.
        """
        if not orbital_state:
            return np.array([], dtype=np.float64)
        vec = []
        for k in sorted(orbital_state.keys()):
            v = orbital_state[k]
            if isinstance(v, (int, float)):
                vec.append(float(v))
        return np.array(vec, dtype=np.float64)

    def osg_score(self,
                  current_orbital_state: dict,
                  episode: "EpisodeRecord",
                  current_timestep: int,
                  beta: float = 0.001) -> float:
        """
        Compute the three-factor Orbital Salience Gating score.

        score = cosine_sim
                × sigmoid(R_j - R_mean)
                × exp(-beta × age)

        Where:
          cosine_sim  — cosine similarity between current and stored
                        orbital state vectors.
          R_j         — episode.total_reward
          R_mean      — mean total_reward across ALL stored episodes
                        (computed fresh from self.episodes each call).
          sigmoid(x)  — 1 / (1 + exp(-x))
          age         — current_timestep - episode.timestamp_step
                        (clamped to 0 if timestamp_step is missing / 0).

        Edge cases:
          • current_orbital_state is None or {}  → return 0.0
          • episode.orbital_state is None or {}  → return 0.0
          • Either vector norm is 0              → cosine_sim = 0.0
          • No stored episodes for R_mean        → R_mean = 0.0

        Args:
            current_orbital_state: The query orbital state dict.
            episode:               The candidate EpisodeRecord.
            current_timestep:      Global step counter at query time.
            beta:                  Temporal decay rate (per step).
                                   Use compute_beta() for a physics
                                   grounded value.

        Returns:
            float score in roughly [0, 1].
        """
        # ── guard: orbital states ────────────────────────────────────────────
        if not current_orbital_state:
            return 0.0
        ep_orbital = getattr(episode, "orbital_state", None)
        if not ep_orbital:
            return 0.0

        # ── cosine similarity ────────────────────────────────────────────────
        cur_vec = self._orbital_state_to_vec(current_orbital_state)
        ep_vec  = self._orbital_state_to_vec(ep_orbital)

        # Vectors must share the same dimensionality; fall back to 0 if not
        if cur_vec.size == 0 or ep_vec.size == 0 or cur_vec.shape != ep_vec.shape:
            cosine_sim = 0.0
        else:
            cur_norm = np.linalg.norm(cur_vec)
            ep_norm  = np.linalg.norm(ep_vec)
            if cur_norm < 1e-12 or ep_norm < 1e-12:
                cosine_sim = 0.0
            else:
                cosine_sim = float(np.dot(cur_vec, ep_vec) / (cur_norm * ep_norm))

        # ── reward salience: sigmoid(R_j - R_mean) ──────────────────────────
        R_j = float(episode.total_reward)
        if self.episodes:
            R_mean = float(np.mean([e.total_reward for e in self.episodes]))
        else:
            R_mean = 0.0
        reward_factor = 1.0 / (1.0 + np.exp(-(R_j - R_mean)))

        # ── temporal decay: exp(-beta × age) ────────────────────────────────
        ts = getattr(episode, "timestamp_step", 0) or 0
        age = max(0, current_timestep - ts)
        temporal_factor = float(np.exp(-beta * age))

        return cosine_sim * reward_factor * temporal_factor

    def osg_retrieve(self,
                     current_orbital_state: dict,
                     current_timestep: int,
                     top_k: int = 5,
                     beta: float = 0.001):
        """
        Retrieve the top-k most relevant past episodes using OSG scoring,
        alongside a cosine-only baseline for comparison.

        Steps:
          1. Loop through all stored episodes.
          2. Compute osg_score() for each.
          3. Compute a cosine-only baseline score for each (same cosine
             similarity, but reward and temporal factors are omitted).
          4. Sort both lists by their respective scores descending.
          5. Return the top_k from each.

        Args:
            current_orbital_state: The query orbital state dict.
            current_timestep:      Global step counter at query time.
            top_k:                 Number of top episodes to return.
            beta:                  Temporal decay rate (per step).

        Returns:
            Tuple (osg_results, baseline_results) where each element is
            a list of (score, EpisodeRecord) tuples of length ≤ top_k.
        """
        if not self.episodes:
            return [], []

        osg_scores      = []
        baseline_scores = []

        cur_vec  = self._orbital_state_to_vec(current_orbital_state) \
                   if current_orbital_state else np.array([])
        cur_norm = float(np.linalg.norm(cur_vec)) if cur_vec.size > 0 else 0.0

        for ep in self.episodes:
            # ── OSG score ───────────────────────────────────────────────────
            osg_s = self.osg_score(current_orbital_state, ep,
                                   current_timestep, beta=beta)
            osg_scores.append((osg_s, ep))

            # ── Baseline: cosine only ────────────────────────────────────────
            ep_orbital = getattr(ep, "orbital_state", None)
            if not ep_orbital or cur_vec.size == 0 or cur_norm < 1e-12:
                cos_s = 0.0
            else:
                ep_vec  = self._orbital_state_to_vec(ep_orbital)
                if ep_vec.shape != cur_vec.shape or ep_vec.size == 0:
                    cos_s = 0.0
                else:
                    ep_norm = float(np.linalg.norm(ep_vec))
                    if ep_norm < 1e-12:
                        cos_s = 0.0
                    else:
                        cos_s = float(np.dot(cur_vec, ep_vec) /
                                      (cur_norm * ep_norm))
            baseline_scores.append((cos_s, ep))

        osg_scores.sort(key=lambda x: x[0], reverse=True)
        baseline_scores.sort(key=lambda x: x[0], reverse=True)

        return osg_scores[:top_k], baseline_scores[:top_k]

    def print_osg_comparison(self,
                              osg_results: list,
                              baseline_results: list,
                              current_timestep: int = 0) -> None:
        """
        Print a side-by-side comparison table of OSG vs baseline retrieval.

        This is for debugging / proof-of-concept only — it shows which
        episodes each method selects so you can inspect whether OSG picks
        higher-reward, contextually richer episodes over stale low-reward
        ones favoured by cosine-only retrieval.

        Args:
            osg_results:       List of (score, EpisodeRecord) from osg_retrieve.
            baseline_results:  List of (score, EpisodeRecord) from osg_retrieve.
            current_timestep:  Used to compute age of each episode.
        """
        print("\n=== OSG vs Baseline Retrieval Comparison ===")
        print(f"{'Rank':<5} {'Method':<10} {'Score':<8} "
              f"{'Ep_ID':<7} {'Reward':<8} {'Age':<6}")
        print(f"{'-'*5} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*6}")

        max_rank = max(len(osg_results), len(baseline_results))
        for rank in range(max_rank):
            for label, results in [("OSG", osg_results),
                                   ("Baseline", baseline_results)]:
                if rank < len(results):
                    score, ep = results[rank]
                    ts  = getattr(ep, "timestamp_step", 0) or 0
                    age = max(0, current_timestep - ts)
                    print(f"{rank + 1:<5} {label:<10} {score:<8.3f} "
                          f"{ep.episode_id:<7} {ep.total_reward:<+8.1f} {age:<6}")

        print()

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
