"""
rl_training/distributed_memory.py
==================================
Distributed Episodic Memory with Gossip Protocol.

Enables satellites to share salient experiences across the constellation
via ISL mesh, implementing epidemic/gossip-style dissemination.
"""

import time
import logging
import random
import threading
import pickle
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from rl_training.memory import EpisodicMemory, EpisodeRecord

logger = logging.getLogger(__name__)


@dataclass
class GossipMessage:
    """Gossip protocol message for memory exchange."""
    message_type: str  # "SYNC_REQUEST", "SYNC_RESPONSE", "EPISODE_PUSH"
    sender_id: int
    timestamp: float
    payload: Dict
    message_id: str = field(default_factory=lambda: f"{random.randint(0, 1_000_000_000)}_{time.time()}")


@dataclass
class MemorySyncState:
    """State for tracking memory synchronization with peers."""
    peer_id: int
    last_sync: float = 0.0
    known_episode_ids: Set[int] = field(default_factory=set)
    pending_episodes: List[EpisodeRecord] = field(default_factory=list)
    sync_in_progress: bool = False


class GossipConfig:
    """Configuration for gossip protocol."""
    
    def __init__(
        self,
        gossip_interval: float = 10.0,          # seconds between gossip rounds
        fanout: int = 2,                        # number of peers to gossip with per round
        max_episodes_per_message: int = 10,     # max episodes to send in one message
        sync_batch_size: int = 20,              # episodes per sync batch
        episode_ttl: float = 86400.0,           # time-to-live for episodes (24 hours)
        min_salience_threshold: float = 0.5,    # minimum salience to gossip
        enable_push: bool = True,               # push new episodes proactively
        enable_pull: bool = True,               # pull missing episodes periodically
    ):
        self.gossip_interval = gossip_interval
        self.fanout = fanout
        self.max_episodes_per_message = max_episodes_per_message
        self.sync_batch_size = sync_batch_size
        self.episode_ttl = episode_ttl
        self.min_salience_threshold = min_salience_threshold
        self.enable_push = enable_push
        self.enable_pull = enable_pull


class DistributedEpisodicMemory:
    """
    Distributed Episodic Memory with Gossip-based Synchronization.
    
    Each satellite maintains a local EpisodicMemory and participates
    in a gossip protocol to disseminate salient episodes across
    the constellation via ISL mesh.
    """
    
    def __init__(
        self,
        node_id: int,
        cluster_ids: List[int],
        local_memory: EpisodicMemory,
        config: GossipConfig,
        isl_send_callback: callable,  # func(dst_id, message_bytes)
    ):
        self.node_id = node_id
        self.cluster_ids = set(cluster_ids)
        self.local_memory = local_memory
        self.config = config
        self.isl_send = isl_send_callback
        
        # Sync state per peer
        self.sync_states: Dict[int, MemorySyncState] = {
            pid: MemorySyncState(peer_id=pid) for pid in cluster_ids if pid != node_id
        }
        
        # Message tracking
        self.seen_message_ids: Set[str] = set()
        self.message_history: deque = deque(maxlen=1000)
        
        # Stats
        self.stats = {
            "episodes_sent": 0,
            "episodes_received": 0,
            "syncs_initiated": 0,
            "syncs_completed": 0,
            "messages_dropped": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        }
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_gossip = 0.0
    
    def start(self):
        """Start gossip background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._gossip_loop, daemon=True)
        self._thread.start()
        logger.info(f"Distributed memory gossip started for node {self.node_id}")
    
    def stop(self):
        """Stop gossip thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _gossip_loop(self):
        """Main gossip loop."""
        while self._running:
            with self._lock:
                now = time.time()
                
                if now - self._last_gossip >= self.config.gossip_interval:
                    self._gossip_round(now)
                    self._last_gossip = now
                
                # Clean up old messages
                self._cleanup_old_messages(now)
            
            time.sleep(1.0)
    
    def _gossip_round(self, now: float):
        """Execute one gossip round."""
        # Select random peers (fanout)
        available_peers = [pid for pid in self.cluster_ids if pid != self.node_id]
        if not available_peers:
            return
        
        selected = random.sample(available_peers, min(self.config.fanout, len(available_peers)))
        
        for peer_id in selected:
            if self.config.enable_push:
                self._push_episodes(peer_id, now)
            
            if self.config.enable_pull:
                self._request_sync(peer_id, now)
    
    def _push_episodes(self, peer_id: int, now: float):
        """Push new salient episodes to peer."""
        sync_state = self.sync_states[peer_id]
        
        # Get episodes peer doesn't know about
        unknown_episodes = []
        for ep in self.local_memory.episodes:
            if ep.episode_id not in sync_state.known_episode_ids:
                # Check salience threshold and TTL
                if ep.salience >= self.config.min_salience_threshold:
                    age = now - ep.timestamp_step if hasattr(ep, 'timestamp_step') else 0
                    if age < self.config.episode_ttl:
                        unknown_episodes.append(ep)
        
        if not unknown_episodes:
            return
        
        # Sort by salience (most salient first)
        unknown_episodes.sort(key=lambda e: e.salience, reverse=True)
        to_send = unknown_episodes[:self.config.max_episodes_per_message]
        
        # Serialize episodes
        payload = {
            "episodes": [self._episode_to_dict(ep) for ep in to_send],
            "sender_id": self.node_id,
            "timestamp": now,
        }
        
        msg = GossipMessage(
            message_type="EPISODE_PUSH",
            sender_id=self.node_id,
            timestamp=now,
            payload=payload
        )
        
        self._send_gossip_message(peer_id, msg)
        
        # Update sync state
        for ep in to_send:
            sync_state.known_episode_ids.add(ep.episode_id)
        
        self.stats["episodes_sent"] += len(to_send)
    
    def _request_sync(self, peer_id: int, now: float):
        """Request synchronization with peer."""
        sync_state = self.sync_states[peer_id]
        if sync_state.sync_in_progress:
            return
        
        sync_state.sync_in_progress = True
        
        payload = {
            "known_episode_ids": list(sync_state.known_episode_ids)[-100:],  # Last 100
            "request_batch_size": self.config.sync_batch_size,
            "sender_id": self.node_id,
        }
        
        msg = GossipMessage(
            message_type="SYNC_REQUEST",
            sender_id=self.node_id,
            timestamp=now,
            payload=payload
        )
        
        self._send_gossip_message(peer_id, msg)
        self.stats["syncs_initiated"] += 1
    
    def _send_gossip_message(self, dst_id: int, msg: GossipMessage):
        """Send gossip message via ISL."""
        try:
            import json
            # Serialize message
            serialized = self._serialize_gossip_message(msg)
            self.isl_send(dst_id, json.dumps(serialized).encode())
            self.stats["bytes_sent"] += len(json.dumps(serialized).encode())
        except Exception as e:
            logger.error(f"Failed to send gossip message to {dst_id}: {e}")
            self.stats["messages_dropped"] += 1
    
    def _serialize_gossip_message(self, msg: GossipMessage) -> Dict:
        """Serialize gossip message to JSON."""
        return {
            "type": "GOSSIP",
            "message_type": msg.message_type,
            "sender_id": msg.sender_id,
            "timestamp": msg.timestamp,
            "message_id": msg.message_id,
            "payload": msg.payload
        }
    
    def receive_message(self, from_id: int, message_bytes: bytes):
        """Receive and process gossip message."""
        try:
            import json
            msg_data = json.loads(message_bytes.decode())
            
            if msg_data.get("type") != "GOSSIP":
                return
            
            msg = GossipMessage(
                message_type=msg_data["message_type"],
                sender_id=msg_data["sender_id"],
                timestamp=msg_data["timestamp"],
                payload=msg_data["payload"],
                message_id=msg_data["message_id"]
            )
            
            # Deduplication
            if msg.message_id in self.seen_message_ids:
                return
            self.seen_message_ids.add(msg.message_id)
            self.message_history.append(msg.message_id)
            
            self.stats["bytes_received"] += len(message_bytes)
            
            # Process based on type
            if msg.message_type == "EPISODE_PUSH":
                self._handle_episode_push(from_id, msg)
            elif msg.message_type == "SYNC_REQUEST":
                self._handle_sync_request(from_id, msg)
            elif msg.message_type == "SYNC_RESPONSE":
                self._handle_sync_response(from_id, msg)
                
        except Exception as e:
            logger.error(f"Failed to process gossip message from {from_id}: {e}")
    
    def _handle_episode_push(self, from_id: int, msg: GossipMessage):
        """Handle incoming episode push."""
        episodes_data = msg.payload.get("episodes", [])
        new_episodes = 0
        
        for ep_data in episodes_data:
            episode = self._dict_to_episode(ep_data)
            
            # Check if we already have this episode
            existing = any(e.episode_id == episode.episode_id for e in self.local_memory.episodes)
            if not existing:
                # Add to local memory
                self.local_memory.episodes.append(episode)
                new_episodes += 1
                
                # Update sync state
                if from_id in self.sync_states:
                    self.sync_states[from_id].known_episode_ids.add(episode.episode_id)
        
        if new_episodes > 0:
            # Re-sort by salience and trim
            self.local_memory.episodes.sort(key=lambda e: e.salience, reverse=True)
            if len(self.local_memory.episodes) > self.local_memory.capacity:
                self.local_memory.episodes = self.local_memory.episodes[:self.local_memory.capacity]
            
            # Recompute stats
            self.local_memory._recompute_stats()
            
            self.stats["episodes_received"] += new_episodes
            logger.debug(f"Node {self.node_id} received {new_episodes} new episodes from {from_id}")
    
    def _handle_sync_request(self, from_id: int, msg: GossipMessage):
        """Handle sync request - send missing episodes."""
        known_ids = set(msg.payload.get("known_episode_ids", []))
        batch_size = msg.payload.get("request_batch_size", self.config.sync_batch_size)
        
        # Find episodes we have that peer doesn't
        missing = []
        for ep in self.local_memory.episodes:
            if ep.episode_id not in known_ids and ep.salience >= self.config.min_salience_threshold:
                missing.append(ep)
                if len(missing) >= batch_size:
                    break
        
        if missing:
            payload = {
                "episodes": [self._episode_to_dict(ep) for ep in missing],
                "sender_id": self.node_id,
                "in_response_to": msg.message_id,
            }
            
            resp = GossipMessage(
                message_type="SYNC_RESPONSE",
                sender_id=self.node_id,
                timestamp=time.time(),
                payload=payload
            )
            
            self._send_gossip_message(from_id, resp)
            self.stats["episodes_sent"] += len(missing)
        
        # Update sync state
        if from_id in self.sync_states:
            self.sync_states[from_id].sync_in_progress = False
            self.stats["syncs_completed"] += 1
    
    def _handle_sync_response(self, from_id: int, msg: GossipMessage):
        """Handle sync response with episodes."""
        self._handle_episode_push(from_id, msg)
        
        if from_id in self.sync_states:
            self.sync_states[from_id].sync_in_progress = False
            self.stats["syncs_completed"] += 1
    
    def _episode_to_dict(self, episode: EpisodeRecord) -> Dict:
        """Serialize episode to dictionary."""
        return {
            "episode_id": episode.episode_id,
            "total_reward": episode.total_reward,
            "collisions": episode.collisions,
            "fuel_outs": episode.fuel_outs,
            "steps": episode.steps,
            "was_eclipse": episode.was_eclipse,
            "was_weather": episode.was_weather,
            "was_fault": episode.was_fault,
            "salience": episode.salience,
            "start_conditions": episode.start_conditions,
            "satellite_id": getattr(episode, 'satellite_id', -1),
            "orbital_state": getattr(episode, 'orbital_state', {}),
            "commander_goal": getattr(episode, 'commander_goal', []),
            "action_sequence": getattr(episode, 'action_sequence', []),
            "maneuver_performed": getattr(episode, 'maneuver_performed', ""),
            "fuel_cost": getattr(episode, 'fuel_cost', 0.0),
            "data_routed_via_isl": getattr(episode, 'data_routed_via_isl', False),
            "timestamp_step": getattr(episode, 'timestamp_step', 0),
        }
    
    def _dict_to_episode(self, data: Dict) -> EpisodeRecord:
        """Deserialize episode from dictionary."""
        episode = EpisodeRecord(
            episode_id=data["episode_id"],
            total_reward=data["total_reward"],
            collisions=data["collisions"],
            fuel_outs=data["fuel_outs"],
            steps=data["steps"],
            was_eclipse=data.get("was_eclipse", False),
            was_weather=data.get("was_weather", False),
            was_fault=data.get("was_fault", False),
            salience=data.get("salience", 0.0),
            start_conditions=data.get("start_conditions", {}),
        )
        # Restore extended fields
        episode.satellite_id = data.get("satellite_id", -1)
        episode.orbital_state = data.get("orbital_state", {})
        episode.commander_goal = data.get("commander_goal", [])
        episode.action_sequence = data.get("action_sequence", [])
        episode.maneuver_performed = data.get("maneuver_performed", "")
        episode.fuel_cost = data.get("fuel_cost", 0.0)
        episode.data_routed_via_isl = data.get("data_routed_via_isl", False)
        episode.timestamp_step = data.get("timestamp_step", 0)
        return episode
    
    def _cleanup_old_messages(self, now: float):
        """Clean up old message IDs."""
        # Keep only recent messages (last 10 minutes)
        cutoff = now - 600.0
        # In practice, we'd track timestamps; for now just limit set size
        if len(self.seen_message_ids) > 5000:
            # Remove oldest half
            old = list(self.seen_message_ids)[:2500]
            for mid in old:
                self.seen_message_ids.discard(mid)
    
    def record_local_episode(self, episode: EpisodeRecord):
        """Record a new local episode and trigger push."""
        # Add to local memory
        self.local_memory.episodes.append(episode)
        
        # Update local sync states
        for sync_state in self.sync_states.values():
            sync_state.known_episode_ids.add(episode.episode_id)
        
        # Trigger immediate push to random peer if enabled
        if self.config.enable_push and self.sync_states:
            peer_id = random.choice(list(self.sync_states.keys()))
            self._push_episodes(peer_id, time.time())
    
    def get_stats(self) -> Dict:
        return {
            **self.stats,
            "local_episodes": len(self.local_memory.episodes),
            "peer_count": len(self.sync_states),
            "memory_stats": self.local_memory.stats,
        }


class ConstellationMemoryGossip:
    """
    Manages distributed memory gossip across entire constellation.
    """
    
    def __init__(self, num_satellites: int, isl_mesh, local_memories: List[EpisodicMemory]):
        self.num_satellites = num_satellites
        self.isl_mesh = isl_mesh
        self.local_memories = local_memories
        self.config = GossipConfig()
        self.gossip_nodes: Dict[int, DistributedEpisodicMemory] = {}
        self._lock = threading.Lock()
        
        cluster_ids = list(range(num_satellites))
        for i in range(num_satellites):
            node = DistributedEpisodicMemory(
                node_id=i,
                cluster_ids=cluster_ids,
                local_memory=local_memories[i],
                config=self.config,
                isl_send_callback=self._send_isl_message,
            )
            self.gossip_nodes[i] = node
    
    def _send_isl_message(self, dst_id: int, message: bytes):
        """Send message via ISL mesh."""
        if dst_id in self.isl_mesh.nodes:
            try:
                import json
                msg = json.loads(message.decode())
                src = msg.get("sender_id", -1)
                if dst_id in self.gossip_nodes:
                    self.gossip_nodes[dst_id].receive_message(src, message)
            except Exception as e:
                logger.error(f"ISL message delivery failed: {e}")
    
    def start_all(self):
        for node in self.gossip_nodes.values():
            node.start()
    
    def stop_all(self):
        for node in self.gossip_nodes.values():
            node.stop()
    
    def record_episode(self, sat_idx: int, episode: EpisodeRecord):
        """Record episode from satellite."""
        if sat_idx in self.gossip_nodes:
            self.gossip_nodes[sat_idx].record_local_episode(episode)
    
    def get_all_stats(self) -> Dict[int, Dict]:
        return {nid: node.get_stats() for nid, node in self.gossip_nodes.items()}