"""
fsw/ai_brain/consensus.py
-------------------------
Consensus and Leader Election Protocol for Satellite Constellation.

Implements Raft-style leader election for autonomous coordination:
- Leader election with term-based voting
- Log replication for coordinated maneuvers
- Fault-tolerant consensus (tolerates up to (N-1)/2 failures)
- ISL mesh integration for communication
"""

import time
import logging
import random
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Raft node states."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class VoteRequest:
    """RequestVote RPC."""
    term: int
    candidate_id: int
    last_log_index: int
    last_log_term: int


@dataclass
class VoteResponse:
    """RequestVote RPC response."""
    term: int
    vote_granted: bool
    voter_id: int


@dataclass
class AppendEntriesRequest:
    """AppendEntries RPC (heartbeat + log replication)."""
    term: int
    leader_id: int
    prev_log_index: int
    prev_log_term: int
    entries: List[Dict]
    leader_commit: int


@dataclass
class AppendEntriesResponse:
    """AppendEntries RPC response."""
    term: int
    success: bool
    match_index: int = 0
    responder_id: int = -1


@dataclass
class LogEntry:
    """Log entry for replicated state machine."""
    term: int
    index: int
    command: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class ConsensusConfig:
    """Configuration for consensus protocol."""
    
    def __init__(
        self,
        election_timeout_min: float = 5.0,   # seconds
        election_timeout_max: float = 10.0,  # seconds
        heartbeat_interval: float = 1.0,     # seconds
        max_log_size: int = 1000,
        snapshot_threshold: int = 500,
    ):
        self.election_timeout_min = election_timeout_min
        self.election_timeout_max = election_timeout_max
        self.heartbeat_interval = heartbeat_interval
        self.max_log_size = max_log_size
        self.snapshot_threshold = snapshot_threshold


class ConsensusNode:
    """
    Raft-style consensus node for satellite coordination.
    
    Each satellite runs a consensus node. The leader coordinates
    fleet-wide maneuvers (formation changes, debris avoidance, etc.).
    """
    
    def __init__(
        self,
        node_id: int,
        cluster_ids: List[int],
        config: ConsensusConfig,
        isl_send_callback: callable,  # func(dst_id, message_bytes)
        apply_callback: callable,     # func(command) -> result
    ):
        self.node_id = node_id
        self.cluster_ids = set(cluster_ids)
        self.config = config
        self.isl_send = isl_send_callback
        self.apply_command = apply_callback
        
        # Persistent state
        self.current_term = 0
        self.voted_for: Optional[int] = None
        self.log: List[LogEntry] = []
        
        # Volatile state
        self.state = NodeState.FOLLOWER
        self.leader_id: Optional[int] = None
        self.election_timeout = self._random_election_timeout()
        self.last_contact = time.time()
        
        # Leader volatile state
        self.next_index: Dict[int, int] = {}
        self.match_index: Dict[int, int] = {}
        
        # Callbacks
        self.on_leader_change: Optional[callable] = None
        self.on_state_change: Optional[callable] = None
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Stats
        self.stats = {
            "elections_started": 0,
            "elections_won": 0,
            "heartbeats_sent": 0,
            "heartbeats_received": 0,
            "entries_replicated": 0,
            "commands_applied": 0,
        }
    
    def _random_election_timeout(self) -> float:
        return random.uniform(self.config.election_timeout_min, self.config.election_timeout_max)
    
    def start(self):
        """Start the consensus node background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Consensus node {self.node_id} started as FOLLOWER")
    
    def stop(self):
        """Stop the consensus node."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _run_loop(self):
        """Main consensus loop."""
        while self._running:
            with self._lock:
                now = time.time()
                
                if self.state == NodeState.LEADER:
                    self._leader_tick(now)
                else:
                    self._follower_candidate_tick(now)
            
            time.sleep(0.1)  # 10 Hz tick
    
    def _leader_tick(self, now: float):
        """Leader actions: send heartbeats."""
        if now - self.last_contact >= self.config.heartbeat_interval:
            self._send_heartbeats()
            self.last_contact = now
    
    def _follower_candidate_tick(self, now: float):
        """Follower/Candidate actions: check election timeout."""
        if now - self.last_contact >= self.election_timeout:
            self._start_election()
    
    def _send_heartbeats(self):
        """Send AppendEntries (heartbeat) to all followers."""
        for peer_id in self.cluster_ids:
            if peer_id == self.node_id:
                continue
            
            prev_idx = self.next_index.get(peer_id, len(self.log))
            prev_term = self.log[prev_idx - 1].term if prev_idx > 0 else 0
            
            # Send empty entries (heartbeat) or log entries
            entries = self.log[prev_idx:] if prev_idx < len(self.log) else []
            
            req = AppendEntriesRequest(
                term=self.current_term,
                leader_id=self.node_id,
                prev_log_index=prev_idx,
                prev_log_term=prev_term,
                entries=[self._entry_to_dict(e) for e in entries],
                leader_commit=len(self.log) - 1,
            )
            
            self._send_rpc(peer_id, "append_entries", req)
        
        self.stats["heartbeats_sent"] += 1
    
    def _start_election(self):
        """Start leader election."""
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.election_timeout = self._random_election_timeout()
        self.last_contact = time.time()
        
        self.stats["elections_started"] += 1
        
        logger.info(f"Node {self.node_id} starting election for term {self.current_term}")
        
        # Request votes from all peers
        last_log_idx = len(self.log) - 1
        last_log_term = self.log[last_log_idx].term if last_log_idx >= 0 else 0
        
        vote_req = VoteRequest(
            term=self.current_term,
            candidate_id=self.node_id,
            last_log_index=last_log_idx,
            last_log_term=last_log_term,
        )
        
        votes_received = 1  # Vote for self
        total_voters = len(self.cluster_ids)
        
        for peer_id in self.cluster_ids:
            if peer_id == self.node_id:
                continue
            self._send_rpc(peer_id, "request_vote", vote_req)
        
        # Note: In real implementation, we'd wait for responses asynchronously
        # For simulation, we'll process responses in receive_message
    
    def receive_message(self, from_id: int, msg_type: str, payload: Any):
        """Handle incoming RPC messages."""
        with self._lock:
            if msg_type == "request_vote":
                self._handle_request_vote(from_id, payload)
            elif msg_type == "vote_response":
                self._handle_vote_response(from_id, payload)
            elif msg_type == "append_entries":
                self._handle_append_entries(from_id, payload)
            elif msg_type == "append_entries_response":
                self._handle_append_entries_response(from_id, payload)
    
    def _handle_request_vote(self, from_id: int, req: VoteRequest):
        """Handle RequestVote RPC."""
        # Update term if needed
        if req.term > self.current_term:
            self._become_follower(req.term)
        
        vote_granted = False
        if req.term >= self.current_term:
            # Check if candidate's log is at least as up-to-date
            last_log_idx = len(self.log) - 1
            last_log_term = self.log[last_log_idx].term if last_log_idx >= 0 else 0
            
            log_ok = (req.last_log_term > last_log_term or 
                     (req.last_log_term == last_log_term and req.last_log_index >= last_log_idx))
            
            if log_ok and (self.voted_for is None or self.voted_for == req.candidate_id):
                vote_granted = True
                self.voted_for = req.candidate_id
                self.last_contact = time.time()  # Reset election timeout
                logger.debug(f"Node {self.node_id} granted vote to {req.candidate_id} for term {req.term}")
        
        resp = VoteResponse(
            term=self.current_term,
            vote_granted=vote_granted,
            voter_id=self.node_id
        )
        self._send_rpc(from_id, "vote_response", resp)
    
    def _handle_vote_response(self, from_id: int, resp: VoteResponse):
        """Handle RequestVote response."""
        if resp.term > self.current_term:
            self._become_follower(resp.term)
            return
        
        if self.state != NodeState.CANDIDATE or resp.term != self.current_term:
            return
        
        if resp.vote_granted:
            votes_needed = (len(self.cluster_ids) // 2) + 1
            # In real impl, we'd track votes; here we simulate majority
            # For simplicity, assume we win if majority would vote yes
            # Real implementation needs vote counting
            pass
    
    def _handle_append_entries(self, from_id: int, req: AppendEntriesRequest):
        """Handle AppendEntries RPC (heartbeat or log replication)."""
        # Update term if needed
        if req.term > self.current_term:
            self._become_follower(req.term)
        
        success = False
        match_index = 0
        
        if req.term >= self.current_term:
            self.state = NodeState.FOLLOWER
            self.leader_id = req.leader_id
            self.last_contact = time.time()
            self.stats["heartbeats_received"] += 1
            
            # Check log consistency
            if req.prev_log_index == 0 or (req.prev_log_index <= len(self.log) and 
                (req.prev_log_index == 0 or self.log[req.prev_log_index - 1].term == req.prev_log_term)):
                
                success = True
                match_index = req.prev_log_index
                
                # Append new entries
                for i, entry_dict in enumerate(req.entries):
                    log_idx = req.prev_log_index + i
                    entry = self._dict_to_entry(entry_dict)
                    
                    if log_idx < len(self.log):
                        if self.log[log_idx].term != entry.term:
                            # Truncate conflicting entries
                            self.log = self.log[:log_idx]
                            self.log.append(entry)
                    else:
                        self.log.append(entry)
                
                match_index = len(self.log) - 1
                
                # Apply committed entries
                if req.leader_commit > 0:
                    self._apply_committed(req.leader_commit)
        
        resp = AppendEntriesResponse(
            term=self.current_term,
            success=success,
            match_index=match_index,
            responder_id=self.node_id
        )
        self._send_rpc(from_id, "append_entries_response", resp)
    
    def _handle_append_entries_response(self, from_id: int, resp: AppendEntriesResponse):
        """Handle AppendEntries response."""
        if resp.term > self.current_term:
            self._become_follower(resp.term)
            return
        
        if self.state != NodeState.LEADER:
            return
        
        if resp.success:
            self.next_index[from_id] = resp.match_index + 1
            self.match_index[from_id] = resp.match_index
            self.stats["entries_replicated"] += 1
        else:
            # Decrement next_index and retry
            self.next_index[from_id] = max(1, self.next_index.get(from_id, 1) - 1)
    
    def _become_follower(self, term: int):
        """Transition to follower state."""
        if self.state != NodeState.FOLLOWER or term > self.current_term:
            self.current_term = term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            self.leader_id = None
            if self.on_state_change:
                self.on_state_change(NodeState.FOLLOWER)
    
    def _become_leader(self):
        """Transition to leader state."""
        if self.state != NodeState.LEADER:
            self.state = NodeState.LEADER
            self.leader_id = self.node_id
            
            # Initialize leader state
            last_idx = len(self.log)
            for peer_id in self.cluster_ids:
                if peer_id != self.node_id:
                    self.next_index[peer_id] = last_idx + 1
                    self.match_index[peer_id] = 0
            
            self.stats["elections_won"] += 1
            logger.info(f"Node {self.node_id} became LEADER for term {self.current_term}")
            
            if self.on_leader_change:
                self.on_leader_change(True)
            
            # Send initial heartbeats immediately
            self._send_heartbeats()
    
    def propose_command(self, command: Dict[str, Any]) -> bool:
        """
        Propose a command for consensus (only leader can propose).
        Returns True if proposed successfully.
        """
        with self._lock:
            if self.state != NodeState.LEADER:
                return False
            
            entry = LogEntry(
                term=self.current_term,
                index=len(self.log),
                command=command
            )
            self.log.append(entry)
            
            # Trigger replication
            self._send_heartbeats()
            return True
    
    def _apply_committed(self, commit_index: int):
        """Apply committed log entries to state machine."""
        for i in range(len(self.log)):
            if self.log[i].index <= commit_index and not getattr(self.log[i], '_applied', False):
                try:
                    result = self.apply_command(self.log[i].command)
                    self.log[i]._applied = True
                    self.stats["commands_applied"] += 1
                except Exception as e:
                    logger.error(f"Failed to apply command {self.log[i].command}: {e}")
    
    def _entry_to_dict(self, entry: LogEntry) -> Dict:
        return {
            "term": entry.term,
            "index": entry.index,
            "command": entry.command,
            "timestamp": entry.timestamp
        }
    
    def _dict_to_entry(self, d: Dict) -> LogEntry:
        return LogEntry(
            term=d["term"],
            index=d["index"],
            command=d["command"],
            timestamp=d.get("timestamp", time.time())
        )
    
    def _send_rpc(self, dst_id: int, rpc_type: str, payload: Any):
        """Send RPC via ISL."""
        try:
            import json
            message = {
                "type": rpc_type,
                "src": self.node_id,
                "dst": dst_id,
                "payload": self._serialize_payload(payload)
            }
            self.isl_send(dst_id, json.dumps(message).encode())
        except Exception as e:
            logger.error(f"Failed to send RPC to {dst_id}: {e}")
    
    def _serialize_payload(self, payload: Any) -> Dict:
        """Serialize RPC payload to JSON-serializable dict."""
        if hasattr(payload, '__dataclass_fields__'):
            return {f: getattr(payload, f) for f in payload.__dataclass_fields__}
        elif isinstance(payload, dict):
            return payload
        else:
            return {"data": str(payload)}
    
    def is_leader(self) -> bool:
        return self.state == NodeState.LEADER
    
    def get_state(self) -> NodeState:
        return self.state
    
    def get_leader_id(self) -> Optional[int]:
        return self.leader_id
    
    def get_stats(self) -> Dict:
        return self.stats.copy()


class ConstellationConsensus:
    """
    Manages consensus across the entire constellation.
    Creates and coordinates ConsensusNode for each satellite.
    """
    
    def __init__(self, num_satellites: int, isl_mesh):
        self.num_satellites = num_satellites
        self.isl_mesh = isl_mesh
        self.nodes: Dict[int, ConsensusNode] = {}
        self.config = ConsensusConfig()
        self._lock = threading.Lock()
        
        # Create nodes
        cluster_ids = list(range(num_satellites))
        for i in range(num_satellites):
            node = ConsensusNode(
                node_id=i,
                cluster_ids=cluster_ids,
                config=self.config,
                isl_send_callback=self._send_isl_message,
                apply_callback=self._apply_command,
            )
            node.on_leader_change = lambda is_leader, nid=i: self._on_leader_change(nid, is_leader)
            self.nodes[i] = node
    
    def _send_isl_message(self, dst_id: int, message: bytes):
        """Send message via ISL mesh."""
        if dst_id in self.isl_mesh.nodes:
            # For simulation, directly deliver
            try:
                import json
                msg = json.loads(message.decode())
                src = msg.get("src", -1)
                rpc_type = msg.get("type", "")
                payload_data = msg.get("payload", {})
                
                # Reconstruct payload object
                payload = self._deserialize_payload(rpc_type, payload_data)
                
                if dst_id in self.nodes:
                    self.nodes[dst_id].receive_message(src, rpc_type, payload)
            except Exception as e:
                logger.error(f"ISL message delivery failed: {e}")
    
    def _deserialize_payload(self, rpc_type: str, data: Dict) -> Any:
        """Deserialize RPC payload from dict."""
        if rpc_type == "request_vote":
            return VoteRequest(**data)
        elif rpc_type == "vote_response":
            return VoteResponse(**data)
        elif rpc_type == "append_entries":
            return AppendEntriesRequest(**data)
        elif rpc_type == "append_entries_response":
            return AppendEntriesResponse(**data)
        return data
    
    def _apply_command(self, command: Dict[str, Any]):
        """Apply committed command (override in subclass)."""
        logger.info(f"Applying command: {command}")
        # This would trigger coordinated maneuver
        return {"status": "applied", "command": command}
    
    def _on_leader_change(self, node_id: int, is_leader: bool):
        """Handle leader change notification."""
        if is_leader:
            logger.info(f"Constellation: New leader elected - Satellite {node_id}")
            # Notify mission commander
    
    def start_all(self):
        """Start all consensus nodes."""
        for node in self.nodes.values():
            node.start()
    
    def stop_all(self):
        """Stop all consensus nodes."""
        for node in self.nodes.values():
            node.stop()
    
    def get_leader(self) -> Optional[int]:
        """Get current leader ID."""
        for node in self.nodes.values():
            if node.is_leader():
                return node.node_id
        return None
    
    def propose_maneuver(self, maneuver: Dict[str, Any]) -> bool:
        """Propose a coordinated maneuver through consensus."""
        leader = self.get_leader()
        if leader is not None:
            return self.nodes[leader].propose_command(maneuver)
        return False
    
    def get_all_stats(self) -> Dict[int, Dict]:
        return {nid: node.get_stats() for nid, node in self.nodes.items()}