"""
fsw/ai_brain/coordination.py
----------------------------
Cross-Satellite Action Coordination for Autonomous Constellation Operations.

Enables coordinated multi-satellite maneuvers:
- Formation reconfiguration
- Coordinated debris avoidance
- Distributed data relay scheduling
- Joint orbital maneuvers
- Conflict resolution for shared resources
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np

from fsw.ai_brain.consensus import ConsensusNode, ConstellationConsensus

logger = logging.getLogger(__name__)


class ManeuverType(Enum):
    """Types of coordinated maneuvers."""
    FORMATION_CHANGE = "formation_change"
    DEBRIS_AVOIDANCE = "debris_avoidance"
    COLLISION_AVOIDANCE = "collision_avoidance"
    DATA_RELAY_SCHEDULE = "data_relay_schedule"
    ORBITAL_RAISE = "orbital_raise"
    ORBITAL_LOWER = "orbital_lower"
    PLANE_CHANGE = "plane_change"
    DIFFERENTIAL_DRAG = "differential_drag"
    EMERGENCY_DEORBIT = "emergency_deorbit"


class CoordinationState(Enum):
    """State of a coordination action."""
    PROPOSED = "proposed"
    VOTING = "voting"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class CoordinationAction:
    """A coordinated action across multiple satellites."""
    action_id: str
    maneuver_type: ManeuverType
    participants: List[int]  # Satellite indices
    initiator: int
    parameters: Dict[str, Any]  # Maneuver-specific parameters
    state: CoordinationState = CoordinationState.PROPOSED
    created_at: float = field(default_factory=time.time)
    approved_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    votes: Dict[int, bool] = field(default_factory=dict)  # participant_id -> vote
    execution_results: Dict[int, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more urgent
    
    def is_unanimous(self) -> bool:
        return all(v for v in self.votes.values())
    
    def has_majority(self) -> bool:
        if not self.votes:
            return False
        yes_votes = sum(1 for v in self.votes.values() if v)
        return yes_votes >= (len(self.participants) // 2) + 1


@dataclass
class FormationTarget:
    """Target formation configuration."""
    formation_type: str  # "walker_delta", "string_of_pearls", "cluster", "custom"
    num_planes: int
    sats_per_plane: int
    altitude_km: float
    inclination_deg: float
    raan_spacing_deg: float
    anomaly_spacing_deg: float
    phase_offset_deg: float = 0.0


class CoordinationEngine:
    """
    Cross-Satellite Action Coordination Engine.
    
    Manages coordinated maneuvers through consensus-based approval
    and execution monitoring.
    """
    
    def __init__(
        self,
        sat_idx: int,
        num_satellites: int,
        consensus: ConsensusNode,
        get_state_callback: callable,  # func(sat_idx) -> SubsystemState
        send_command_callback: callable,  # func(sat_idx, command_dict)
    ):
        self.sat_idx = sat_idx
        self.num_satellites = num_satellites
        self.consensus = consensus
        self.get_state = get_state_callback
        self.send_command = send_command_callback
        
        # Pending and active coordinations
        self.pending_actions: Dict[str, CoordinationAction] = {}
        self.active_actions: Dict[str, CoordinationAction] = {}
        self.completed_actions: deque = deque(maxlen=100)
        
        # Local state
        self.current_formation: Optional[FormationTarget] = None
        self.maneuver_sequence: List[Dict] = []  # Planned sequence for this sat
        
        # Callbacks
        self.on_action_approved: Optional[callable] = None
        self.on_action_completed: Optional[callable] = None
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Stats
        self.stats = {
            "actions_proposed": 0,
            "actions_approved": 0,
            "actions_executed": 0,
            "actions_failed": 0,
            "conflicts_resolved": 0,
        }
    
    def start(self):
        """Start coordination engine."""
        self._running = True
        self._thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self._thread.start()
        logger.info(f"Coordination engine started for satellite {self.sat_idx}")
    
    def stop(self):
        """Stop coordination engine."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _coordination_loop(self):
        """Main coordination loop."""
        while self._running:
            with self._lock:
                now = time.time()
                
                # Process pending actions
                self._process_pending_actions(now)
                
                # Monitor active actions
                self._monitor_active_actions(now)
                
                # Check for new coordination needs
                self._check_coordination_triggers(now)
            
            time.sleep(1.0)  # 1 Hz
    
    def _process_pending_actions(self, now: float):
        """Process actions awaiting approval."""
        to_remove = []
        
        for action_id, action in self.pending_actions.items():
            if action.state == CoordinationState.PROPOSED:
                # Initiate voting via consensus
                if self._initiate_voting(action):
                    action.state = CoordinationState.VOTING
            
            elif action.state == CoordinationState.VOTING:
                # Check if voting complete
                if self._check_voting_complete(action):
                    if action.has_majority():
                        action.state = CoordinationState.APPROVED
                        action.approved_at = now
                        self.stats["actions_approved"] += 1
                        logger.info(f"Action {action_id} approved by majority")
                        
                        if self.on_action_approved:
                            self.on_action_approved(action)
                    else:
                        action.state = CoordinationState.FAILED
                        logger.warning(f"Action {action_id} failed to get majority")
                    
                    to_remove.append(action_id)
            
            elif action.state == CoordinationState.APPROVED:
                # Start execution
                if self._start_execution(action):
                    action.state = CoordinationState.EXECUTING
                    action.started_at = now
                    self.active_actions[action_id] = action
                    to_remove.append(action_id)
        
        for action_id in to_remove:
            self.pending_actions.pop(action_id, None)
    
    def _monitor_active_actions(self, now: float):
        """Monitor execution of active actions."""
        completed = []
        
        for action_id, action in self.active_actions.items():
            # Check if all participants report completion
            if self._check_execution_complete(action):
                action.state = CoordinationState.COMPLETED
                action.completed_at = now
                self.stats["actions_executed"] += 1
                completed.append(action_id)
                
                if self.on_action_completed:
                    self.on_action_completed(action)
            
            # Check for timeout
            elif now - action.started_at > 300.0:  # 5 min timeout
                action.state = CoordinationState.FAILED
                self.stats["actions_failed"] += 1
                completed.append(action_id)
                logger.error(f"Action {action_id} timed out")
        
        for action_id in completed:
            action = self.active_actions.pop(action_id)
            self.completed_actions.append(action)
    
    def _initiate_voting(self, action: CoordinationAction) -> bool:
        """Initiate voting through consensus."""
        if not self.consensus.is_leader() and self.sat_idx != action.initiator:
            # Only leader or initiator can start voting
            return False
        
        # Propose through consensus
        command = {
            "type": "COORDINATION_VOTE",
            "action_id": action.action_id,
            "maneuver_type": action.maneuver_type.value,
            "participants": action.participants,
            "parameters": action.parameters,
            "initiator": action.initiator,
        }
        
        return self.consensus.propose_command(command)
    
    def _check_voting_complete(self, action: CoordinationAction) -> bool:
        """Check if all participants have voted."""
        # In real implementation, votes come via consensus log
        # Here we simulate by checking if we have votes from all participants
        return len(action.votes) >= len(action.participants)
    
    def _start_execution(self, action: CoordinationAction) -> bool:
        """Start executing approved action."""
        # Generate maneuver sequence for each participant
        sequences = self._generate_maneuver_sequences(action)
        
        # Send commands to each participant
        for participant_id, sequence in sequences.items():
            if participant_id == self.sat_idx:
                self.maneuver_sequence = sequence
            else:
                self.send_command(participant_id, {
                    "type": "EXECUTE_MANEUVER",
                    "action_id": action.action_id,
                    "sequence": sequence,
                })
        
        return True
    
    def _check_execution_complete(self, action: CoordinationAction) -> bool:
        """Check if all participants completed maneuver."""
        # In real implementation, check telemetry from each participant
        # For now, simulate based on local sequence completion
        if self.sat_idx in action.participants:
            return len(self.maneuver_sequence) == 0
        return True  # Assume others completed
    
    def _generate_maneuver_sequences(self, action: CoordinationAction) -> Dict[int, List[Dict]]:
        """Generate maneuver sequences for each participant."""
        sequences = {}
        
        if action.maneuver_type == ManeuverType.FORMATION_CHANGE:
            sequences = self._generate_formation_change_sequences(action)
        elif action.maneuver_type == ManeuverType.DEBRIS_AVOIDANCE:
            sequences = self._generate_debris_avoidance_sequences(action)
        elif action.maneuver_type == ManeuverType.COLLISION_AVOIDANCE:
            sequences = self._generate_collision_avoidance_sequences(action)
        elif action.maneuver_type == ManeuverType.DATA_RELAY_SCHEDULE:
            sequences = self._generate_relay_schedule_sequences(action)
        elif action.maneuver_type == ManeuverType.DIFFERENTIAL_DRAG:
            sequences = self._generate_differential_drag_sequences(action)
        
        return sequences
    
    def _generate_formation_change_sequences(self, action: CoordinationAction) -> Dict[int, List[Dict]]:
        """Generate sequences for formation change."""
        target = action.parameters.get("target_formation")
        if not target:
            return {}
        
        sequences = {}
        for participant_id in action.participants:
            # Calculate target position for this satellite
            state = self.get_state(participant_id)
            current_pos = state.adcs.position_eci_km[0] if state else 0
            
            # Simplified: each satellite gets a target anomaly
            sat_index = action.participants.index(participant_id)
            target_anomaly = (sat_index * target.anomaly_spacing_deg) % 360
            
            sequences[participant_id] = [
                {"type": "HOHMANN_TRANSFER", "target_anomaly": target_anomaly, "duration": 100},
                {"type": "STATION_KEEP", "target_anomaly": target_anomaly, "tolerance": 1.0},
            ]
        
        return sequences
    
    def _generate_debris_avoidance_sequences(self, action: CoordinationAction) -> Dict[int, List[Dict]]:
        """Generate sequences for coordinated debris avoidance."""
        debris_pos = action.parameters.get("debris_position", 0)
        avoidance_direction = action.parameters.get("direction", 1)  # +1 or -1
        
        sequences = {}
        for participant_id in action.participants:
            state = self.get_state(participant_id)
            if not state:
                continue
            
            current_pos = state.adcs.position_eci_km[0]
            angular_sep = min(abs(current_pos - debris_pos), 360 - abs(current_pos - debris_pos))
            
            if angular_sep < 10.0:  # Only close satellites maneuver
                sequences[participant_id] = [
                    {"type": "AVOIDANCE_BURN", "delta_v": 0.5 * avoidance_direction, "duration": 10},
                    {"type": "RETURN_TO_SLOT", "duration": 50},
                ]
            else:
                sequences[participant_id] = [
                    {"type": "MONITOR", "duration": 60},
                ]
        
        return sequences
    
    def _generate_collision_avoidance_sequences(self, action: CoordinationAction) -> Dict[int, List[Dict]]:
        """Generate sequences for satellite-satellite collision avoidance."""
        sat_a = action.parameters.get("sat_a")
        sat_b = action.parameters.get("sat_b")
        
        sequences = {}
        for participant_id in action.participants:
            if participant_id == sat_a:
                sequences[participant_id] = [
                    {"type": "AVOIDANCE_BURN", "delta_v": 0.3, "direction": 1, "duration": 10},
                    {"type": "RETURN_TO_SLOT", "duration": 50},
                ]
            elif participant_id == sat_b:
                sequences[participant_id] = [
                    {"type": "AVOIDANCE_BURN", "delta_v": 0.3, "direction": -1, "duration": 10},
                    {"type": "RETURN_TO_SLOT", "duration": 50},
                ]
            else:
                sequences[participant_id] = [
                    {"type": "MONITOR", "duration": 60},
                ]
        
        return sequences
    
    def _generate_relay_schedule_sequences(self, action: CoordinationAction) -> Dict[int, List[Dict]]:
        """Generate data relay schedule."""
        schedule = action.parameters.get("schedule", [])
        
        sequences = {}
        for entry in schedule:
            src = entry.get("src")
            dst = entry.get("dst")
            data_gb = entry.get("data_gb", 0)
            
            if src in action.participants:
                if src not in sequences:
                    sequences[src] = []
                sequences[src].append({
                    "type": "DATA_RELAY",
                    "dst": dst,
                    "data_gb": data_gb,
                    "duration": 10,
                })
        
        return sequences
    
    def _generate_differential_drag_sequences(self, action: CoordinationAction) -> Dict[int, List[Dict]]:
        """Generate differential drag maneuver for formation keeping."""
        sequences = {}
        for i, participant_id in enumerate(action.participants):
            # Alternate drag modulation
            drag_mod = 1.0 if i % 2 == 0 else 0.5
            sequences[participant_id] = [
                {"type": "DRAG_MODULATION", "factor": drag_mod, "duration": 3600},  # 1 hour
                {"type": "RESTORE_DRAG", "duration": 100},
            ]
        
        return sequences
    
    def _check_coordination_triggers(self, now: float):
        """Check for conditions requiring coordination."""
        # Check for debris threats
        state = self.get_state(self.sat_idx)
        if state and state.global_fleet.debris_positions:
            for debris_pos in state.global_fleet.debris_positions:
                angular_sep = min(
                    abs(state.adcs.position_eci_km[0] - debris_pos),
                    360 - abs(state.adcs.position_eci_km[0] - debris_pos)
                )
                if angular_sep < 5.0:  # Close debris
                    self._propose_debris_avoidance(debris_pos)
        
        # Check for formation drift
        if self.current_formation and state:
            self._check_formation_drift(state)
    
    def _propose_debris_avoidance(self, debris_pos: float):
        """Propose coordinated debris avoidance."""
        state = self.get_state(self.sat_idx)
        if not state:
            return
        
        # Find affected satellites
        affected = []
        for i in range(self.num_satellites):
            sat_state = self.get_state(i)
            if sat_state:
                sep = min(
                    abs(sat_state.adcs.position_eci_km[0] - debris_pos),
                    360 - abs(sat_state.adcs.position_eci_km[0] - debris_pos)
                )
                if sep < 10.0:
                    affected.append(i)
        
        if len(affected) > 1:
            action = CoordinationAction(
                action_id=f"debris_avoid_{int(time.time())}_{self.sat_idx}",
                maneuver_type=ManeuverType.DEBRIS_AVOIDANCE,
                participants=affected,
                initiator=self.sat_idx,
                parameters={
                    "debris_position": debris_pos,
                    "direction": 1 if self.sat_idx % 2 == 0 else -1,
                },
                priority=10,
            )
            self.propose_action(action)
    
    def _check_formation_drift(self, state):
        """Check if formation has drifted beyond tolerance."""
        if not self.current_formation:
            return
        
        # Calculate current vs target spacing
        # Simplified check
        pass
    
    def propose_action(self, action: CoordinationAction) -> bool:
        """Propose a new coordination action."""
        with self._lock:
            if action.action_id in self.pending_actions:
                return False
            
            self.pending_actions[action.action_id] = action
            self.stats["actions_proposed"] += 1
            logger.info(f"Sat {self.sat_idx} proposed action {action.action_id}: {action.maneuver_type.value}")
            return True
    
    def propose_formation_change(self, target: FormationTarget) -> str:
        """Propose a formation change maneuver."""
        action = CoordinationAction(
            action_id=f"formation_change_{int(time.time())}_{self.sat_idx}",
            maneuver_type=ManeuverType.FORMATION_CHANGE,
            participants=list(range(self.num_satellites)),
            initiator=self.sat_idx,
            parameters={"target_formation": target},
            priority=5,
        )
        self.propose_action(action)
        return action.action_id
    
    def propose_relay_schedule(self, schedule: List[Dict]) -> str:
        """Propose a data relay schedule."""
        participants = set()
        for entry in schedule:
            participants.add(entry.get("src"))
            participants.add(entry.get("dst"))
        
        action = CoordinationAction(
            action_id=f"relay_schedule_{int(time.time())}_{self.sat_idx}",
            maneuver_type=ManeuverType.DATA_RELAY_SCHEDULE,
            participants=list(participants),
            initiator=self.sat_idx,
            parameters={"schedule": schedule},
            priority=3,
        )
        self.propose_action(action)
        return action.action_id
    
    def vote_on_action(self, action_id: str, vote: bool):
        """Record vote on an action (called when consensus delivers vote)."""
        with self._lock:
            if action_id in self.pending_actions:
                self.pending_actions[action_id].votes[self.sat_idx] = vote
    
    def report_execution_result(self, action_id: str, result: Any):
        """Report execution result for an action."""
        with self._lock:
            if action_id in self.active_actions:
                self.active_actions[action_id].execution_results[self.sat_idx] = result
    
    def get_pending_actions(self) -> List[CoordinationAction]:
        return list(self.pending_actions.values())
    
    def get_active_actions(self) -> List[CoordinationAction]:
        return list(self.active_actions.values())
    
    def get_stats(self) -> Dict:
        return {
            **self.stats,
            "pending_count": len(self.pending_actions),
            "active_count": len(self.active_actions),
            "completed_count": len(self.completed_actions),
        }


class ConstellationCoordinator:
    """
    Coordinates constellation-wide operations through consensus.
    """
    
    def __init__(
        self,
        num_satellites: int,
        consensus: ConstellationConsensus,
        state_provider: callable,
        command_sender: callable,
    ):
        self.num_satellites = num_satellites
        self.consensus = consensus
        self.engines: Dict[int, CoordinationEngine] = {}
        
        for i in range(num_satellites):
            engine = CoordinationEngine(
                sat_idx=i,
                num_satellites=num_satellites,
                consensus=consensus.nodes[i],
                get_state_callback=state_provider,
                send_command_callback=command_sender,
            )
            self.engines[i] = engine
    
    def start_all(self):
        for engine in self.engines.values():
            engine.start()
        self.consensus.start_all()
    
    def stop_all(self):
        for engine in self.engines.values():
            engine.stop()
        self.consensus.stop_all()
    
    def propose_constellation_maneuver(self, maneuver_type: ManeuverType, parameters: Dict) -> str:
        """Propose a constellation-wide maneuver."""
        action = CoordinationAction(
            action_id=f"constellation_{maneuver_type.value}_{int(time.time())}",
            maneuver_type=maneuver_type,
            participants=list(range(self.num_satellites)),
            initiator=0,  # Leader proposes
            parameters=parameters,
            priority=10,
        )
        
        # Propose through consensus leader
        leader = self.consensus.get_leader()
        if leader is not None:
            return self.engines[leader].propose_action(action)
        return ""
    
    def get_all_stats(self) -> Dict[int, Dict]:
        return {sat_idx: engine.get_stats() for sat_idx, engine in self.engines.items()}