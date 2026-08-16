"""
fsw/hal/isl_mesh.py
-------------------
ISL Mesh Network Layer with Retransmission, Routing, and Encryption.
Provides reliable multi-hop data relay across the satellite constellation.
"""

import time
import logging
import base64
import hashlib
import struct
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading

logger = logging.getLogger(__name__)


class PacketType(Enum):
    """ISL packet types."""
    DATA_RELAY = 0x01
    TELEMETRY_BEACON = 0x02
    ROUTING_UPDATE = 0x03
    ACK = 0x04
    NACK = 0x05
    HEARTBEAT = 0x06


@dataclass
class ISLPacket:
    """ISL Mesh packet structure."""
    packet_type: PacketType
    src_sat_idx: int
    dst_sat_idx: int
    seq_num: int
    payload: bytes
    hops: int = 0
    max_hops: int = 4
    timestamp: float = field(default_factory=time.time)
    route: List[int] = field(default_factory=list)  # Path taken
    
    def to_bytes(self) -> bytes:
        """Serialize packet to bytes."""
        header = struct.pack(
            ">BBBBI",
            self.packet_type.value,
            self.src_sat_idx & 0xFF,
            self.dst_sat_idx & 0xFF,
            self.hops & 0xFF,
            self.seq_num
        )
        route_bytes = struct.pack(f">{len(self.route)}B", *self.route) if self.route else b""
        route_len = struct.pack(">B", len(self.route))
        return header + route_len + route_bytes + self.payload
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["ISLPacket"]:
        """Deserialize packet from bytes."""
        if len(data) < 8:
            return None
        try:
            pkt_type, src, dst, hops, seq = struct.unpack(">BBBBI", data[:8])
            route_len = data[8]
            route = list(data[9:9+route_len]) if route_len > 0 else []
            payload = data[9+route_len:]
            return cls(
                packet_type=PacketType(pkt_type),
                src_sat_idx=src,
                dst_sat_idx=dst,
                hops=hops,
                seq_num=seq,
                payload=payload,
                route=route
            )
        except Exception:
            return None


class ISLKeyManager:
    """
    Manages encryption keys for ISL links.
    Uses SHA-256 key derivation from mission seed + satellite indices.
    """
    
    def __init__(self, mission_seed: bytes = b"ARTEMIS_SWARM_2025"):
        self.mission_seed = mission_seed
        self._key_cache: Dict[Tuple[int, int], bytes] = {}
        self._lock = threading.RLock()
    
    def derive_key(self, src: int, dst: int) -> bytes:
        """Derive 16-byte key for directed link src->dst."""
        with self._lock:
            key_pair = (src, dst)
            if key_pair not in self._key_cache:
                material = self.mission_seed + struct.pack(">HH", src, dst)
                digest = hashlib.sha256(material).digest()
                self._key_cache[key_pair] = digest[:16]
            return self._key_cache[key_pair]
    
    def encrypt(self, key: bytes, plaintext: bytes, seq: int) -> bytes:
        """Rolling XOR encryption with sequence salt (vectorized via numpy)."""
        if not plaintext:
            return b""
            
        key_len = len(key)
        seq_bytes = struct.pack(">I", seq)
        
        pt_arr = np.frombuffer(plaintext, dtype=np.uint8)
        k_buf = np.frombuffer(key, dtype=np.uint8)
        s_buf = np.frombuffer(seq_bytes, dtype=np.uint8)
        
        # Vectorized key combination and XOR
        idx = np.arange(len(plaintext))
        k_combined = k_buf[(idx + seq) % key_len] ^ s_buf[idx % 4]
        result = pt_arr ^ k_combined
        
        return result.tobytes()
    
    def decrypt(self, key: bytes, ciphertext: bytes, seq: int) -> bytes:
        """Decrypt (XOR is symmetric)."""
        return self.encrypt(key, ciphertext, seq)


@dataclass
class LinkState:
    """State of a single ISL link."""
    neighbor_idx: int
    link_quality: float = 0.0
    last_rx_time: float = 0.0
    last_tx_time: float = 0.0
    rx_seq: int = 0
    tx_seq: int = 0
    consecutive_failures: int = 0
    is_active: bool = False
    avg_latency_ms: float = 0.0


class ISLRoutingTable:
    """
    Distance-vector routing table for mesh networking.
    Each satellite maintains routes to all other satellites.
    """
    
    def __init__(self, sat_idx: int, num_satellites: int):
        self.sat_idx = sat_idx
        self.num_satellites = num_satellites
        # routes[dest] = (next_hop, cost, hops, last_update)
        self.routes: Dict[int, Tuple[int, float, int, float]] = {}
        self._lock = threading.RLock()
        
        # Initialize direct routes
        for i in range(num_satellites):
            if i != sat_idx:
                self.routes[i] = (i, 1.0, 1, time.time())
    
    def get_next_hop(self, dest: int) -> Optional[int]:
        """Get next hop for destination."""
        with self._lock:
            if dest in self.routes:
                return self.routes[dest][0]
            return None
    
    def get_route_cost(self, dest: int) -> float:
        """Get cost to destination."""
        with self._lock:
            if dest in self.routes:
                return self.routes[dest][1]
            return float('inf')
    
    def update_route(self, dest: int, next_hop: int, cost: float, hops: int) -> bool:
        """Update route if better. Returns True if changed."""
        with self._lock:
            if dest == self.sat_idx:
                return False
            
            old_cost = self.routes.get(dest, (None, float('inf'), 0, 0))[1]
            if cost < old_cost or dest not in self.routes:
                self.routes[dest] = (next_hop, cost, hops, time.time())
                return True
            return False
    
    def receive_routing_update(self, sender: int, sender_routes: Dict[int, Tuple[int, float, int]]):
        """Process routing update from neighbor (distance vector)."""
        with self._lock:
            for dest, (_, cost, hops) in sender_routes.items():
                if dest == self.sat_idx:
                    continue
                new_cost = cost + 1.0  # Add link cost
                new_hops = hops + 1
                if new_hops > 4:  # Max hops
                    continue
                self.update_route(dest, sender, new_cost, new_hops)
    
    def get_routing_update(self) -> Dict[int, Tuple[int, float, int]]:
        """Get current routes for broadcasting."""
        with self._lock:
            return {dest: (next_hop, cost, hops) for dest, (next_hop, cost, hops, _) in self.routes.items()}


class ISLMeshNode:
    """
    ISL Mesh Network Node.
    Handles packet transmission, reception, routing, retransmission, and encryption.
    """
    
    def __init__(
        self,
        sat_idx: int,
        num_satellites: int,
        key_manager: ISLKeyManager,
        max_queue_size: int = 100,
        retransmit_timeout_ms: float = 500.0,
        max_retries: int = 3
    ):
        self.sat_idx = sat_idx
        self.num_satellites = num_satellites
        self.key_manager = key_manager
        self.max_queue_size = max_queue_size
        self.retransmit_timeout_ms = retransmit_timeout_ms
        self.max_retries = max_retries
        
        # Routing
        self.routing_table = ISLRoutingTable(sat_idx, num_satellites)
        
        # Link states
        self.links: Dict[int, LinkState] = {}
        for i in range(num_satellites):
            if i != sat_idx:
                self.links[i] = LinkState(neighbor_idx=i)
        
        # Queues
        self.tx_queue: deque = deque(maxlen=max_queue_size)
        self.rx_queue: deque = deque(maxlen=max_queue_size)
        self.pending_acks: Dict[int, Tuple[ISLPacket, float, int]] = {}  # seq -> (packet, sent_time, retries)
        
        # Stats
        self.stats = {
            "tx_packets": 0,
            "rx_packets": 0,
            "tx_bytes": 0,
            "rx_bytes": 0,
            "retransmissions": 0,
            "dropped": 0,
            "decrypt_failures": 0,
            "routing_updates": 0
        }
        
        self._lock = threading.RLock()
        self._seq_counter = 0
    
    def update_link_quality(self, neighbor_idx: int, quality: float, latency_ms: float = 0.0):
        """Update link quality from radio layer."""
        with self._lock:
            if neighbor_idx in self.links:
                link = self.links[neighbor_idx]
                link.link_quality = quality
                link.is_active = quality > 0.1
                link.avg_latency_ms = latency_ms
                if quality < 0.1:
                    link.consecutive_failures += 1
                else:
                    link.consecutive_failures = 0
    
    def send_data(self, dst_sat_idx: int, payload: bytes, packet_type: PacketType = PacketType.DATA_RELAY) -> bool:
        """Queue a packet for transmission."""
        with self._lock:
            if len(self.tx_queue) >= self.max_queue_size:
                self.stats["dropped"] += 1
                return False
            
            self._seq_counter = (self._seq_counter + 1) & 0xFFFFFFFF
            seq = self._seq_counter
            
            packet = ISLPacket(
                packet_type=packet_type,
                src_sat_idx=self.sat_idx,
                dst_sat_idx=dst_sat_idx,
                seq_num=seq,
                payload=payload
            )
            
            # Determine next hop
            next_hop = self.routing_table.get_next_hop(dst_sat_idx)
            if next_hop is None:
                self.stats["dropped"] += 1
                return False
            
            packet.route.append(self.sat_idx)
            self.tx_queue.append((next_hop, packet))
            return True
    
    def send_telemetry_beacon(self, telemetry_data: bytes):
        """Broadcast telemetry to all neighbors."""
        for neighbor_idx in self.links:
            if self.links[neighbor_idx].is_active:
                self.send_data(neighbor_idx, telemetry_data, PacketType.TELEMETRY_BEACON)
    
    def send_routing_update(self):
        """Broadcast routing table to active neighbors only."""
        routes = self.routing_table.get_routing_update()
        payload = self._encode_routes(routes)
        active_neighbors = [nid for nid, link in self.links.items() if link.is_active]
        for neighbor_idx in active_neighbors:
            self.send_data(neighbor_idx, payload, PacketType.ROUTING_UPDATE)
        self.stats["routing_updates"] += 1
    
    def _encode_routes(self, routes: Dict[int, Tuple[int, float, int]]) -> bytes:
        """Encode routing table for transmission."""
        # Simple binary format: count + (dest, next_hop, cost, hops) entries
        data = struct.pack(">B", len(routes))
        for dest, (next_hop, cost, hops) in routes.items():
            data += struct.pack(">BBfB", dest & 0xFF, next_hop & 0xFF, cost, hops & 0xFF)
        return data
    
    def _decode_routes(self, data: bytes) -> Dict[int, Tuple[int, float, int]]:
        """Decode routing table from received data."""
        if len(data) < 1:
            return {}
        count = data[0]
        routes = {}
        offset = 1
        for _ in range(count):
            if offset + 7 > len(data):
                break
            dest, next_hop, cost, hops = struct.unpack(">BBfB", data[offset:offset+7])
            routes[dest] = (next_hop, cost, hops)
            offset += 7
        return routes
    
    def process_tx_queue(self) -> List[Tuple[int, bytes]]:
        """
        Process transmit queue, apply encryption, handle retransmissions.
        Returns list of (neighbor_idx, encrypted_packet_bytes) ready for radio.
        """
# print(f"[DEBUG] Node {self.sat_idx}: process_tx_queue start, queue={len(self.tx_queue)}")
        with self._lock:
            now = time.time()
            ready_packets = []
            
            # Handle retransmissions
            to_retransmit = []
            for seq, (packet, sent_time, retries) in list(self.pending_acks.items()):
                elapsed_ms = (now - sent_time) * 1000
                if elapsed_ms > self.retransmit_timeout_ms:
                    if retries < self.max_retries:
                        to_retransmit.append((seq, packet, retries + 1))
                    else:
                        # Max retries exceeded
                        del self.pending_acks[seq]
                        self.stats["dropped"] += 1
            
            for seq, packet, retries in to_retransmit:
                next_hop = self.routing_table.get_next_hop(packet.dst_sat_idx)
                if next_hop is not None:
                    self.pending_acks[seq] = (packet, now, retries)
                    self.tx_queue.appendleft((next_hop, packet))
                    self.stats["retransmissions"] += 1
            
            # Process new packets
            output = []
            while self.tx_queue:
                next_hop, packet = self.tx_queue.popleft()
# print(f"[DEBUG] Node {self.sat_idx}: processing packet to {next_hop}, type={packet.packet_type}, dst={packet.dst_sat_idx}")
                
                # Encrypt
                key = self.key_manager.derive_key(self.sat_idx, next_hop)
                plaintext = packet.to_bytes()
                ciphertext = self.key_manager.encrypt(key, plaintext, packet.seq_num)
                
                # Add to pending ACKs only for packet types that expect ACKs
                if packet.dst_sat_idx != 0xFF and packet.packet_type == PacketType.DATA_RELAY:
                    self.pending_acks[packet.seq_num] = (packet, now, 0)
                
                self.links[next_hop].last_tx_time = now
                self.links[next_hop].tx_seq = packet.seq_num
                
                output.append((next_hop, ciphertext))
                self.stats["tx_packets"] += 1
                self.stats["tx_bytes"] += len(ciphertext)
            
# print(f"[DEBUG] Node {self.sat_idx}: process_tx_queue end, output={len(output)}")
            return output
    
    def receive_packet(self, from_sat_idx: int, ciphertext: bytes) -> List[ISLPacket]:
        """
        Receive and decrypt packet from radio layer.
        Returns list of processed packets (may include routed packets).
        """
# print(f"[DEBUG] Node {self.sat_idx}: receive_packet from {from_sat_idx}")
        with self._lock:
            key = self.key_manager.derive_key(from_sat_idx, self.sat_idx)
            
            # Try to decrypt - we need the sequence number which is in the packet
            # Brute force recent sequence numbers
            packet = None
            for seq_guess in range(self.links[from_sat_idx].rx_seq - 10, self.links[from_sat_idx].rx_seq + 10):
                if seq_guess < 0:
                    continue
                try:
                    plaintext = self.key_manager.decrypt(key, ciphertext, seq_guess)
                    candidate = ISLPacket.from_bytes(plaintext)
                    if candidate and candidate.src_sat_idx == from_sat_idx:
                        packet = candidate
                        break
                except Exception:
                    continue
            
            if packet is None:
                self.stats["decrypt_failures"] += 1
                self.links[from_sat_idx].consecutive_failures += 1
                return []
            
# print(f"[DEBUG] Node {self.sat_idx}: decrypted packet type={packet.packet_type}, src={packet.src_sat_idx}, dst={packet.dst_sat_idx}, seq={packet.seq_num}")
            
            # Verify sequence
            if packet.seq_num <= self.links[from_sat_idx].rx_seq:
                # Duplicate or old packet
# print(f"[DEBUG] Node {self.sat_idx}: duplicate/old packet, ignoring")
                return []
            
            self.links[from_sat_idx].rx_seq = packet.seq_num
            self.links[from_sat_idx].last_rx_time = time.time()
            self.links[from_sat_idx].consecutive_failures = 0
            
            self.stats["rx_packets"] += 1
            self.stats["rx_bytes"] += len(ciphertext)
            
            processed = []
            
            # Handle different packet types
            if packet.packet_type == PacketType.DATA_RELAY:
                if packet.dst_sat_idx == self.sat_idx:
                    # For us - send ACK
# print(f"[DEBUG] Node {self.sat_idx}: DATA_RELAY for us, sending ACK")
                    self._send_ack(from_sat_idx, packet.seq_num)
                    processed.append(packet)
                elif packet.hops < packet.max_hops:
                    # Forward
# print(f"[DEBUG] Node {self.sat_idx}: DATA_RELAY forwarding, hops={packet.hops}")
                    packet.hops += 1
                    packet.route.append(self.sat_idx)
                    next_hop = self.routing_table.get_next_hop(packet.dst_sat_idx)
                    if next_hop is not None:
                        self.tx_queue.append((next_hop, packet))
            
            elif packet.packet_type == PacketType.TELEMETRY_BEACON:
# print(f"[DEBUG] Node {self.sat_idx}: TELEMETRY_BEACON received")
                processed.append(packet)
            
            elif packet.packet_type == PacketType.ROUTING_UPDATE:
# print(f"[DEBUG] Node {self.sat_idx}: ROUTING_UPDATE received")
                routes = self._decode_routes(packet.payload)
                self.routing_table.receive_routing_update(packet.src_sat_idx, routes)
            
            elif packet.packet_type == PacketType.ACK:
# print(f"[DEBUG] Node {self.sat_idx}: ACK received for seq={packet.seq_num}")
                # Remove from pending
                if packet.seq_num in self.pending_acks:
                    del self.pending_acks[packet.seq_num]
            
# print(f"[DEBUG] Node {self.sat_idx}: receive_packet end, processed={len(processed)}")
            return processed
    
    def _send_ack(self, to_sat_idx: int, seq_num: int):
        """Send ACK for received packet."""
        ack_packet = ISLPacket(
            packet_type=PacketType.ACK,
            src_sat_idx=self.sat_idx,
            dst_sat_idx=to_sat_idx,
            seq_num=seq_num,
            payload=b""
        )
        self.tx_queue.appendleft((to_sat_idx, ack_packet))
    
    def get_received_packets(self) -> List[ISLPacket]:
        """Get and clear received packet queue."""
        with self._lock:
            packets = list(self.rx_queue)
            self.rx_queue.clear()
            return packets
    
    def get_link_states(self) -> Dict[int, LinkState]:
        """Get current link states."""
        with self._lock:
            return {k: LinkState(
                neighbor_idx=v.neighbor_idx,
                link_quality=v.link_quality,
                last_rx_time=v.last_rx_time,
                last_tx_time=v.last_tx_time,
                rx_seq=v.rx_seq,
                tx_seq=v.tx_seq,
                consecutive_failures=v.consecutive_failures,
                is_active=v.is_active,
                avg_latency_ms=v.avg_latency_ms
            ) for k, v in self.links.items()}
    
    def get_stats(self) -> Dict:
        """Get mesh statistics."""
        with self._lock:
            return self.stats.copy()


class ISLMeshNetwork:
    """
    Manages ISL mesh network for entire constellation.
    Creates and coordinates mesh nodes for each satellite.
    """
    
    def __init__(self, num_satellites: int, mission_seed: bytes = b"ARTEMIS_SWARM_2025"):
        self.num_satellites = num_satellites
        self.key_manager = ISLKeyManager(mission_seed)
        self.nodes: Dict[int, ISLMeshNode] = {}
        
        for i in range(num_satellites):
            self.nodes[i] = ISLMeshNode(i, num_satellites, self.key_manager)
    
    def get_node(self, sat_idx: int) -> ISLMeshNode:
        return self.nodes[sat_idx]
    
    def step(self, link_qualities: Dict[Tuple[int, int], float]):
        """
        Simulate one network step.
        link_qualities: {(src, dst): quality} from radio layer
        """
        # Update link qualities
        for (src, dst), quality in link_qualities.items():
            if src in self.nodes:
                self.nodes[src].update_link_quality(dst, quality)
        
        # Process all nodes
        all_transmissions = {}
        for sat_idx, node in self.nodes.items():
            # Send routing updates periodically (only 1/4 of nodes per step, and only if they have active links)
            if sat_idx % 4 == 0:
                active_neighbors = [nid for nid, link in node.links.items() if link.is_active]
                if active_neighbors:
                    node.send_routing_update()
            
            # Process TX queue
            transmissions = node.process_tx_queue()
            for neighbor_idx, ciphertext in transmissions:
                all_transmissions[(sat_idx, neighbor_idx)] = ciphertext
        
        # Deliver packets
        for (src, dst), ciphertext in all_transmissions.items():
            if dst in self.nodes:
                received = self.nodes[dst].receive_packet(src, ciphertext)
                for pkt in received:
                    self.nodes[dst].rx_queue.append(pkt)
    
    def get_all_stats(self) -> Dict[int, Dict]:
        return {idx: node.get_stats() for idx, node in self.nodes.items()}


# Backward compatibility with existing ISLPacketCrypto interface
class ISLPacketCrypto:
    """
    Compatibility wrapper for existing code using ISLPacketCrypto.
    """
    
    def __init__(self, src_sat_idx: int, dst_sat_idx: int, mission_seed: bytes = b"ARTEMIS_SWARM_2025"):
        self.src = src_sat_idx
        self.dst = dst_sat_idx
        self.key_manager = ISLKeyManager(mission_seed)
        self._seq = 0
    
    @staticmethod
    def _derive_key_static(src: int, dst: int, seed: bytes) -> bytes:
        material = seed + struct.pack(">HH", src, dst)
        return hashlib.sha256(material).digest()[:16]
    
    def encrypt(self, plaintext: bytes) -> str:
        self._seq += 1
        key = self._derive_key_static(self.src, self.dst, b"ARTEMIS_SWARM_2025")
        seq_bytes = struct.pack(">I", self._seq)
        key_len = len(key)
        result = bytearray(len(plaintext))
        for i, byte in enumerate(plaintext):
            k = key[(i + self._seq) % key_len] ^ seq_bytes[i % 4]
            result[i] = byte ^ k
        wire = struct.pack(">I", self._seq) + bytes(result)
        return base64.b64encode(wire).decode("ascii")
    
    def decrypt(self, b64_payload: str) -> Optional[bytes]:
        try:
            wire = base64.b64decode(b64_payload)
            if len(wire) < 4:
                return None
            seq = struct.unpack(">I", wire[:4])[0]
            ciphertext = wire[4:]
            key = self._derive_key_static(self.src, self.dst, b"ARTEMIS_SWARM_2025")
            key_len = len(key)
            seq_bytes = struct.pack(">I", seq)
            result = bytearray(len(ciphertext))
            for i, byte in enumerate(ciphertext):
                k = key[(i + seq) % key_len] ^ seq_bytes[i % 4]
                result[i] = byte ^ k
            return bytes(result)
        except Exception:
            return None
    
    @staticmethod
    def pack_data_packet(src_idx: int, data_gb: float, timestamp: float) -> bytes:
        return f"RELAY:{src_idx}:{data_gb:.4f}:{timestamp:.3f}".encode()
    
    @staticmethod
    def unpack_data_packet(raw: bytes) -> Optional[dict]:
        try:
            parts = raw.decode().split(":")
            if parts[0] != "RELAY" or len(parts) < 4:
                return None
            return {
                "src": int(parts[1]),
                "data_gb": float(parts[2]),
                "timestamp": float(parts[3]),
            }
        except Exception:
            return None