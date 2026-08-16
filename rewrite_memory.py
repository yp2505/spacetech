import re

with open("rl_training/memory.py", "r") as f:
    content = f.read()

# Replace EpisodeRecord
old_record = """@dataclass
class EpisodeRecord:
    \"\"\"One remembered episode.\"\"\"
    episode_id:       int
    total_reward:     float
    collisions:       int
    fuel_outs:        int
    steps:            int
    was_eclipse:      bool   = False
    was_weather:      bool   = False
    was_fault:        bool   = False   # Phase E: episode had hardware fault injection
    salience:         float  = field(default=0.0, compare=False)
    start_conditions: dict   = field(default_factory=dict)"""

new_record = """@dataclass
class EpisodeRecord:
    \"\"\"One remembered episode.\"\"\"
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
    timestamp_step:      int = 0"""

content = content.replace(old_record, new_record)

with open("rl_training/memory.py", "w") as f:
    f.write(content)
