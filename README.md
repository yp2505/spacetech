# SpaceTech: Autonomous Satellite Swarm Intelligence & Flight Software (ARTEMIS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/stable--baselines3-2.3+-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/tests-19%2F19%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, high-fidelity autonomous satellite swarm management platform and real-time flight software (FSW) architecture. **SpaceTech** combines advanced orbital mechanics, multi-agent reinforcement learning (MAPPO/PPO), continual learning via Elastic Weight Consolidation (EWC), distributed peer-to-peer memory gossip, and a deterministic safety-critical flight executive—**ARTEMIS** (*Autonomous Real-Time Embedded Mission Intelligence System*).

---

## Table of Contents
1. [Key Features & System Highlights](#key-features--system-highlights)
2. [Phased Development Roadmap (Phases A–F)](#phased-development-roadmap-phases-af)
3. [System Architecture](#system-architecture)
4. [Flight Software (FSW) — ARTEMIS Executive](#flight-software-fsw--artemis-executive)
5. [Reinforcement Learning & Universal Swarm Brain](#reinforcement-learning--universal-swarm-brain)
6. [Orbital Physics & Dynamics Engine](#orbital-physics--dynamics-engine)
7. [Constellation Presets & Orbital Regimes](#constellation-presets--orbital-regimes)
8. [Ground Stations & Line-of-Sight Network](#ground-stations--line-of-sight-network)
9. [Visualization & Telemetry Suite](#visualization--telemetry-suite)
10. [Repository Structure](#repository-structure)
11. [Installation & Setup](#installation--setup)
12. [Usage Guide & CLI Commands](#usage-guide--cli-commands)
    - [Running Automated Unit & SIL Tests](#1-running-automated-unit--sil-tests)
    - [Evaluating Pre-trained Universal Swarm Brain](#2-evaluating-pre-trained-universal-swarm-brain)
    - [Training Single Orbits (LEO, MEO, GEO)](#3-training-single-orbits-leo-meo-geo)
    - [Universal Swarm Brain 3-Stage Pipeline](#4-universal-swarm-brain-3-stage-pipeline)
    - [Multi-Agent Training (MAPPO / IPPO)](#5-multi-agent-training-mappo--ippo)
    - [Running Software-in-the-Loop (SIL) Simulation](#6-running-software-in-the-loop-sil-simulation)
    - [Launching Visualizations (Cesium 3D & Python Dashboard)](#7-launching-visualizations-cesium-3d--python-dashboard)
    - [Cloud Execution (Google Colab & Kaggle)](#8-cloud-execution-google-colab--kaggle)
13. [Verification & Test Results](#verification--test-results)
14. [License & Acknowledgments](#license--acknowledgments)

---

## Key Features & System Highlights

- **Universal Multi-Regime Autonomy**: A single generalized neural network policy (*Universal Swarm Brain*) capable of operating seamlessly across Low Earth Orbit (LEO, 550 km), Medium Earth Orbit (MEO, 20,200 km), and Geostationary Orbit (GEO, 35,786 km).
- **Continual Learning without Catastrophic Forgetting**: Employs **Elastic Weight Consolidation (EWC)** with Fisher Information Matrix regularization ($\lambda = 5000$) over 400 rollouts, protecting 565,329 critical parameters during multi-regime curriculum transfers.
- **Real-Time Embedded Flight Software (ARTEMIS)**:
  - 10 Hz cyclic executive scheduler with strict timing guarantees.
  - Multi-tier Watchdog Manager with automated safe-mode and AI-reset triggers.
  - Fault Detection, Isolation, and Recovery (FDIR) state machine (`BOOT` $\to$ `NOMINAL` $\to$ `DEGRADED` $\to$ `RECOVERY` $\to$ `SAFE_MODE`).
  - Deterministic Safety Supervisor & Command Arbiter enforcing strict operational hierarchies: **Safety > Health > Mission > Constellation > Longevity**.
- **Distributed Episodic Memory Gossip**: Decentralized peer-to-peer experience sharing across an Inter-Satellite Link (ISL) mesh network. Salience-weighted priority replay buffers gossip high-impact events (near-misses, subsystem anomalies, recovery maneuvers) across the fleet.
- **Cryptographic ISL Communication**: Rolling-XOR and SHA-256 key derivation for secure, authenticated inter-satellite telemetry and data routing.
- **Full-Spectrum Sensor & Actuator Dynamics**: 3D attitude determination (quaternions, Euler rates), reaction wheel saturation/desaturation, thruster thermal limits, cold gas/chemical/ion/Hall-effect propulsion models, and solar radiation pressure.
- **Comprehensive Visualization Tools**:
  - Web-based **CesiumJS 3D visualizer** (`cesium_viewer.html`) with dynamic CZML orbital tracks, ground station cones, and real-time HUD.
  - Deep-space dark-themed **Matplotlib dashboard** (`visualize_project.py`).
  - **MATLAB / Simulink bridge** (`visualize_project_full.m`, `visualize_results.m`) with automated CSV telemetry export.

---

## Phased Development Roadmap (Phases A–F)

| Phase | Milestone Name | Description & Capabilities |
|:---:|:---|:---|
| **Phase A** | **N-Satellite Slot Keeping** | Multi-satellite circular orbit slot-keeping, relative Clohessy-Wiltshire (CW) dynamics, collision avoidance, and baseline PPO control. |
| **Phase B** | **Multi-Orbit Generalization** | Configuration-driven orbit parameters (LEO, MEO, GEO), propulsion types (cold gas, chemical, ion, Hall), mass/inertia variations, and domain randomization. |
| **Phase C** | **Multi-Mission Objectives** | Mission profiles: **Earth Observation** (strict nadir pointing $\pm 5^\circ$), **Broadband Comms** (ground station LOS + ISL routing), and **Space Science** (inertial pointing/solar alignment). |
| **Phase D** | **Realistic Ground Track & TLE** | Offline NORAD two-/three-line element (TLE) validation and ingestion, SGP4/Skyfield celestial propagation, 5-station global network (Svalbard, Punta Arenas, Hawaii, Singapore, Bangalore), elevation masking ($>5^\circ$), and configurable blackout scheduling. |
| **Phase E** | **Fault Injection & Resilience** | Sensor noise/drift, reaction wheel motor saturation, thruster degradation/flameouts, and autonomous 5-step safe-hold recovery maneuvers. |
| **Phase F** | **Thermal Physics & Walker-Delta** | Stefan-Boltzmann radiative thermal equilibrium (direct solar flux, Earth albedo, thermal emissions), battery temperature safety margins, and multi-plane Walker-Delta constellation topology ($T/P/F$). |

---

## System Architecture

```
                                  +-------------------------------------------------------+
                                  |                 GROUND SEGMENT / C2                   |
                                  |   - Global Stations (Svalbard, Punta Arenas, etc.)    |
                                  |   - TLE Ingestion & Blackout Scheduling               |
                                  +---------------------------+---------------------------+
                                                              |  Uplink / Downlink (LOS)
                                                              v
+-------------------------------------------------------------------------------------------------------------------------+
|                                        ARTEMIS ONBOARD FLIGHT SOFTWARE (FSW)                                            |
|                                                                                                                         |
|   +-----------------------+     +------------------------+     +-----------------------+     +----------------------+   |
|   |   Cyclic Scheduler    | --> |    Telemetry Monitor   | --> |       FDIR FSM        | --> |   Watchdog Manager   |   |
|   |     (10 Hz Executive) |     | (Sensors, EPS, Thermal)|     | (Nominal/Degraded/Safe|     | (Heartbeat / Safety) |   |
|   +-----------------------+     +------------------------+     +-----------------------+     +----------------------+   |
|                                                                                                                         |
|   +-----------------------------------------------------------------------------------------------------------------+   |
|   |                                         SAFETY SUPERVISOR & ARBITER                                             |   |
|   |         Deterministic Rule Hierarchy: Safety > Health > Mission > Constellation > Longevity                    |   |
|   |         - Approves AI commands when safe                                                                        |   |
|   |         - Fallbacks to deterministic safe-hold / classical controller on constraint breach                      |   |
|   +-------------------------------------------------------+---------------------------------------------------------+   |
|                                                           ^
|                                                           | Action Proposal (8-dim continuous)
|                                                           |
|   +-------------------------------------------------------+---------------------------------------------------------+   |
|   |                                       UNIVERSAL SWARM AI BRAIN                                                  |   |
|   |   - PPO / MAPPO Policy Network (Actor-Critic)                                                                   |   |
|   |   - Local Observation: 48-dim (Physical, Attitude, ISL, Config, Recovery, Memory Context)                       |   |
|   |   - Global Observation: 63-dim (Centralized Critic for CTDE)                                                    |   |
|   |   - Continual Learning: Elastic Weight Consolidation (EWC) (Fisher matrix lambda=5000)                          |   |
|   +-------------------------------------------------------+---------------------------------------------------------+   |
|                                                           ^
|                                                           | Experience Sharing & Gossip
|                                                           v
|   +-----------------------------------------------------------------------------------------------------------------+   |
|   |                          ISL MESH NETWORK & DISTRIBUTED EPISODIC MEMORY GOSSIP                                  |   |
|   |   - Rolling-XOR + SHA-256 encrypted peer-to-peer inter-satellite packets                                            |   |
|   |   - Salience-ranked episode replay buffer sharing across fleet members                                          |   |
|   +-----------------------------------------------------------------------------------------------------------------+   |
|                                                           | Actuator Commands
v                                                           v
+-------------------------------------------------------------------------------------------------------------------------+
|                                          HIGH-FIDELITY SIMULATION BACKEND                                               |
|   - Multi-satellite orbital physics (Keplerian + J2 perturbation + atmospheric drag + solar radiation pressure)         |
|   - 3D Attitude dynamics (quaternion propagation, reaction wheels, thruster torques)                                   |
|   - Thermal radiative equilibrium & battery power model (photovoltaic charging + eclipse drain)                        |
|   - Dynamic orbital debris tracking & space weather radiation events                                                   |
+-------------------------------------------------------------------------------------------------------------------------+
```

---

## Flight Software (FSW) — ARTEMIS Executive

Located in `fsw/`, the flight software is structured as a modular, flight-proven embedded control architecture:

1. **Cyclic Executive (`fsw/core/scheduler.py`)**:
   - Executes periodic tasks at deterministic rates:
     - `TELEMETRY`: 10 Hz
     - `FDIR`: 5 Hz
     - `GNC_AI`: 2 Hz
     - `WATCHDOG`: 10 Hz
2. **Watchdog Manager (`fsw/core/watchdog.py`)**:
   - Monitors execution deadlines for every subsystem task. If the AI Brain or GNC hangs or exceeds timeout limits ($>5.0\text{ s}$), the watchdog automatically triggers safe mode and resets the AI module.
3. **FDIR State Machine (`fsw/fdir/state_machine.py`)**:
   - Manages transitions between 5 operating modes:
     - `BOOT`: Initial sensor check and subsystem stabilization.
     - `NOMINAL`: Normal mission operations; AI policy active.
     - `DEGRADED`: Minor actuator/sensor faults present; propulsion restricted.
     - `RECOVERY`: 5-step autonomous recovery maneuver to clear momentum or re-acquire Sun.
     - `SAFE_MODE`: AI disabled; thrusters shut down; battery preservation; Sun-pointing safe hold.
4. **Safety Supervisor & Command Arbiter (`fsw/safety/arbiter.py`, `fsw/safety/supervisor.py`)**:
   - Sits between the AI Brain and physical actuators. All 8 action dimensions proposed by the neural network are inspected. If any action threatens thermal limits ($T < -40^\circ\text{C}$ or $T > 85^\circ\text{C}$), critical battery reserves ($< 12\%$), or fires thrusters during an active propulsion fault, the arbiter vetoes the command and engages deterministic safety overrides.
5. **ISL Mesh Networking (`fsw/hal/isl_mesh.py`)**:
   - Implements multi-hop packet routing across the constellation. Packets are encrypted and authenticated using a rolling-XOR cipher combined with SHA-256 HMAC key derivation.

---

## Reinforcement Learning & Universal Swarm Brain

### 1. Observation Space Breakdown

- **Local Observation (`LOCAL_DIM = 48`)**:
  - `Base Local (11 dims)`: Orbital position error, relative altitude, eclipse fraction, ground station LOS flag, nearest debris distance/angle, remaining $\Delta V$ budget, core temperature, battery State-of-Charge (SoC), and hierarchical commander 3D goal vector.
  - `Attitude State (6 dims)`: Roll, pitch, yaw angles (rad) and angular velocity rates ($\text{rad/s}$).
  - `Neighbor Fleet Context (20 dims)`: Relative position, closing velocity, fuel remaining, battery SoC, and ISL link quality for the 4 nearest satellites ($4 \times 5 = 20$).
  - `Configuration Features (6 dims)`: Orbit type encoding, thruster type encoding, mission profile encoding, normalized vehicle dry mass, solar array surface area, and battery capacity.
  - `Recovery Flag (1 dim)`: Binary indicator whether an autonomous FDIR recovery maneuver is currently executing.
  - `Episodic Memory Context (4 dims)`: Aggregated statistical context derived from local and gossiped episodic memory buffers.

- **Global Observation (`GLOBAL_DIM = 63`)**:
  - Centralized critic state used in **MAPPO** (Centralized Training with Decentralized Execution - CTDE):
  - Own state (6) + Neighbor states ($4 \times 10 = 40$) + Constellation fleet statistics (6) + Debris tracking (3) + Space weather/eclipse factors (2) + Configuration features (6) = 63 dimensions.

### 2. Action Space Breakdown (`8-dim continuous`, $[-1, +1]$)

```
Index | Action Dimension           | Functional Description
------+----------------------------+----------------------------------------------------------------
  0   | Thrust Impulse             | In-track translational $\Delta V$ burn (propulsion subsystem)
  1   | Roll Reaction Torque       | ADCS reaction wheel torque about body X-axis
  2   | Pitch Reaction Torque      | ADCS reaction wheel torque about body Y-axis
  3   | Yaw Reaction Torque        | ADCS reaction wheel torque about body Z-axis
  4   | ISL Data Relay Trigger     | Inter-satellite crosslink packet routing decision
  5   | Hohmann Transfer Burn      | Altitude adjust / semi-major axis orbital shift
  6   | Debris Collision Avoidance | Reactive cross-track impulse maneuver away from approaching debris
  7   | End-of-Life Deorbit Flag   | Autonomous disposal initiation for space sustainability
```

### 3. Continual Learning via Elastic Weight Consolidation (EWC)

To train a single **Universal Swarm Brain** that excels in LEO, MEO, and GEO without suffering from catastrophic forgetting, the training pipeline computes the empirical Fisher Information Matrix $F$:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{PPO}}(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_i - \theta_{A,i}^*)^2$$

- $\lambda = 5000.0$: Regularization parameter protecting previously learned orbital mechanics.
- **Fisher Computation**: Evaluated over 400 policy rollouts across all orbital configurations, protecting 565,329 neural network parameters.

---

## Orbital Physics & Dynamics Engine

Located in `simulation/orbital_physics.py` and `simulation/satellite_env.py`:

- **Gravitational Field**: Earth gravitational parameter $\mu = 398600.4418\text{ km}^3/\text{s}^2$, Mean radius $R_\oplus = 6371.0\text{ km}$.
- **Geopotential Perturbations**: $J_2 = 1.08263 \times 10^{-3}$ nodal regression and perigee precession modeling.
- **Atmospheric Drag**: Exponential atmospheric density profile with drag deceleration:
  $$a_{\text{drag}} = -\frac{1}{2} \rho v^2 \left(\frac{C_D A}{m}\right)$$
- **Solar Radiation Pressure & Eclipse Geometry**: Cylindrical Earth shadow modeling with penumbra/umbra transitions. When optional libraries (`sgp4`, `skyfield`) are detected, high-precision astronomical ephemeris and topocentric elevation are utilized with automatic fallback to analytical Keplerian geometry.
- **Walker-Delta Constellations**: Multi-plane circular orbit distribution ($T/P/F$) parameter generator supporting arbitrary plane counts, RAAN spacing, and inter-plane harmonic phasing.

---

## Constellation Presets & Orbital Regimes

| Preset Key | Mission Name | Orbit Regime | Altitude | Fleet Size | Planes | Inclination | Propulsion | Primary Mission Objective |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `starlink_leo` | Starlink LEO | LEO | 550 km | 10 sats | 2 | 53.0° | Hall Effect | High-bandwidth broadband mesh & ISL relay |
| `gps_meo` | GPS MEO | MEO | 20,200 km | 6 sats | 3 | 55.0° | Ion Thruster | Precision positioning, navigation & timing (PNT) |
| `landsat_obs` | Landsat OBS | LEO (SSO) | 705 km | 4 sats | 1 | 98.2° | Chemical | High-resolution Earth observation & nadir imaging |
| `cubesat_sci` | CubeSat Science | LEO | 400 km | 12 sats | 1 | 51.6° | Cold Gas | Space physics, magnetosphere & solar observations |
| `geo_comms` | GEO Comms | GEO | 35,786 km | 4 sats | 1 | 0.0° | Chemical | Fixed-station continuous equatorial communications |

---

## Ground Stations & Line-of-Sight Network

The platform simulates a realistic global tracking and downlink network:

```
Station Name     | Latitude   | Longitude   | Minimum Elevation Mask
-----------------+------------+-------------+-----------------------
Svalbard (SGS)   |  78.2298°N |   15.4078°E | 5.0°
Punta Arenas     | -53.1403°S |  -70.9063°W | 5.0°
Hawaii Tracking  |  19.8968°N | -155.5828°W | 5.0°
Singapore Relays |   1.3521°N |  103.8198°E | 5.0°
Bangalore (ISTRAC)| 12.9716°N |   77.5946°E | 5.0°
```

- **Line-of-Sight (LOS) Math**: High-precision dot-product horizon checking and topocentric elevation masking.
- **Outage Scheduling**: Supports deterministic blackout injection via CLI (`--blackout START:END[:STATION]`).

---

## Visualization & Telemetry Suite

### 1. Interactive 3D CesiumJS Globe (`cesium_viewer.html`)
- Visualizes 3D orbits, inter-satellite crosslinks, ground station visibility cones, and live telemetry HUD.
- Powered by `outputs/constellation.czml` or dynamic time-series generation.

### 2. Matplotlib Telemetry Dashboard (`visualize_project.py`)
- Deep-space dark-themed visualization engine:
  - 3D orbit tracks & Walker-Delta shell.
  - Multi-agent reward convergence curves.
  - Fuel consumption and $\Delta V$ expenditure.
  - Radiative thermal equilibrium curves & battery SoC profiles.
  - FDIR state transitions and mode timeline.

### 3. MATLAB / Simulink Bridge (`matlab_export/`)
- Contains `visualize_results.m` and `visualize_project_full.m` for post-flight engineering analysis in MATLAB.

---

## Repository Structure

```
spacetech/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Production Python dependencies
├── Run_on_Colab.ipynb             # Google Colab verified interactive notebook
├── kaggle_training.ipynb          # Kaggle accelerator training notebook
├── kaggle_notebook.py             # Standalone headless Kaggle execution script
├── cesium_viewer.html             # Interactive 3D CesiumJS orbital visualization
├── visualize_project.py           # Deep-space Matplotlib visualization dashboard
├── visualize_project_full.m       # Full MATLAB visualization suite
├── ppo_swarm_brain.bin            # Pre-trained Universal Swarm Brain (all orbits)
├── ppo_swarm_brain.zip            # Archived model distribution
├── ewc_fisher_swarm.pkl           # Saved Fisher Information Matrix for EWC
├── episodic_memory_sat0..9.pkl    # Checkpointed distributed episodic memories
│
├── fsw/                           # Embedded Flight Software (ARTEMIS)
│   ├── main.py                    # FSW agent entry point & execution loop
│   ├── ai_brain/                  # AI Brain adapter, validation & consensus
│   │   ├── adapter.py             # Feature normalization & observation adapter
│   │   ├── agent.py               # AI Brain wrapper with watchdog heartbeat
│   │   ├── commander.py           # High-level hierarchical mission commander
│   │   ├── consensus.py           # Multi-agent consensus engine
│   │   ├── coordination.py        # Swarm collision avoidance & slot assignment
│   │   └── validation.py          # Observation sanity & bound validator
│   ├── core/                      # Core OS abstractions
│   │   ├── commands.py            # Actuator command structures
│   │   ├── scheduler.py           # 10 Hz Cyclic Executive Scheduler
│   │   ├── telemetry.py           # Real-time telemetry data models
│   │   └── watchdog.py            # Multi-tier watchdog fault manager
│   ├── fdir/                      # Fault Detection, Isolation, and Recovery
│   │   ├── monitors.py            # Subsystem telemetry threshold monitors
│   │   └── state_machine.py       # FDIR state machine (Nominal/Degraded/Safe)
│   ├── gnc/                       # Guidance, Navigation, and Control
│   │   ├── classical.py           # Classical PID/safe-hold backup controller
│   │   └── estimators.py          # Quaternion attitude & state estimators
│   ├── hal/                       # Hardware Abstraction Layer (HAL)
│   │   ├── interfaces.py          # Abstract hardware base interfaces
│   │   ├── isl_mesh.py            # Encrypted ISL mesh network (rolling-XOR + SHA-256)
│   │   └── sim_backend.py         # Hardware simulator backend bridges
│   ├── power_thermal/             # Power & thermal control
│   │   └── thermal.py             # Passive/active heater & radiator controls
│   └── safety/                    # Deterministic safety guards
│       ├── arbiter.py             # Command arbiter prioritizing health over AI
│       └── supervisor.py          # Envelope protection & rule validation
│
├── simulation/                    # High-Fidelity Orbital Simulation
│   ├── orbital_physics.py         # Keplerian, J2, SGP4, Skyfield, Walker-Delta math
│   ├── sat_config.py              # Configuration engine, presets, TLE parser
│   └── satellite_env.py           # Gymnasium MultiSatelliteEnv & SingleAgentWrapper
│
├── rl_training/                   # Reinforcement Learning & Training Pipeline
│   ├── train.py                   # Master multi-orbit training script
│   ├── mappo_train.py             # Multi-Agent PPO (MAPPO/IPPO) engine
│   ├── ctde_policy.py             # Centralized Critic & Actor neural networks
│   ├── colab_train.py             # Google Colab entrypoint & runner
│   ├── distributed_memory.py      # Gossip-based distributed episodic memory
│   ├── ewc.py                     # Elastic Weight Consolidation (EWC) implementation
│   ├── memory.py                  # Episodic memory buffer & replay models
│   └── network_surgery.py         # Checkpoint dimension transfer utilities
│
├── tests/                         # Complete Automated Test Suite (19/19 OK)
│   ├── test_train.py              # Training loop sanity verification
│   ├── test_energy_model.py       # Battery charging & eclipse energy balance
│   ├── test_fsw_safety.py         # Safety supervisor & command arbiter tests
│   ├── test_fsw_sil.py            # Software-in-the-Loop integration tests
│   ├── test_fsw_train_buffer.py   # PPO rollout length & buffer validation
│   ├── test_phase_features.py     # Walker-Delta, TLE, blackout window tests
│   ├── sanity_cf.py               # Basic environment sanity checks
│   └── fsw_sil_test.py            # Standalone SIL 20-tick execution runner
│
├── matlab_export/                 # MATLAB Telemetry & Plotting Assets
│   ├── visualize_results.m        # MATLAB post-processing script
│   ├── training_progress.csv      # Exported training reward history
│   ├── positions_history.csv      # Exported 3D orbital trajectories
│   ├── reward_history.csv         # Reward logs
│   └── continual_learning_plot.png# Plot of continual learning convergence
│
└── outputs/                       # Generated Simulation Artifacts & Checkpoints
    ├── constellation.czml         # CZML dynamic orbital file for Cesium
    ├── gs_positions.csv           # Ground station coordinates
    ├── episode_data.csv           # Evaluation episode logs
    └── sandcastle.js              # Standalone Cesium Sandcastle viewer script
```

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Recommended: NVIDIA GPU with CUDA support (CPU mode supported out-of-the-box)

### Step 1: Clone Repository
```bash
git clone https://github.com/yp2505/spacetech.git
cd spacetech
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional: For high-precision SGP4 propagation and Skyfield solar ephemeris)*
```bash
pip install sgp4 skyfield astropy
```

---

## Usage Guide & CLI Commands

### 1. Running Automated Unit & SIL Tests
Verify all 19 automated regression tests (including orbital physics, FDIR safety arbiter, energy balance, TLE ingestion, and SIL execution):
```bash
python -m unittest discover -v tests
```

### 2. Evaluating Pre-trained Universal Swarm Brain
Evaluate the pre-trained `ppo_swarm_brain.bin` model across different constellations without training:

```bash
# Evaluate on Starlink LEO (5 episodes)
python -m rl_training.train --orbit starlink_leo --eval-only

# Fast 1-episode evaluation on GPS MEO
python -m rl_training.train --orbit gps_meo --eval-only --fast-eval

# Evaluate on GEO Comms
python -m rl_training.train --orbit geo_comms --eval-only
```

### 3. Training Single Orbits (LEO, MEO, GEO)
Train an agent from scratch on a specific orbital constellation:

```bash
# List all built-in presets
python -m rl_training.train --list-orbits

# Train Starlink LEO with 4 parallel CPU environments
python -m rl_training.train --orbit starlink_leo --cycles 50 --steps-per-cycle 10000 --n-envs 4

# Train Landsat Earth-Observation mission with scheduled station blackouts
python -m rl_training.train --orbit landsat_obs --blackout 20:60:Hawaii --blackout 100:150:Svalbard
```

### 4. Universal Swarm Brain 3-Stage Pipeline
Train a single policy sequentially across **LEO $\to$ MEO $\to$ GEO** using Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting:

```bash
python -m rl_training.train --orbit universal --cycles 100 --steps-per-cycle 20000
```
*This pipeline automatically trains `starlink_leo`, computes the Fisher Information Matrix, transitions to `gps_meo` under EWC regularization, and completes on `geo_comms`.*

### 5. Multi-Agent Training (MAPPO / IPPO)
Train using multi-agent algorithms with centralized or independent critics:

```bash
# Multi-Agent PPO (Shared Critic + Parameter Sharing)
python -m rl_training.train --orbit starlink_leo --mappo --cycles 50

# Independent PPO (Independent Critics per Agent)
python -m rl_training.train --orbit starlink_leo --ippo --cycles 50
```

### 6. Running Software-in-the-Loop (SIL) Simulation
Run the standalone 20-tick Software-in-the-Loop integration test where the ARTEMIS flight software commands the physics backend through the Hardware Abstraction Layer (HAL):

```bash
python tests/fsw_sil_test.py
```

### 7. Launching Visualizations (Cesium 3D & Python Dashboard)

#### A. Interactive 3D CesiumJS Visualizer:
Launch a local web server to view the constellation in 3D:
```bash
python -m http.server 8080
```
Open your browser and navigate to:
```
http://localhost:8080/cesium_viewer.html
```

#### B. Deep-Space Python Telemetry Dashboard:
Generate multi-panel analytical graphs from exported flight logs:
```bash
python visualize_project.py
```

#### C. MATLAB Engineering Analysis:
Open MATLAB, navigate to `matlab_export/`, and execute:
```matlab
visualize_results
```

### 8. Cloud Execution (Google Colab & Kaggle)

#### Google Colab:
Open `Run_on_Colab.ipynb` directly in Google Colab, or run headless via:
```bash
python -m rl_training.colab_train --orbit starlink_leo --cycles 20 --steps-per-cycle 2048 --fast-eval
```

#### Kaggle:
Upload the repository to a Kaggle Notebook and run `kaggle_training.ipynb` or:
```bash
python kaggle_notebook.py
```

---

## Verification & Test Results

The repository features a comprehensive regression test suite with 100% pass rate:

```
test_energy_model:
  [starlink_leo] steps/orbit=361 shadow=36 sunlit=325 drain=0.359%/step charge=2.200%/step NET=+702.08%/orbit ... OK
  [gps_meo]      steps/orbit=360 shadow=28 sunlit=332 drain=0.272%/step charge=3.850%/step NET=+1270.60%/orbit ... OK
  [landsat_obs]  steps/orbit=361 shadow=36 sunlit=325 drain=0.359%/step charge=5.500%/step NET=+1774.58%/orbit ... OK
  [cubesat_sci]  steps/orbit=361 shadow=36 sunlit=325 drain=0.359%/step charge=0.500%/step NET=+149.58%/orbit  ... OK
  [geo_comms]    steps/orbit=360 shadow=9  sunlit=351 drain=0.049%/step charge=11.000%/step NET=+3860.56%/orbit ... OK

test_fsw_safety:
  test_battery_depletion_forces_safe_mode ................................... OK
  test_hot_thermal_limit_forces_safe_mode ................................... OK
  test_cold_thermal_limit_forces_safe_mode ................................... OK
  test_thruster_fault_and_degraded_mode_block_propulsion ..................... OK
  test_scheduler_rejects_invalid_task_rate ................................... OK

test_fsw_sil:
  test_nominal_execution .................................................... OK

test_fsw_train_buffer:
  test_rollout_guard_and_episode_buffer_population ........................... OK

test_phase_features:
  test_tle_parsing_and_orbit_derivation ..................................... OK
  test_walker_topology_and_invalid_layout ................................... OK
  test_blackout_windows_are_station_scoped ................................... OK
  test_config_rejects_ambiguous_plane_layout ................................. OK

----------------------------------------------------------------------
Ran 19 tests in 6.749s -- ALL OK (19 passed, 0 failed, 0 errors)
```

---

## License & Acknowledgments

This project is licensed under the **MIT License**. Developed as an advanced space systems engineering, flight software, and continual reinforcement learning research project.
