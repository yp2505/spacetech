# Universal Satellite RL Environment

The project trains a PPO policy for formation keeping across LEO, MEO and GEO
satellite configurations. The environment implements all requested phases:

- A: N-satellite slot keeping.
- B: configuration-driven orbit, vehicle and constellation parameters.
- C: Earth-observation, communications and science objectives.
- D: validated local TLE ingestion, line-of-sight downlink and configurable
  station blackout windows.
- E: wheel, thruster and sensor faults with a five-step safe-hold recovery.
- F: thermal state/safe mode and Walker-Delta multi-plane topology.

## Setup and checks

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -v
python -m rl_training.train --list-orbits
python -m rl_training.train --orbit starlink_leo
```

For Google Colab, create and upload `spacetech_fsw_colab_ready.zip` with
`python -m scripts.export_colab`; it excludes checkpoints by default. Then run:

```bash
!python -m rl_training.colab_train --orbit starlink_leo --cycles 20 --steps-per-cycle 2048
```

The launcher checks the regression suite first and installs Gymnasium and SB3
only if the Colab runtime does not already provide them.

Training writes model, Fisher matrix, memory, checkpoints and CSV exports to
the configured output paths. Use `--eval-only` to evaluate a saved model.

## TLE and blackout configuration

Create a `SatelliteConfig` from a two- or three-line NORAD TLE with
`SatelliteConfig.from_tle(...)`; checksum and satellite-number consistency are
validated locally. TLE propagation is represented by the environment's
circular-orbit model derived from the TLE mean motion. Specify blackout windows
as `(start_step, end_step, station_name)` tuples; use `None` for every station.
