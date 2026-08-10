"""Google Colab entrypoint for verified SpaceTech PPO training.

Examples:
    !python colab_train.py --orbit starlink_leo
    !python colab_train.py --orbit gps_meo --cycles 20 --steps-per-cycle 2048
    !python colab_train.py --eval-only --orbit geo_comms
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys


def ensure_dependencies() -> None:
    """Install only packages missing from the active Colab runtime."""
    required = {"gymnasium": "gymnasium>=0.29,<1.1", "stable_baselines3": "stable-baselines3>=2.3,<3.0"}
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab launcher for satellite PPO training")
    parser.add_argument("--orbit", default="starlink_leo")
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--steps-per-cycle", type=int, default=5_000)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-tests", action="store_true", help="Skip fast regression tests before training")
    parser.add_argument("--tle-file")
    parser.add_argument("--blackout", action="append", default=[])
    args = parser.parse_args()

    if args.cycles < 1 or args.steps_per_cycle < 2048:
        parser.error("--cycles must be positive and --steps-per-cycle must be at least 2048 (PPO rollout size).")
    ensure_dependencies()

    if not args.skip_tests:
        subprocess.check_call([sys.executable, "-m", "unittest", "discover", "-v"])

    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import rl_training.train as train

    train.CYCLES = args.cycles
    train.STEPS_PER_CYCLE = args.steps_per_cycle
    train_args = ["train.py", "--orbit", args.orbit]
    if args.eval_only:
        train_args.append("--eval-only")
    if args.tle_file:
        train_args.extend(["--tle-file", args.tle_file])
    for blackout in args.blackout:
        train_args.extend(["--blackout", blackout])
    sys.argv = train_args
    train.main()


if __name__ == "__main__":
    main()
