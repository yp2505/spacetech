import sys
import os
import shutil
import rl_training.train as train

def main() -> None:
    # This intentionally moves existing artefacts, so it is only run explicitly.
    for f in ["ppo_swarm_brain.zip", "ewc_fisher_swarm.pkl", "episodic_memory.pkl"]:
        if os.path.exists(f):
            shutil.move(f, f + ".phase_b.bak")
            print(f"Moved {f} to backup.")

    train.CYCLES = 1
    train.STEPS_PER_CYCLE = 2048
    print("Running sanity check for Phases C-F...")
    train.main()
    print("Sanity check PASSED!")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
