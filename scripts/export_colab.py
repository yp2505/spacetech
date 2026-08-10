"""
export_colab.py
---------------
Packages the entire Spacetech Universal FSW + RL Training codebase 
into a clean ZIP file for Google Colab upload.
"""

import argparse
import os
import zipfile

# Files/Folders to include
INCLUDE_FILES = [
    "fsw/main.py",
    "fsw/ai_brain/agent.py",
    "fsw/ai_brain/validation.py",
    "fsw/ai_brain/adapter.py",
    "fsw/gnc/classical.py",
    "fsw/gnc/estimators.py",
    "fsw/core/watchdog.py",
    "fsw/core/telemetry.py",
    "fsw/core/commands.py",
    "fsw/core/scheduler.py",
    "fsw/fdir/monitors.py",
    "fsw/fdir/state_machine.py",
    "fsw/safety/supervisor.py",
    "fsw/safety/arbiter.py",
    "fsw/power_thermal/thermal.py",
    "fsw/hal/interfaces.py",
    "fsw/hal/sim_backend.py",
    
    "simulation/sat_config.py",
    "simulation/orbital_physics.py",
    "simulation/satellite_env.py",
    "simulation/__init__.py",
    
    "rl_training/network_surgery.py",
    "rl_training/train.py",
    "rl_training/memory.py",
    "rl_training/ewc.py",
    "rl_training/ctde_policy.py",
    "rl_training/__init__.py",
    "rl_training/colab_train.py",
    
    "tests/test_phase_features.py",
    "tests/test_train.py",
    "tests/fsw_sil_test.py",
    "tests/test_fsw_sil.py",
    "tests/__init__.py",
    "tests/sanity_cf.py",
    
    "Run_on_Colab.ipynb",
    "requirements.txt",
    "scripts/export_episode.py",
    "scripts/export_colab.py",
    "visualize_constellation.m",
    "ppo_swarm_brain.zip",
    "ewc_fisher_swarm.pkl",
    "episodic_memory.pkl"
]

OUTPUT_ZIP = "spacetech_fsw_colab_ready.zip"

def main():
    parser = argparse.ArgumentParser(description="Package the source-only Colab archive.")
    parser.add_argument("--include-model", action="store_true", help="Include the current PPO model if present.")
    args = parser.parse_args()
    print(f"Packaging codebase into {OUTPUT_ZIP}...")
    include = list(INCLUDE_FILES)
    if args.include_model:
        include.append("ppo_swarm_brain.zip")
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in include:
            if os.path.isdir(item):
                # Walk directory
                for root, _, files in os.walk(item):
                    # Skip __pycache__
                    if "__pycache__" in root:
                        continue
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.endswith('.pyc'):
                            continue
                        zipf.write(file_path, arcname=file_path)
                        print(f"  Added {file_path}")
            elif os.path.exists(item):
                zipf.write(item, arcname=item)
                print(f"  Added {item}")
            else:
                print(f"  WARNING: {item} not found, skipping.")

    print(f"\nSuccessfully created {OUTPUT_ZIP}!")
    size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
    size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
    if args.for_kaggle:
        print("\n[Kaggle] In your Kaggle notebook, run this BEFORE training:")
        print("  import shutil")
        print("  shutil.copy('ppo_swarm_brain.bin', 'ppo_swarm_brain.zip')")
        print("  # This restores the weights so train.py can load them!")

if __name__ == "__main__":
    main()
