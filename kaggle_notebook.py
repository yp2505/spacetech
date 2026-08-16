# ============================================================
#  KAGGLE NOTEBOOK — SATELLITE SWARM AI TRAINING
#  Copy each cell below into a separate Kaggle code cell.
# ============================================================

# ════════════════════════════════════════════════════════════
#  CELL 1 — Setup + Train  (run this first, let it finish)
# ════════════════════════════════════════════════════════════
import subprocess, shutil, os, sys

# ── 1a. Verify GPU ────────────────────────────────────────
import torch
print("🚀 GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "❌ NO GPU FOUND")

# ── 1b. Copy your dataset into the working directory ──────
SRC = "/kaggle/input/spacetechs"          # change to your actual dataset name if different
DST = "/kaggle/working/spacetech"
if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)
print(f"✅ Files copied → {DST}")
os.chdir(DST)
sys.path.insert(0, DST)

# ── 1c. Install dependencies ──────────────────────────────
subprocess.run([
    "pip", "install", "-q",
    "stable-baselines3>=2.3.0",
    "gymnasium==1.0.0",
    "astropy",
], check=True)
print("✅ Dependencies installed!")

# ── 1d. Confirm old-brain checkpoint is present ──────────
CHECKPOINT = "checkpoints/phase_b_archive/ppo_swarm_brain.zip"
if os.path.exists(CHECKPOINT):
    size_mb = os.path.getsize(CHECKPOINT) / 1e6
    print(f"✅ Old brain found ({size_mb:.1f} MB) — Transfer learning ENABLED (AI builds on old knowledge!)")
else:
    print("⚠️  Old brain NOT found — Will train from scratch")

# ── 1e. Launch training ───────────────────────────────────
# MAPPO mode: 10 satellites share one brain, 200 cycles x 5000 steps = 1M total steps
os.system(
    "python -u rl_training/train.py "
    "--orbit starlink_leo "
    "--mappo "
    "--cycles 200 "
    "--steps-per-cycle 5000"
)
print("\n🎉 Training complete!")


# ════════════════════════════════════════════════════════════
#  CELL 2 — Save & Download  (run AFTER Cell 1 finishes)
# ════════════════════════════════════════════════════════════
import subprocess, os, glob

os.chdir("/kaggle/working/spacetech")

OUTPUT_ZIP = "/kaggle/working/trained_satellite_brain.zip"
if os.path.exists(OUTPUT_ZIP):
    os.remove(OUTPUT_ZIP)

patterns = [
    "ppo_swarm_brain.zip",
    "ewc_fisher_swarm.pkl",
    "episodic_memory*.pkl",
    "outputs/reward_history.csv",
]
files_to_save = []
for p in patterns:
    files_to_save.extend(glob.glob(p))

if files_to_save:
    subprocess.run(["zip", OUTPUT_ZIP] + files_to_save, check=True)
    size_mb = os.path.getsize(OUTPUT_ZIP) / 1e6
    print(f"✅ Saved! Download → {OUTPUT_ZIP}  ({size_mb:.1f} MB)")
    print("\nFiles inside:")
    for f in files_to_save:
        sz = os.path.getsize(f) / 1e6 if os.path.exists(f) else 0
        print(f"  • {f}  ({sz:.2f} MB)")
else:
    print("❌ No model files found — did Cell 1 finish?")
