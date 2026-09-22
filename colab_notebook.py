# ============================================================
#  GOOGLE COLAB — SATELLITE SWARM MAPPO TRAINING
#  Copy each cell below into a separate Colab code cell.
#  Runtime → Change runtime type → T4 GPU  (before running!)
# ============================================================


# ════════════════════════════════════════════════════════════
#  CELL 1 — Mount Drive & Setup
# ════════════════════════════════════════════════════════════
import subprocess, shutil, os, sys, zipfile

# 1a. Verify GPU
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "❌ NO GPU — change runtime!"
print("🚀 GPU:", gpu)
assert torch.cuda.is_available(), "Set Runtime → T4 GPU and restart!"

# 1b. Mount Google Drive (your zip lives here after upload)
from google.colab import drive
drive.mount("/content/drive")

# 1c. Find and unzip the project
#     Upload spacetech_project.zip to your Google Drive first,
#     then update this path if needed:
ZIP_PATH = "/content/drive/MyDrive/spacetech_project.zip"

if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(
        f"Could not find {ZIP_PATH}\n"
        "→ Upload spacetech_project.zip to your Google Drive root folder."
    )

DST = "/content/spacetech"
if os.path.exists(DST):
    shutil.rmtree(DST)
os.makedirs(DST, exist_ok=True)

print(f"📦 Extracting {os.path.basename(ZIP_PATH)} ...")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(DST)

# Flatten if zip created a nested spacetech/ folder
inner = os.path.join(DST, "spacetech")
if os.path.isdir(inner):
    for item in os.listdir(inner):
        shutil.move(os.path.join(inner, item), os.path.join(DST, item))
    os.rmdir(inner)

os.chdir(DST)
sys.path.insert(0, DST)
print(f"✅ Project ready at {DST}")
print(f"   Files: {[f for f in os.listdir(DST) if not f.startswith('__')][:12]}")


# ════════════════════════════════════════════════════════════
#  CELL 2 — Install Dependencies
# ════════════════════════════════════════════════════════════
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "stable-baselines3>=2.3.0,<3.0",
    "gymnasium>=0.29,<1.1",
    "astropy",
], check=True)
print("✅ Dependencies installed!")


# ════════════════════════════════════════════════════════════
#  CELL 3 — Load Previous Weights (Continual Learning)
#  The trainer now auto-detects old weights AND fixes the
#  curriculum so rewards don't go negative.
# ════════════════════════════════════════════════════════════
import glob

# Check what brain files are already present in the project
brain_files = (
    glob.glob(os.path.join(DST, "ppo_swarm_brain*.zip"))
    + glob.glob(os.path.join(DST, "ppo_swarm_brain*.bin"))
)

if brain_files:
    for f in brain_files:
        sz = os.path.getsize(f) / 1e6
        print(f"🧠 Found brain: {os.path.basename(f)}  ({sz:.1f} MB)")
    print("✅ Continual learning ENABLED — training will resume from these weights")
    print("   Curriculum will start at Phase 4 (skipping easy phases to avoid reward collapse)")
else:
    # Also check the phase_b_archive fallback
    pb = "checkpoints/phase_b_archive/ppo_swarm_brain.bin"
    if os.path.exists(pb):
        sz = os.path.getsize(pb) / 1e6
        print(f"🧠 Phase B baseline found ({sz:.1f} MB) — Transfer learning ENABLED")
    else:
        print("⚠️  No brain found — training from scratch (Phase 1 curriculum)")


# ════════════════════════════════════════════════════════════
#  CELL 4 — Train!  (this takes ~30-90 min on T4)
# ════════════════════════════════════════════════════════════
os.chdir(DST)

# MAPPO mode: 10 satellites share one brain
# 200 cycles × 5000 steps = 1M total steps
# --fast-eval: skips slow 15-min baseline eval, starts training immediately
result = subprocess.run([
    sys.executable, "-u",
    "rl_training/mappo_train.py",
    "--orbit",          "starlink_leo",
    "--mappo",
    "--cycles",         "200",
    "--steps-per-cycle", "5000",
    "--fast-eval",
], check=False)

if result.returncode == 0:
    print("\n🎉 Training complete!")
else:
    print(f"\n❌ Training exited with code {result.returncode}")
    print("Check the output above for the error.")


# ════════════════════════════════════════════════════════════
#  CELL 5 — Save to Google Drive
#  Run AFTER Cell 4 finishes.
# ════════════════════════════════════════════════════════════
import glob, zipfile, os
from datetime import datetime

os.chdir(DST)

# Collect all model files
patterns = [
    "ppo_swarm_brain*.zip",
    "ppo_swarm_brain*.bin",
    "ewc_fisher_swarm.pkl",
    "episodic_memory*.pkl",
    "outputs/reward_history.csv",
]
files_to_save = []
for p in patterns:
    files_to_save.extend(glob.glob(p))

if not files_to_save:
    print("❌ No model files found — did Cell 4 finish successfully?")
else:
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
    DRIVE_DIR  = "/content/drive/MyDrive/spacetech_trained"
    OUTPUT_ZIP = f"{DRIVE_DIR}/satellite_brain_{timestamp}.zip"
    os.makedirs(DRIVE_DIR, exist_ok=True)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_save:
            zf.write(f, arcname=os.path.basename(f))

    size_mb = os.path.getsize(OUTPUT_ZIP) / 1e6
    print(f"✅ Saved to Google Drive: {OUTPUT_ZIP}  ({size_mb:.1f} MB)")
    print("\nFiles saved:")
    for f in files_to_save:
        sz = os.path.getsize(f) / 1e6 if os.path.exists(f) else 0
        print(f"  • {os.path.basename(f)}  ({sz:.2f} MB)")
    print(f"\n💡 Next run: download {os.path.basename(OUTPUT_ZIP)}, put it")
    print(f"   inside spacetech_project.zip as ppo_swarm_brain.zip before uploading.")
