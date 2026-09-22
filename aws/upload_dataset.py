"""
aws/upload_dataset.py
=====================
Run this ONCE before your first SageMaker training job.
Uploads your entire spacetech project to S3.

Usage:
    pip install boto3 awscli
    aws configure
    python aws/upload_dataset.py
"""

import boto3
import os
import sys
import zipfile
from pathlib import Path

# ─── CONFIG — must match sagemaker_launcher.py ────────────────────────────
S3_BUCKET  = "your-spacetech-bucket"   # <- CHANGE THIS (must match launcher)
AWS_REGION = "us-east-1"               # <- CHANGE if needed
S3_KEY     = "spacetech-input/spacetech_project.zip"
# ──────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent  # spacetech/ directory
ZIP_PATH     = PROJECT_ROOT / "aws_upload.zip"

# Folders/files to EXCLUDE from upload
EXCLUDE = {
    ".git", "venv", "__pycache__", ".vscode",
    "aws_upload.zip", "spacetech_project.zip",
    "satellite_ppo_tensorboard", "*.pyc",
}

def should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE or part.endswith(".pyc"):
            return True
    return False

def create_zip():
    print(f"Creating zip from {PROJECT_ROOT} ...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in PROJECT_ROOT.rglob("*"):
            rel = f.relative_to(PROJECT_ROOT)
            if should_exclude(rel):
                continue
            if f.is_file():
                zf.write(f, arcname=str(rel))
    size_mb = ZIP_PATH.stat().st_size / 1e6
    print(f"  Created {ZIP_PATH.name}  ({size_mb:.1f} MB)")
    return ZIP_PATH

def upload_to_s3(zip_path: Path):
    s3 = boto3.client("s3", region_name=AWS_REGION)

    # Create bucket if it doesn't exist
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        print(f"Bucket s3://{S3_BUCKET} already exists.")
    except Exception:
        print(f"Creating bucket s3://{S3_BUCKET} ...")
        if AWS_REGION == "us-east-1":
            s3.create_bucket(Bucket=S3_BUCKET)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
            )

    size_mb = zip_path.stat().st_size / 1e6
    print(f"Uploading {size_mb:.1f} MB to s3://{S3_BUCKET}/{S3_KEY} ...")

    def progress(bytes_transferred):
        pct = bytes_transferred / zip_path.stat().st_size * 100
        print(f"\r  {pct:.1f}%", end="", flush=True)

    s3.upload_file(
        str(zip_path), S3_BUCKET, S3_KEY,
        Callback=progress,
    )
    print(f"\nUpload complete: s3://{S3_BUCKET}/{S3_KEY}")

if __name__ == "__main__":
    zip_path = create_zip()
    upload_to_s3(zip_path)
    zip_path.unlink()  # clean up local zip
    print("\nDone! You can now run: python aws/sagemaker_launcher.py")
