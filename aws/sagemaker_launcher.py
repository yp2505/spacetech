"""
aws/sagemaker_launcher.py
=========================
Run Satellite Swarm MAPPO (RL) training on AWS SageMaker.

RL does NOT need a dataset — the agent generates its own experience
by interacting with the satellite simulation environment.

What we upload:
  1. Source code  — SageMaker auto-zips the project folder (no manual step)
  2. (Optional)   — ppo_swarm_brain.bin for continual learning / warm start

USAGE:
------
1.  pip install boto3 sagemaker awscli
2.  aws configure          (enter your Access Key, Secret, Region)
3.  python aws/sagemaker_launcher.py
4.  Watch logs:
      aws logs tail /aws/sagemaker/TrainingJobs --follow
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import os

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — only these 3 lines need editing
# ─────────────────────────────────────────────────────────────────────────────

S3_BUCKET      = "your-spacetech-bucket"   # <- any unique bucket name you want
SAGEMAKER_ROLE = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"  # <- from IAM
AWS_REGION     = "us-east-1"

# Instance type:
#   ml.g4dn.xlarge  -> 1x T4  GPU ($0.74/hr) — good for your model size
#   ml.g5.xlarge    -> 1x A10 GPU ($1.41/hr) — faster
INSTANCE_TYPE = "ml.g4dn.xlarge"

USE_SPOT = True  # ~70% cheaper; auto-resumes from checkpoint if interrupted

# ─── Training hyperparameters (mirrors your mappo_train.py argparse) ────────
HYPERPARAMS = {
    "orbit":           "starlink_leo",
    "cycles":          "200",
    "steps-per-cycle": "5000",
    "mappo":           True,      # flag — enable MAPPO mode
    "fast-eval":       True,      # flag — skip slow baseline eval
}

# ─── Optional: path to existing brain for warm-start / continual learning ───
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXISTING_BRAIN = os.path.join(PROJECT_ROOT, "ppo_swarm_brain.zip")  # or .bin

# ─────────────────────────────────────────────────────────────────────────────
def main():
    boto_sess = boto3.Session(region_name=AWS_REGION)
    sm_sess   = sagemaker.Session(boto_session=boto_sess)
    s3        = boto_sess.client("s3")

    # Ensure S3 bucket exists
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except Exception:
        print(f"Creating S3 bucket: s3://{S3_BUCKET}")
        if AWS_REGION == "us-east-1":
            s3.create_bucket(Bucket=S3_BUCKET)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
            )

    # ── Source code ──────────────────────────────────────────────────────────
    # SageMaker auto-zips and uploads source_dir — no manual upload needed!
    # We point it at the project root so all modules (simulation/, rl_training/)
    # are available inside the container.
    source_dir  = PROJECT_ROOT
    entry_point = "aws/aws_train_entry.py"

    # ── Optional: upload existing brain weights for continual learning ───────
    inputs = {}
    if os.path.exists(EXISTING_BRAIN):
        brain_s3_key = "spacetech-weights/ppo_swarm_brain.zip"
        brain_s3_uri = f"s3://{S3_BUCKET}/{brain_s3_key}"
        print(f"Uploading existing brain weights -> {brain_s3_uri}")
        s3.upload_file(EXISTING_BRAIN, S3_BUCKET, brain_s3_key)
        inputs["weights"] = brain_s3_uri   # mounted at /opt/ml/input/data/weights/
    else:
        print("No existing brain found — will train from scratch.")

    print(f"\nSubmitting RL training job...")
    print(f"  Instance  : {INSTANCE_TYPE}  (Spot={USE_SPOT})")
    print(f"  Source    : {source_dir}  (auto-uploaded by SageMaker)")
    print(f"  Output    : s3://{S3_BUCKET}/spacetech-output/")

    estimator = PyTorch(
        entry_point=entry_point,
        source_dir=source_dir,
        role=SAGEMAKER_ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        use_spot_instances=USE_SPOT,
        max_wait=86400 if USE_SPOT else None,  # 24h spot wait
        max_run=43200,                          # 12h training limit
        hyperparameters=HYPERPARAMS,
        output_path=f"s3://{S3_BUCKET}/spacetech-output/",
        checkpoint_s3_uri=f"s3://{S3_BUCKET}/spacetech-checkpoints/",
        sagemaker_session=sm_sess,
    )

    estimator.fit(inputs=inputs or None, wait=False, logs=False)

    job_name = estimator.latest_training_job.name
    print(f"\nJob submitted: {job_name}")
    print(f"\nWatch live logs:")
    print(f"  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {job_name} --follow")
    print(f"\nDownload trained model after completion:")
    print(f"  aws s3 cp s3://{S3_BUCKET}/spacetech-output/{job_name}/output/model.tar.gz ./")
    print(f"  tar -xzf model.tar.gz")


if __name__ == "__main__":
    main()
