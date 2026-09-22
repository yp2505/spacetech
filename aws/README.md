# Training on AWS SageMaker

## Files
| File | Purpose |
|------|---------|
| `upload_dataset.py` | Zip & upload your project to S3 (run once) |
| `sagemaker_launcher.py` | Submit training job to SageMaker |
| `aws_train_entry.py` | Entry-point that runs inside AWS container (don't edit) |

---

## One-Time Setup (do this once ever)

### 1. Install AWS tools locally
```bash
pip install boto3 sagemaker awscli
```

### 2. Get AWS credentials
- Go to [AWS Console](https://console.aws.amazon.com) → IAM → Users → your user → Security Credentials
- Create an Access Key
- Run:
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (e.g. us-east-1), output format (json)
```

### 3. Create SageMaker IAM Role
- AWS Console → IAM → Roles → Create Role
- Trusted entity: **SageMaker**
- Attach policies: `AmazonSageMakerFullAccess` + `AmazonS3FullAccess`
- Name it: `SageMakerExecutionRole`
- Copy the ARN (looks like `arn:aws:iam::123456789012:role/SageMakerExecutionRole`)

### 4. Edit config in both scripts
In `sagemaker_launcher.py` and `upload_dataset.py`:
```python
S3_BUCKET      = "your-actual-bucket-name"   # any unique name
SAGEMAKER_ROLE = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
AWS_REGION     = "us-east-1"
```

---

## Every Training Run

### Step 1 — Upload project to S3 (only need to re-run if code changed)
```bash
cd /path/to/spacetech
python aws/upload_dataset.py
```

### Step 2 — Submit training job
```bash
python aws/sagemaker_launcher.py
```

### Step 3 — Watch logs
```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

### Step 4 — Download trained model
```bash
# Replace JOB_NAME with the printed job name
aws s3 cp s3://your-bucket/spacetech-output/JOB_NAME/output/model.tar.gz ./
tar -xzf model.tar.gz
# You'll get: ppo_swarm_brain.bin, ewc_fisher_swarm.pkl, episodic_memory*.pkl
```

---

## Cost Estimate

| Mode | Instance | GPU | Cost/hr | 200 cycles est. |
|------|----------|-----|---------|-----------------|
| On-demand | ml.g4dn.xlarge | T4 | $0.74 | ~$3-8 |
| **Spot (recommended)** | ml.g4dn.xlarge | T4 | ~$0.22 | **~$1-2** |
| On-demand | ml.g5.xlarge | A10 | $1.41 | ~$6-14 |

> Spot instances are interrupted occasionally but **auto-resume from checkpoint** — your training won't be lost.

---

## Troubleshooting

**"No module named 'simulation'"** — Your zip didn't include all folders. Check `upload_dataset.py` EXCLUDE list.

**"ResourceLimitExceeded"** — You hit the default quota for GPU instances. Request a limit increase in AWS Console → Service Quotas.

**Job stays in "Starting" for >10 min** — Normal for spot instances during high demand. Switch `USE_SPOT = False` for immediate start.
