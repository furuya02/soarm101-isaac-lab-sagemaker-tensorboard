# soarm101-isaac-lab-sagemaker-tensorboard

Sample code for training the SO-ARM101 Reach task with Isaac Lab on Amazon SageMaker Training Job + Managed Spot Training, extended with TensorBoard visualization.

Two viewing modes are supported: post-hoc inspection of completed training logs on a local TensorBoard (Part 1), and near-real-time inspection of an in-flight training job's logs streamed via S3 (Part 2).

Companion blog post: [Visualize Isaac Lab reinforcement learning on SageMaker Training Job with TensorBoard](https://dev.classmethod.jp/articles/) (link to be filled in after publication).

> Japanese version: [README.ja.md](README.ja.md)

## Overview

- Base image: `nvcr.io/nvidia/isaac-lab:2.3.2` (Isaac Lab 2.3.2 + Isaac Sim 5.1.x)
- Task: `Isaac-SO-ARM101-Reach-v0` (from [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101), main branch)
- Training: SageMaker Training Job, **ml.g5.2xlarge** (NVIDIA A10G 24 GB) by default, Managed Spot
- Region: `ap-northeast-1`
- Verified cost (one full Reach run, max_iterations=1000): **about 0.31 USD** on-demand, **about 0.13 USD with Managed Spot (58% off)**

## Repository layout

```
.
├── cdk/                    # AWS CDK (TypeScript): S3, ECR, IAM Role
├── scripts/
│   ├── push_to_ecr.sh      # Build & push the training image to ECR
│   ├── download_logs.sh    # Part 1: download completed logs from S3 for local TB
│   └── tb_live.sh          # Part 2: launch local TB pointed at an in-flight job's S3 logs
├── src/
│   ├── train.py            # SageMaker entrypoint (SIGTERM forwarding, ckpt resume, TB symlink)
│   └── entrypoint.sh
├── Dockerfile              # Inherits NGC isaac-lab:2.3.2
├── submit.py               # SageMaker Estimator launcher
└── README.md / README.ja.md
```

## Prerequisites

- AWS account with SageMaker / S3 / ECR access
- AWS CLI v2 configured for `ap-northeast-1`
- Docker (with the `linux/amd64` build platform available)
- Node.js 20.x and AWS CDK v2 (`pnpm add -g aws-cdk`)
- An NVIDIA NGC account and API key (`docker login nvcr.io`)
- Python 3.11 with the SageMaker Python SDK (`pip install sagemaker`)

## Setup

### 1. Clone

```bash
git clone https://github.com/furuya02/soarm101-isaac-lab-sagemaker-tensorboard.git
cd soarm101-isaac-lab-sagemaker-tensorboard
```

### 2. Deploy AWS resources with CDK

```bash
cd cdk
pnpm install

export AWS_REGION=ap-northeast-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

cdk bootstrap aws://${ACCOUNT_ID}/${AWS_REGION}
cdk deploy \
  -c account_id=${ACCOUNT_ID} \
  -c region=${AWS_REGION}
```

The stack creates:

- S3 bucket: `soarm101-isaac-lab-sagemaker-tensorboard-<ACCOUNT_ID>`
- ECR repository: `soarm101-isaac-lab-sagemaker-tensorboard`
- IAM role: `soarm101-isaac-lab-sagemaker-tensorboard-sagemaker-execution-role`

To override the bucket suffix:

```bash
cdk deploy \
  -c account_id=${ACCOUNT_ID} \
  -c bucket_suffix=20260503
```

### 3. Build & push the training image to ECR

```bash
cd ..

# Log in to NGC (NGC API key required)
docker login nvcr.io

./scripts/push_to_ecr.sh
```

The first push transfers ~15 GB and takes 30-60 minutes depending on your uplink. SageMaker pulls the image from the same region, so there is no inter-region data transfer charge.

### 4. Submit a SageMaker Training Job

```bash
export SAGEMAKER_ROLE_ARN=$(aws iam get-role \
  --role-name soarm101-isaac-lab-sagemaker-tensorboard-sagemaker-execution-role \
  --query 'Role.Arn' --output text)
export ECR_IMAGE_URI=${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/soarm101-isaac-lab-sagemaker-tensorboard:latest
export S3_BUCKET=soarm101-isaac-lab-sagemaker-tensorboard-${ACCOUNT_ID}

# On-demand single run for sanity check
USE_SPOT=false MAX_RUN_HOURS=1 python submit.py

# Managed Spot run
USE_SPOT=true MAX_RUN_HOURS=1 MAX_WAIT_HOURS=2 python submit.py
```

### 5. Retrieve the trained model

```bash
JOB_NAME=<job-name-from-submit.py-output>
aws s3 cp s3://${S3_BUCKET}/output/${JOB_NAME}/output/model.tar.gz .
tar xzf model.tar.gz
# rsl_rl/<task>/<run>/model_<iter>.pt
```

### 6. View training progress with TensorBoard

rsl_rl is launched with `--logger tensorboard` and emits episode mean reward, PPO losses, exploration noise, etc. as TensorBoard event files. In this repository, `src/train.py` symlinks the TensorBoard log directory into `/opt/ml/checkpoints/tensorboard/`, so events are **streamed to S3 in near real time during training** via the SageMaker `checkpoint_s3_uri` Continuous upload mode.

Prerequisites:

```bash
pip install tensorboard tensorflow-io
```

#### Part 1: Inspect completed training logs locally

Download the `model.tar.gz` of a finished job from S3, extract it, and start TensorBoard locally.

```bash
./scripts/download_logs.sh ${S3_BUCKET} ${JOB_NAME}
tensorboard --logdir ./logs/${JOB_NAME}/rsl_rl/
# Open http://localhost:6006 in your browser
```

#### Part 2: Watch an in-flight training job in near real time

Right after submitting a job, launch a local TensorBoard pointed directly at the S3 checkpoint dir.

```bash
./scripts/tb_live.sh ${S3_BUCKET} ${JOB_NAME}
# Open http://localhost:6006 in your browser
```

Notes:

- TensorBoard continuously scans S3, and **leaving it running for long periods can run up S3 GET API costs** (see [tensorboard issue #6564](https://github.com/tensorflow/tensorboard/issues/6564)). Close the browser and stop the TensorBoard process (Ctrl-C) when you are done.
- Sync interval depends on the SageMaker `checkpoint_s3_uri` Continuous upload mode and may have several tens of seconds to a few minutes of lag.

### 7. (Optional) Render a trained policy to mp4

A second SageMaker job runs `play.py --headless --video` to render the
trained policy without any GUI. The Dockerfile installs `ffmpeg` because
Isaac Lab's `RecordVideo` wrapper needs it to write mp4 from frame buffers.

```bash
MODE=play \
MODEL_S3_URI=s3://${S3_BUCKET}/output/${JOB_NAME}/output/model.tar.gz \
USE_SPOT=true MAX_RUN_HOURS=1 MAX_WAIT_HOURS=2 \
python submit.py

# After the play job completes, download the mp4:
PLAY_JOB=<job-name-from-the-play-submit>
aws s3 cp s3://${S3_BUCKET}/output/${PLAY_JOB}/output/model.tar.gz play_output.tar.gz
tar xzf play_output.tar.gz   # videos/*.mp4
```

A typical play job runs in 5-10 minutes and costs less than 0.10 USD on Spot.

## Measured cost (ap-northeast-1, May 2026)

| Item | On-demand | Managed Spot |
|---|---|---|
| Instance | ml.g5.2xlarge (A10G 24GB) | ml.g5.2xlarge (A10G 24GB) |
| Training time (real) | 727 s (12 min) | 733 s (12 min) |
| Billable time | 727 s | 309 s |
| **Cost per run** | **0.306 USD** | **0.130 USD (58% off)** |

Other recurring costs:

| Resource | Cost |
|---|---|
| ECR storage (~40 GB image, uncompressed) | ~ 4.0 USD / month while the image is retained |
| S3 (artifacts + checkpoints, < 100 MB) | < 0.10 USD / month |

`ml.g6.2xlarge` (NVIDIA L4) at ~ 1.81 USD/hour on-demand, ~ 0.54 USD/hour Spot is also a valid choice once its quota is granted (see Caveats below).

## Cleanup

```bash
cd cdk
cdk destroy
# Empty the S3 bucket and delete ECR images manually if you want to remove them too.
```

## Caveats

- **Managed Spot requires checkpoints.** Without a checkpoint implementation, `max_wait` is capped at 1 hour. `src/train.py` resumes from `/opt/ml/checkpoints/model_*.pt` automatically.
- **`max_run` is mandatory.** Always set this to prevent runaway training charges.
- **Region pinning.** Keep ECR, S3 and SageMaker all in `ap-northeast-1` to avoid inter-region data transfer fees.
- **Image size.** The `nvcr.io/nvidia/isaac-lab:2.3.2` base image is about 40 GB uncompressed. The first ECR push takes 30-60 minutes on a home connection, and SageMaker pulls the image at job start (5-10 minutes overhead per job).
- **Service Quotas.** New AWS accounts often have **0 quota** for SageMaker GPU instances, even on the latest generation (g6, L4). `ml.g5.2xlarge` (used as default here) typically has a quota of 1 out of the box. To switch to `ml.g6.2xlarge` request the quota first under "Service Quotas" -> "Amazon SageMaker" -> "ml.g6.2xlarge for training job usage" and "for spot training job usage".
- **`isaac_so_arm101` repository layout.** The `v1.2.0` tag uses an older `source/SO_100/` layout. Use the `main` branch (or pin a recent commit via the `ISAAC_SO_ARM101_REF` Docker build arg) so that `pip install -e .` succeeds against the standard `src/isaac_so_arm101/` layout.
- **Mac Docker disk limit.** macOS Docker Desktop defaults the virtual disk size to 64 GB, which is too small once the 40 GB Isaac Lab image is added. Either run `docker builder prune -a -f` to recover space, or raise the limit to 200 GB in Docker Desktop -> Settings -> Resources -> Advanced.

## License

This sample code is released under the MIT License.

`isaac_so_arm101` is BSD-3-Clause. NVIDIA Isaac Sim and Isaac Lab follow the [NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/install-guide/latest/common/NVIDIA_Omniverse_License_Agreement.html).
