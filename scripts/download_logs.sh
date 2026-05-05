#!/usr/bin/env bash
# Part 1: download finished training logs from S3 and extract for local TensorBoard.
# Usage: ./scripts/download_logs.sh <bucket> <job-name> [output-dir]
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <bucket> <job-name> [output-dir]"
  echo "  Example: $0 soarm101-isaac-lab-sagemaker-tensorboard-123456789012 soarm101-reach-1234567890"
  exit 1
fi

BUCKET="$1"
JOB_NAME="$2"
OUT_DIR="${3:-./logs/${JOB_NAME}}"

mkdir -p "${OUT_DIR}"
aws s3 cp "s3://${BUCKET}/output/${JOB_NAME}/output/model.tar.gz" "${OUT_DIR}/model.tar.gz"
tar -xzf "${OUT_DIR}/model.tar.gz" -C "${OUT_DIR}"

echo ""
echo "Logs extracted to ${OUT_DIR}/rsl_rl/"
echo "Run: tensorboard --logdir ${OUT_DIR}/rsl_rl/"
