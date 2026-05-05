#!/usr/bin/env bash
# Part 2: launch local TensorBoard pointed at S3 checkpoint dir for live monitoring.
# Usage: ./scripts/tb_live.sh <bucket> <job-name> [region]
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <bucket> <job-name> [region]"
  echo "  Example: $0 soarm101-isaac-lab-sagemaker-tensorboard-123456789012 soarm101-reach-1234567890 ap-northeast-1"
  exit 1
fi

BUCKET="$1"
JOB_NAME="$2"
export AWS_REGION="${3:-ap-northeast-1}"

S3_LOGDIR="s3://${BUCKET}/checkpoints/${JOB_NAME}/tensorboard/"

echo "Starting TensorBoard against ${S3_LOGDIR}"
echo "Open http://localhost:6006 in your browser."
echo "WARNING: keep session short to avoid S3 GET API cost spikes"
echo "         (see https://github.com/tensorflow/tensorboard/issues/6564)."
echo ""

tensorboard --logdir "${S3_LOGDIR}"
