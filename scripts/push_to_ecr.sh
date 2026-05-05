#!/usr/bin/env bash
# Build and push the training image to ECR. Usage: ./scripts/push_to_ecr.sh [tag]
set -euo pipefail

REGION="${AWS_REGION:-ap-northeast-1}"
PROJECT_NAME="soarm101-isaac-lab-sagemaker-tensorboard"
TAG="${1:-latest}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_HOST="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${ECR_HOST}/${PROJECT_NAME}:${TAG}"

aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_HOST}"
# --provenance=false --sbom=false: BuildKit がデフォルトで生成する
# attestation / Image Index を抑止する。これらは ECR 上で untagged
# manifest として残るため、UNTAGGED 削除ライフサイクルで本体まで
# 巻き込まれて latest がリンク切れになるのを防ぐ。
docker buildx build \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  -t "${IMAGE_URI}" \
  --push .
echo "${IMAGE_URI}"
