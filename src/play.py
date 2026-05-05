"""SageMaker entrypoint: download model.tar.gz, render policy to mp4, copy videos to /opt/ml/model."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import boto3

ISAACLAB_DIR = Path("/workspace/isaaclab")
MODEL_DIR = Path("/opt/ml/model")
WORK_DIR = Path("/opt/ml/code/play_work")


def main() -> int:
    parsed = urlparse(os.environ["MODEL_S3_URI"])
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tarball = WORK_DIR / "model.tar.gz"
    boto3.client("s3").download_file(parsed.netloc, parsed.path.lstrip("/"), str(tarball))
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(WORK_DIR)

    ckpt = sorted(WORK_DIR.rglob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[-1]

    cmd = [
        str(ISAACLAB_DIR / "isaaclab.sh"), "-p",
        "/opt/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py",
        "--task", os.environ.get("TASK_NAME", "Isaac-SO-ARM101-Reach-Play-v0"),
        "--headless",
        "--video",
        "--video_length", os.environ.get("VIDEO_LENGTH", "200"),
        "--num_envs", os.environ.get("NUM_ENVS", "4"),
        "--checkpoint", str(ckpt),
    ]
    proc = subprocess.run(cmd, cwd=str(ISAACLAB_DIR))

    # play.py writes videos under <ckpt_dir>/videos (== inside WORK_DIR)
    for videos in WORK_DIR.rglob("videos"):
        if videos.is_dir() and any(videos.iterdir()):
            shutil.copytree(videos, MODEL_DIR / "videos", dirs_exist_ok=True)
            break

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
