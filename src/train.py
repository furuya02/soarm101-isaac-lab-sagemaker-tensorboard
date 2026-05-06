"""SageMaker entrypoint: forward SIGTERM and copy logs to /opt/ml/model."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

CKPT_DIR = Path("/opt/ml/checkpoints")
MODEL_DIR = Path("/opt/ml/model")
ISAACLAB_DIR = Path("/workspace/isaaclab")
LOG_DIR = ISAACLAB_DIR / "logs" / "rsl_rl"


def main() -> int:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(CKPT_DIR.glob("model_*.pt"))
    resume_args = ["--resume", "--checkpoint", str(ckpts[-1])] if ckpts else []

    cmd = [
        str(ISAACLAB_DIR / "isaaclab.sh"), "-p",
        "/opt/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/train.py",
        "--task", os.environ.get("TASK_NAME", "Isaac-SO-ARM101-Reach-v0"),
        "--headless",
        "--num_envs", os.environ.get("NUM_ENVS", "64"),
        "--max_iterations", os.environ.get("MAX_ITERATIONS", "1000"),
        "--logger", "tensorboard",
        "--experiment_name", os.environ.get("EXPERIMENT_NAME", "so_arm101_reach"),
        *resume_args,
    ]
    proc = subprocess.Popen(cmd, cwd=str(ISAACLAB_DIR))

    # Managed Spot grace: forward SIGTERM so rsl_rl flushes a checkpoint.
    signal.signal(signal.SIGTERM, lambda *_: proc.send_signal(signal.SIGTERM))

    return_code = proc.wait()

    if LOG_DIR.exists():
        shutil.copytree(LOG_DIR, MODEL_DIR / "rsl_rl", dirs_exist_ok=True)
    for ckpt in CKPT_DIR.glob("model_*.pt"):
        shutil.copy2(ckpt, MODEL_DIR / ckpt.name)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
