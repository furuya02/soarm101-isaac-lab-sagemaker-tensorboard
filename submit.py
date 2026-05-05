"""Submit a SageMaker Training Job (MODE=train|play)."""

from __future__ import annotations

import os
import time

from sagemaker.estimator import Estimator


def main() -> None:
    role = os.environ["SAGEMAKER_ROLE_ARN"]
    image_uri = os.environ["ECR_IMAGE_URI"]
    bucket = os.environ["S3_BUCKET"]
    mode = os.environ.get("MODE", "train").lower()
    use_spot = os.environ.get("USE_SPOT", "true").lower() in ("1", "true", "yes")

    if mode == "train":
        env = {
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "MODE": "train",
            "TASK_NAME": "Isaac-SO-ARM101-Reach-v0",
            "NUM_ENVS": os.environ.get("NUM_ENVS", "64"),
            "MAX_ITERATIONS": os.environ.get("MAX_ITERATIONS", "1000"),
            "EXPERIMENT_NAME": "so_arm101_reach",
        }
        max_run_h = int(os.environ.get("MAX_RUN_HOURS", "1"))
        max_wait_h = int(os.environ.get("MAX_WAIT_HOURS", "2"))
        job_name = f"soarm101-reach-{int(time.time())}"
    else:
        env = {
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "MODE": "play",
            "TASK_NAME": "Isaac-SO-ARM101-Reach-Play-v0",
            "MODEL_S3_URI": os.environ["MODEL_S3_URI"],
            "NUM_ENVS": os.environ.get("NUM_ENVS", "4"),
            "VIDEO_LENGTH": os.environ.get("VIDEO_LENGTH", "200"),
        }
        max_run_h = int(os.environ.get("MAX_RUN_HOURS", "1"))
        max_wait_h = int(os.environ.get("MAX_WAIT_HOURS", "2"))
        job_name = f"soarm101-reach-play-{int(time.time())}"

    kwargs: dict = dict(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=os.environ.get("INSTANCE_TYPE", "ml.g5.2xlarge"),
        output_path=f"s3://{bucket}/output/",
        environment=env,
        max_run=max_run_h * 3600,
    )
    if use_spot:
        kwargs.update(use_spot_instances=True, max_wait=max_wait_h * 3600)
        if mode == "train":
            kwargs.update(
                checkpoint_s3_uri=f"s3://{bucket}/checkpoints/{job_name}/",
                checkpoint_local_path="/opt/ml/checkpoints",
            )

    Estimator(**kwargs).fit(job_name=job_name, wait=False)
    print(f"Submitted: {job_name} (mode={mode})")


if __name__ == "__main__":
    main()
