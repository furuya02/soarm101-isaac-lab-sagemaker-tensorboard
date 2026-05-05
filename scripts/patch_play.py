"""Update pretrained_checkpoint import path for the pinned base image.

Isaac Lab 2.3.x で pretrained_checkpoint モジュールが
isaaclab.utils → isaaclab_rl.utils に移動。base image を 2.3.2 に pin
している本記事では新 path のみ存在するため、isaac_so_arm101 の play.py
を新 path に書き換える。
"""

from pathlib import Path

PLAY_PY = Path("/opt/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py")
OLD = "from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint"
NEW = "from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint"

src = PLAY_PY.read_text()
if OLD in src:
    PLAY_PY.write_text(src.replace(OLD, NEW, 1))
elif NEW not in src:
    raise SystemExit(f"pattern not found in {PLAY_PY}")
