#!/usr/bin/env bash
set -eu
case "${MODE:-train}" in
  train) exec /workspace/isaaclab/isaaclab.sh -p /opt/ml/code/train.py "$@" ;;
  play)  exec /workspace/isaaclab/isaaclab.sh -p /opt/ml/code/play.py  "$@" ;;
  *) echo "unknown MODE: ${MODE:-train}" >&2; exit 2 ;;
esac
