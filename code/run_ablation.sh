#!/usr/bin/env bash
# run_ablation.sh — launch the six Week 3 training runs sequentially.
#
# PRECONDITIONS:
#   - ablation_env_cfg.py installed + imported in go2/__init__.py
#   - conda env active:  source /workspace/miniconda3/etc/profile.d/conda.sh
#                        conda activate env_isaaclab
#   - run inside tmux so SSH disconnects don't kill training:
#       tmux new -s ablation
#       bash run_ablation.sh 2>&1 | tee ablation_launch.log
#     (detach: Ctrl-b d   |   reattach: tmux attach -t ablation)
#
# INVARIANTS: same seed, same iterations, same everything — the ONLY
# difference between runs is the task name (one config term each).

set -euo pipefail

cd /home/ubuntu/IsaacLab

SEED=42
ITERS=1000          # matches Week 2 (model_999) — ~10 min/run on the 4090
TRAIN=scripts/reinforcement_learning/rsl_rl/train.py

TASKS=(
  Isaac-Go2-Ablation-FullDr-v0
  Isaac-Go2-Ablation-NoFriction-v0
  Isaac-Go2-Ablation-NoMass-v0
  Isaac-Go2-Ablation-NoMotorStrength-v0
  Isaac-Go2-Ablation-NoSensorNoise-v0
  Isaac-Go2-Ablation-NoActionLatency-v0
)

for TASK in "${TASKS[@]}"; do
  echo "════════════════════════════════════════════════════════"
  echo "  LAUNCHING: ${TASK}   (seed=${SEED}, iters=${ITERS})"
  echo "════════════════════════════════════════════════════════"
  python "${TRAIN}" --task "${TASK}" --headless \
      --seed "${SEED}" --max_iterations "${ITERS}"
  echo "  DONE: ${TASK}"
done

echo "All six runs complete. Checkpoint directories:"
ls -d /home/ubuntu/IsaacLab/logs/rsl_rl/*/20* | tail -n 6
