#!/usr/bin/env bash
#PJM -L rscgrp=share-short
#PJM -L gpu=2
#PJM -L elapse=00:15:00
#PJM -g <project-group>
#PJM -j
#PJM --fs /work,/data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$PROJECT_ROOT/.env"
  set +a
fi

if [ -n "${CONDA_HOME:-}" ] && [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  . "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-hami}"
fi

if command -v module >/dev/null 2>&1; then
  module load cuda/12.6
  module load gcc/12.2.0
fi

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"

MODEL_NAME="${MODEL_NAME:-llama3_8b}"
DATA_NAME="${DATA_NAME:-trivia_qa}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/Data/$MODEL_NAME}"
SAVE_DIR="${SAVE_DIR:-$PROJECT_ROOT/results}"

if [ "${DATA_NAME}" = "bioasq" ]; then
  VAL_SAMPLE="${VAL_SAMPLE:-600}"
else
  VAL_SAMPLE="${VAL_SAMPLE:-900}"
fi

python -u "$PROJECT_ROOT/Detection/main.py" \
    --data_dir="$DATA_DIR" \
    --model_name="$MODEL_NAME" \
    --data_name="$DATA_NAME" \
    --seed=42 \
    --val_interval=1 \
    --save_dir="$SAVE_DIR" \
    --epoch=30 \
    --val_sample="$VAL_SAMPLE" \
    --batch_size=128 \
    --test_data_list="[\"$DATA_NAME\"]" \
    --reduced_dim=256 \
    --pooling_method="gated_attention"
