#!/usr/bin/env bash
#PJM -L rscgrp=share-short
#PJM -L gpu=4
#PJM -L elapse=02:00:00
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
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_ROOT/.cache/triton}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

MODEL_NAME="${MODEL_NAME:-llama3_70b}"
DATA_NAME="${DATA_NAME:-nq}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/Data/$MODEL_NAME/$DATA_NAME}"

python "$PROJECT_ROOT/gen/generation/generate_answers.py" \
    --data_dir="$DATA_DIR" \
    --model_name="$MODEL_NAME" \
    --data_name="$DATA_NAME" \
    --num_generations=5 \
    --return_layers='0,6' \
    --train_num_samples=4000 \
    --valid_num_samples=2000 \
    --split='train'
