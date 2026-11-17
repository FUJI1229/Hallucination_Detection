#!/bin/bash
# LongFact++ experiment (Section 6.3): build the cells, extract hidden states
# from the generator itself, and train the max- and mean-pooling detectors
# with an unweighted loss on the class-balanced training split.
#   MODEL_NAME=llama3_8b_Instruct bash longfact_pp/job_lfpp.sh
#   MODEL_NAME=llama3_70b         bash longfact_pp/job_lfpp.sh
# No API key is needed: the labels come with the dataset.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${MODEL_NAME:-llama3_8b_Instruct}"
SAVE_DIR="${SAVE_DIR:-$ROOT/results/longfact_pp}"
SEEDS="${SEEDS:-42 43 44 45 46}"
POOLINGS="${POOLINGS:-max mean}"

if [ "$MODEL_NAME" = "llama3_70b" ]; then
  LAYERS="${LAYERS:-$(seq -s, 0 4 80)}"   # every fourth layer
else
  LAYERS="${LAYERS:-$(seq -s, 0 2 32)}"   # every second layer
fi
# stored-file indices are layer // 2
IDX=$(echo "$LAYERS" | tr ',' '\n' | awk '{printf "%s%d", (NR>1?",":""), $1/2}')

python "$ROOT/longfact_pp/build_cells.py" --model_name "$MODEL_NAME" --data_dir "$ROOT/Data"
python "$ROOT/longfact_pp/extract_hidden.py" --model_name "$MODEL_NAME" --data_dir "$ROOT/Data" --layers "$LAYERS"

for POOL in $POOLINGS; do
  for SEED in $SEEDS; do
    python -u "$ROOT/Detection/main.py" \
      --data_dir "$ROOT/Data/$MODEL_NAME" \
      --model_name "$MODEL_NAME" \
      --data_name longfact_pp \
      --test_data_list '["longfact_pp"]' \
      --fixed_split \
      --layers "$IDX" \
      --seed "$SEED" \
      --epoch 100 \
      --batch_size 128 \
      --reduced_dim 256 \
      --pos_weight 1.0 \
      --pooling_method "$POOL" \
      --save_dir "$SAVE_DIR"
  done
done
python "$ROOT/longfact_pp/summarize.py" --save_dir "$SAVE_DIR" --model_name "$MODEL_NAME"
