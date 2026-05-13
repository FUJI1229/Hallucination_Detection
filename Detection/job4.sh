MODEL_NAME="${MODEL_NAME:-llama3_8b}"
# "llama3_8b" "mistral_12b" "llama3_70b"
DATA_NAME="${DATA_NAME:-trivia_qa}"
# "trivia_qa" "squad" "nq" "bioasq"
DATA_DIR="${DATA_DIR:-Data/$MODEL_NAME/$DATA_NAME}"
SAVE_DIR="${SAVE_DIR:-results}"

if [ "${DATA_NAME}" = "bioasq" ]; then
  VAL_SAMPLE="${VAL_SAMPLE:-600}"
else
  VAL_SAMPLE="${VAL_SAMPLE:-900}"
fi

python -u "main.py" \
    --data_dir="$DATA_DIR" \
    --model_name="$MODEL_NAME" \
    --data_name="$DATA_NAME" \
    --seed=42 \
    --val_interval=1 \
    --save_dir="$SAVE_DIR" \
    --epoch=100 \
    --val_sample="$VAL_SAMPLE" \
    --batch_size=128 \
    --test_data_list="[\"$DATA_NAME\"]" \
    --reduced_dim=256 \
    --pooling_method="max"
    # "max" "mean" "attention" "gated_attention"
