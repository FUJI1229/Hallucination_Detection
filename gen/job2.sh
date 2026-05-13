MODEL_NAME="${MODEL_NAME:-llama3_8b}"
# "llama3_8b" "mistral_12b" "llama3_70b"
DATA_NAME="${DATA_NAME:-trivia_qa}"
# "trivia_qa" "squad" "nq" "bioasq"
DATA_DIR="${DATA_DIR:-Data/$MODEL_NAME/$DATA_NAME}"
EVAL_MODEL="${EVAL_MODEL:-gpt-5-mini}"

python "generation/eval_correctness.py" \
    --data_dir="$DATA_DIR" \
    --model_name="$MODEL_NAME" \
    --eval_model="$EVAL_MODEL" \
    --use_num_generations=5 \
    --split='train'
