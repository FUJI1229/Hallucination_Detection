ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${MODEL_NAME:-llama3_8b}"
# "llama3_8b" "mistral_12b" "llama3_70b"
DATA_NAME="${DATA_NAME:-trivia_qa}"
# "trivia_qa" "squad" "nq" "bioasq"
DATA_DIR="${DATA_DIR:-$ROOT/Data/$MODEL_NAME/$DATA_NAME}"

python "$ROOT/gen/generation/compute_entailment.py" \
    --data_dir="$DATA_DIR" \
    --model_name="$MODEL_NAME" \
    --data_name="$DATA_NAME"
