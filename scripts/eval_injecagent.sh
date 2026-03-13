#!/usr/bin/env bash
# ============================================================
# eval_injecagent.sh — InjecAgent RL Attacker Evaluation
#
# Starts the target vLLM server (for local models), waits until
# ready, runs evaluation, then shuts down the server.
#
# Usage:
#   bash scripts/eval_injecagent.sh \
#       <checkpoint> [target_type] [target_gpu] [target_port] [eval_gpu] [num_samples]
#
# Arguments:
#   checkpoint   Path to trained attacker checkpoint (required)
#   target_type  vllm | gpt4o-mini (default: vllm)
#   target_gpu   GPU for target vLLM server (default: 0)
#   target_port  Port for vLLM target server (default: 8000)
#   eval_gpu     GPU for local attacker inference (default: 1)
#   num_samples  Pass@k samples per case (default: 1)
#
# Examples:
#   # Evaluate against local target (pass@1)
#   bash scripts/eval_injecagent.sh \
#       checkpoints/injecagent/checkpoint-500 vllm 0 8000 1
#
#   # Evaluate against GPT-4o-mini API
#   bash scripts/eval_injecagent.sh \
#       checkpoints/injecagent/checkpoint-500 gpt4o-mini
#
#   # Evaluate with external attacker vLLM server (faster), pass@10
#   ATTACKER_URL=http://localhost:8001/v1 \
#   bash scripts/eval_injecagent.sh \
#       checkpoints/injecagent/checkpoint-500 vllm 0 8000 1 10
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/..:${PYTHONPATH:-}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# ── Arguments ────────────────────────────────────────────────
CHECKPOINT=${1:-"checkpoints/injecagent/checkpoint-"}
TARGET_TYPE=${2:-vllm}
TARGET_GPU=${3:-0}
TARGET_PORT=${4:-8000}
EVAL_GPU=${5:-1}
NUM_SAMPLES=${6:-10}

TEST_DATA="data/injecagent/dataset/test.json"
OUTPUT_DIR="eval_results/injecagent_$(basename "$CHECKPOINT")"
VLLM_PID=""

# ── Target model config per type ─────────────────────────────
case "$TARGET_TYPE" in
  vllm)
    TARGET_MODEL="checkpoints/Meta-SecAlign-8B-merged"
    NEEDS_VLLM=1
    echo "Target type  : vLLM (Meta-SecAlign-8B)"
    ;;
  gpt4o-mini)
    TARGET_MODEL="gpt-4o-mini-2024-07-18"
    NEEDS_VLLM=0
    echo "Target type  : GPT-4o-mini (OpenAI/Azure API)"
    ;;
  *)
    echo "Unknown target_type: $TARGET_TYPE. Available: vllm, gpt4o-mini"
    exit 1
    ;;
esac

echo "============================================================"
echo "  Checkpoint  : $CHECKPOINT"
echo "  Target type : $TARGET_TYPE"
echo "  Target GPU  : $TARGET_GPU   port: $TARGET_PORT"
echo "  Eval GPU    : $EVAL_GPU"
echo "  Pass@k      : $NUM_SAMPLES"
echo "  Test data   : $TEST_DATA"
echo "  Output dir  : $OUTPUT_DIR"
echo "============================================================"

# ── Cleanup handler ───────────────────────────────────────────
cleanup() {
    echo ""
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Shutting down vLLM server (PID $VLLM_PID)..."
        kill "$VLLM_PID"
        echo "  Done."
    fi
}
trap cleanup EXIT

# ── Start target vLLM server (only if needed) ────────────────
if [ "$NEEDS_VLLM" -eq 1 ]; then
    mkdir -p logs
    LOG="logs/vllm_target_eval_gpu${TARGET_GPU}_port${TARGET_PORT}.log"
    echo "Starting vLLM server → $LOG"
    ulimit -n 65536 2>/dev/null || true

    CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m vllm.entrypoints.openai.api_server \
        --model "$TARGET_MODEL" \
        --port "$TARGET_PORT" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.8 \
        --dtype bfloat16 \
        --trust-remote-code \
        > "$LOG" 2>&1 &
    VLLM_PID=$!
    echo "  vLLM PID: $VLLM_PID"

    TARGET_CHECK_URL="http://localhost:${TARGET_PORT}/v1/models"
    echo "Waiting for vLLM at $TARGET_CHECK_URL ..."
    for i in $(seq 1 120); do
        if curl -sf "$TARGET_CHECK_URL" > /dev/null 2>&1; then
            echo "  Server ready."
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM process died. Check $LOG" >&2
            exit 1
        fi
        echo "  Attempt $i/120 — not ready, sleeping 10s ..."
        sleep 10
    done
fi

# ── Attacker setup ────────────────────────────────────────────
if [ -n "${ATTACKER_URL:-}" ]; then
    ATTACKER_ARGS="--attacker_server_url ${ATTACKER_URL}"
else
    ATTACKER_ARGS=""
    export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
fi

# ── Build target args ─────────────────────────────────────────
if [ "$NEEDS_VLLM" -eq 1 ]; then
    TARGET_ARGS=(
        --target_model_url "http://localhost:${TARGET_PORT}/v1"
        --target_model_name_or_path "$TARGET_MODEL"
    )
else
    TARGET_ARGS=(
        --target_model_name_or_path "$TARGET_MODEL"
        --use_openai_target
    )
fi

# ── Run evaluation ────────────────────────────────────────────
echo ""
echo "Running evaluation..."
python -m eval.eval_injecagent \
    --attacker_model "$CHECKPOINT" \
    $ATTACKER_ARGS \
    "${TARGET_ARGS[@]}" \
    --data_path "$TEST_DATA" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 20 \
    --num_workers 10

echo "============================================================"
echo "Evaluation complete. Results: $OUTPUT_DIR"
echo "============================================================"
