#!/usr/bin/env bash
# ============================================================
# train_injecagent.sh — InjecAgent RL Attacker Training
#
# Starts the target vLLM server (for local models), waits until
# ready, runs GRPO training, then shuts down the server.
#
# Usage:
#   bash scripts/train_injecagent.sh [target_type] [train_gpus] [target_gpu] [target_port]
#
# target_type:
#   vllm        Local vLLM target (default, uses Llama-3.1-8B)
#   gpt4o-mini  Azure/OpenAI GPT-4o-mini (no server needed)
#   multi       Multiple targets: Llama + GPT-4o-mini (semicolon-separated)
#
# Examples:
#   # Local vLLM target (most common for OSS research)
#   bash scripts/train_injecagent.sh vllm "1,2,3" 0 8000
#
#   # GPT-4o-mini API target (requires Azure env vars)
#   bash scripts/train_injecagent.sh gpt4o-mini "1,2,3"
#
#   # Multi-target: local + API
#   bash scripts/train_injecagent.sh multi "1,2,3" 0 8000
#
#   # Single GPU
#   bash scripts/train_injecagent.sh vllm "1" 0 8000
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: ${PROJECT_ROOT}"

# Allow `python -m train` to resolve regardless of where the repo
# is cloned (the parent of PROJECT_ROOT must be on PYTHONPATH).
export PYTHONPATH="${PROJECT_ROOT}/..:${PYTHONPATH:-}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# ── Arguments ────────────────────────────────────────────────
TARGET_TYPE=${1:-vllm}
TRAIN_GPUS=${2:-"1,2,3"}
TARGET_GPU=${3:-0}
TARGET_PORT=${4:-8000}

DATA_PATH="data/injecagent/dataset/train.json"
ACCEL_CONFIG="configs/accelerate.yaml"
NUM_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
VLLM_PID=""

# ── Target model config per type ─────────────────────────────
case "$TARGET_TYPE" in
  vllm)
    TARGET_MODEL="checkpoints/Meta-SecAlign-8B-merged"
    TARGET_URL="http://localhost:${TARGET_PORT}/v1"
    NEEDS_VLLM=1
    echo "Target type  : vLLM (Meta-SecAlign-8B)"
    ;;
  gpt4o-mini)
    TARGET_MODEL="gpt-4o-mini-2024-07-18"
    TARGET_URL="http://localhost:${TARGET_PORT}/v1"   # unused but required by config
    NEEDS_VLLM=0
    echo "Target type  : GPT-4o-mini (OpenAI API)"
    echo "Required env : GPT-4O-MINI_API_KEY, GPT-4O-MINI_ENDPOINT"
    ;;
  multi)
    # Semicolon-separated: local Llama + GPT-4o-mini
    TARGET_MODEL="meta-llama/Llama-3.1-8B-Instruct;gpt-4o-mini-2024-07-18"
    TARGET_URL="http://localhost:${TARGET_PORT}/v1;http://localhost:${TARGET_PORT}/v1"
    NEEDS_VLLM=1
    echo "Target type  : Multi (Llama-3.1-8B + GPT-4o-mini)"
    ;;
  *)
    echo "Unknown target_type: $TARGET_TYPE"
    echo "Available: vllm, gpt4o-mini, multi"
    exit 1
    ;;
esac

echo "============================================================"
echo "  Target type : $TARGET_TYPE"
echo "  Train GPUs  : $TRAIN_GPUS ($NUM_GPUS GPU(s))"
echo "  Target GPU  : $TARGET_GPU   port: $TARGET_PORT"
echo "  Data        : $DATA_PATH"
echo "  Output      : checkpoints/injecagent"
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
    LOG="logs/vllm_target_gpu${TARGET_GPU}_port${TARGET_PORT}.log"
    echo "Starting vLLM server → $LOG"

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

# ── Launch training ───────────────────────────────────────────
echo ""
echo "Launching training..."
TRAIN_CMD=(
    -m train
    --benchmark injecagent
    --config_file configs/injecagent.yaml
    --dataset "$DATA_PATH"
    --target_model_name_or_path "$TARGET_MODEL"
    --target_model_url "$TARGET_URL"
)

if [ "$NUM_GPUS" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python "${TRAIN_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" accelerate launch \
        --config_file "$ACCEL_CONFIG" \
        --num_processes "$NUM_GPUS" \
        "${TRAIN_CMD[@]}"
fi

echo "============================================================"
echo "Training complete. Checkpoints: checkpoints/injecagent"
echo "============================================================"
