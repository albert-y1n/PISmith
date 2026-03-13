#!/usr/bin/env bash
# ============================================================
# train_piarena.sh — PIArena RL Attacker Training (single target)
#
# Starts the target vLLM server, waits until ready,
# runs GRPO training, then shuts down the server.
#
# Usage (run from the PISmith/ directory):
#   bash scripts/train_piarena.sh [defense] [train_gpus] [target_gpu] [target_port]
#
# Examples:
#   bash scripts/train_piarena.sh secalign "1,2,3" 0 8010
#   bash scripts/train_piarena.sh none      "1,2,3" 0 8010
#   bash scripts/train_piarena.sh secalign  "1"     0 8010
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/..:${PYTHONPATH:-}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# ── Arguments ────────────────────────────────────────────────
DEFENSE=${1:-secalign}
TRAIN_GPUS=${2:-"1,2,3"}
TARGET_GPU=${3:-0}
TARGET_PORT=${4:-8010}

SECALIGN_MODEL="checkpoints/Meta-SecAlign-8B-merged"
QWEN_MODEL="Qwen/Qwen3-4B-Instruct-2507"

# ── Resolve config and target model ──────────────────────────
case "$DEFENSE" in
  secalign)      CONFIG_FILE="configs/piarena.yaml";              TARGET_MODEL="$SECALIGN_MODEL" ;;
  none)          CONFIG_FILE="configs/piarena_none.yaml";          TARGET_MODEL="$QWEN_MODEL" ;;
  pisanitizer)   CONFIG_FILE="configs/piarena_pisanitizer.yaml";   TARGET_MODEL="$QWEN_MODEL" ;;
  promptguard)   CONFIG_FILE="configs/piarena_promptguard.yaml";   TARGET_MODEL="$QWEN_MODEL" ;;
  promptarmor)   CONFIG_FILE="configs/piarena_promptarmor.yaml";   TARGET_MODEL="$QWEN_MODEL" ;;
  sandwich)      CONFIG_FILE="configs/piarena_sandwich.yaml";      TARGET_MODEL="$QWEN_MODEL" ;;
  instructional) CONFIG_FILE="configs/piarena_instructional.yaml"; TARGET_MODEL="$QWEN_MODEL" ;;
  datasentinel)  CONFIG_FILE="configs/piarena_datasentinel.yaml";  TARGET_MODEL="$QWEN_MODEL" ;;
  piguard)       CONFIG_FILE="configs/piarena_piguard.yaml";       TARGET_MODEL="$QWEN_MODEL" ;;
  datafilter)    CONFIG_FILE="configs/piarena_datafilter.yaml";    TARGET_MODEL="$QWEN_MODEL" ;;
  *)
    echo "Unknown defense: $DEFENSE"
    echo "Supported: secalign, none, pisanitizer, promptguard, promptarmor, sandwich, instructional, datasentinel, piguard, datafilter"
    exit 1
    ;;
esac

NUM_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
ACCEL_CONFIG="configs/accelerate.yaml"
VLLM_PID=""

# ── Cleanup handler ───────────────────────────────────────────
cleanup() {
    echo ""
    echo "Shutting down vLLM server (PID $VLLM_PID)..."
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        echo "  Done."
    fi
}
trap cleanup EXIT

echo "============================================================"
echo "  Defense      : $DEFENSE"
echo "  Config       : $CONFIG_FILE"
echo "  Train GPUs   : $TRAIN_GPUS ($NUM_GPUS GPU(s))"
echo "  Target model : $TARGET_MODEL"
echo "  Target GPU   : $TARGET_GPU   port: $TARGET_PORT"
echo "============================================================"

# ── Start vLLM target server ─────────────────────────────────
mkdir -p logs
LOG="logs/vllm_target_gpu${TARGET_GPU}_port${TARGET_PORT}.log"

echo "Starting vLLM server → $LOG"
CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" \
    --port "$TARGET_PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.35 \
    --dtype bfloat16 \
    --trust-remote-code \
    > "$LOG" 2>&1 &
VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

# ── Wait for server to be ready ──────────────────────────────
TARGET_URL="http://localhost:${TARGET_PORT}/v1/models"
echo "Waiting for vLLM at $TARGET_URL ..."
for i in $(seq 1 120); do
    if curl -sf "$TARGET_URL" > /dev/null 2>&1; then
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

# ── Launch training ───────────────────────────────────────────
echo ""
echo "Launching training..."
if [ "$NUM_GPUS" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python -m train \
        --benchmark piarena \
        --config_file "$CONFIG_FILE" \
        --target_model_url "http://localhost:${TARGET_PORT}/v1" \
        --target_model_name_or_path "$TARGET_MODEL"
else
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" accelerate launch \
        --config_file "$ACCEL_CONFIG" \
        --num_processes "$NUM_GPUS" \
        -m train \
        --benchmark piarena \
        --config_file "$CONFIG_FILE" \
        --target_model_url "http://localhost:${TARGET_PORT}/v1" \
        --target_model_name_or_path "$TARGET_MODEL"
fi

echo "============================================================"
echo "Training complete."
echo "============================================================"
