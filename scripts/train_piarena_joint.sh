#!/usr/bin/env bash
# ============================================================
# train_piarena_joint.sh — PIArena Joint RL Attacker Training
#
# Starts TWO target vLLM servers (Meta-SecAlign-8B + Llama-3.1-8B-Instruct),
# waits until both are ready, runs joint GRPO training,
# then shuts down both servers.
#
# Reward structure (defined in piarena_joint.yaml):
#   SecAlign only succeeded  → 0.7
#   Llama only succeeded     → 0.3
#   Both succeeded           → 1.0
#   Neither                  → 0.0
# No KL divergence (beta=0.0).
#
# Both target models share a single GPU (each 40% VRAM).
#
# Usage:
#   bash scripts/train_piarena_joint.sh [train_gpus] [target_gpu] [secalign_port] [llama_port]
#
# Examples:
#   # GPU 0 for both targets, GPUs 1,2,3 for training
#   bash scripts/train_piarena_joint.sh "1,2,3" 0 8010 8011
#
#   # Single training GPU
#   bash scripts/train_piarena_joint.sh "1" 0 8010 8011
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/..:${PYTHONPATH:-}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# ── Arguments ────────────────────────────────────────────────
TRAIN_GPUS=${1:-"1,2,3"}
TARGET_GPU=${2:-0}          # single GPU shared by both target servers
SECALIGN_PORT=${3:-8010}
LLAMA_PORT=${4:-8011}

SECALIGN_MODEL="checkpoints/Meta-SecAlign-8B-merged"
LLAMA_MODEL="meta-llama/Llama-3.1-8B-Instruct"

CONFIG_FILE="configs/piarena_joint.yaml"
NUM_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
ACCEL_CONFIG="configs/accelerate.yaml"
VLLM_PIDS=()

# ── Cleanup handler ───────────────────────────────────────────
cleanup() {
    echo ""
    echo "Shutting down vLLM servers..."
    for pid in "${VLLM_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "  Killed PID $pid"
        fi
    done
}
trap cleanup EXIT

echo "============================================================"
echo "  Mode         : Joint (SecAlign + Llama, both on GPU $TARGET_GPU)"
echo "  Config       : $CONFIG_FILE"
echo "  Train GPUs   : $TRAIN_GPUS ($NUM_GPUS GPU(s))"
echo "  SecAlign     : $SECALIGN_MODEL (port $SECALIGN_PORT, mem 0.4)"
echo "  Llama        : $LLAMA_MODEL (port $LLAMA_PORT, mem 0.4)"
echo "============================================================"

mkdir -p logs

# ── Start SecAlign vLLM server (GPU $TARGET_GPU, 40% VRAM) ────
LOG_SECALIGN="logs/vllm_secalign_port${SECALIGN_PORT}.log"
echo "Starting SecAlign vLLM → $LOG_SECALIGN"
CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$SECALIGN_MODEL" \
    --port "$SECALIGN_PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.4 \
    --dtype bfloat16 \
    --trust-remote-code \
    > "$LOG_SECALIGN" 2>&1 &
VLLM_PIDS+=($!)
echo "  SecAlign vLLM PID: ${VLLM_PIDS[-1]}"

# ── Start Llama vLLM server (GPU 0, 40% VRAM) ────────────────
LOG_LLAMA="logs/vllm_llama_port${LLAMA_PORT}.log"
echo "Starting Llama vLLM → $LOG_LLAMA"
CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$LLAMA_MODEL" \
    --port "$LLAMA_PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.4 \
    --dtype bfloat16 \
    --trust-remote-code \
    > "$LOG_LLAMA" 2>&1 &
VLLM_PIDS+=($!)
echo "  Llama vLLM PID: ${VLLM_PIDS[-1]}"

# ── Wait for both servers ────────────────────────────────────
wait_vllm() {
    local PORT="$1"
    local NAME="$2"
    local PID="$3"
    local URL="http://localhost:${PORT}/v1/models"
    echo "Waiting for $NAME at $URL ..."
    for i in $(seq 1 120); do
        if curl -sf "$URL" > /dev/null 2>&1; then
            echo "  $NAME ready."
            return
        fi
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "ERROR: $NAME vLLM process died. Check the log." >&2
            exit 1
        fi
        echo "  Attempt $i/120 — not ready, sleeping 10s ..."
        sleep 10
    done
    echo "ERROR: $NAME on port $PORT failed to start after 20 minutes." >&2
    exit 1
}

wait_vllm "$SECALIGN_PORT" "SecAlign" "${VLLM_PIDS[0]}"
wait_vllm "$LLAMA_PORT"    "Llama"    "${VLLM_PIDS[1]}"

# ── Launch joint training ─────────────────────────────────────
echo ""
echo "Launching joint training..."
if [ "$NUM_GPUS" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python -m train \
        --benchmark piarena \
        --config_file "$CONFIG_FILE" \
        --target_model_url "http://localhost:${SECALIGN_PORT}/v1" \
        --target_model_name_or_path "$SECALIGN_MODEL" \
        --target_model_url_2 "http://localhost:${LLAMA_PORT}/v1" \
        --target_model_name_or_path_2 "$LLAMA_MODEL"
else
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" accelerate launch \
        --config_file "$ACCEL_CONFIG" \
        --num_processes "$NUM_GPUS" \
        -m train \
        --benchmark piarena \
        --config_file "$CONFIG_FILE" \
        --target_model_url "http://localhost:${SECALIGN_PORT}/v1" \
        --target_model_name_or_path "$SECALIGN_MODEL" \
        --target_model_url_2 "http://localhost:${LLAMA_PORT}/v1" \
        --target_model_name_or_path_2 "$LLAMA_MODEL"
fi

echo "============================================================"
echo "Joint training complete."
echo "============================================================"
