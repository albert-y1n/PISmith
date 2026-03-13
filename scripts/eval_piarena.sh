#!/usr/bin/env bash
# ============================================================
# eval_piarena.sh — PIArena RL Attacker Evaluation
#
# Like train_piarena.sh: starts target vLLM and attacker vLLM,
# waits until ready, runs evaluation, then shuts down both.
#
# Usage:
#   bash scripts/eval_piarena.sh \
#       <checkpoint> <defense> [target_port] [target_gpu] [attacker_gpu] [attacker_port] [num_samples]
#
# Arguments:
#   checkpoint     Path to trained attacker checkpoint
#   defense        Defense method (secalign, none, pisanitizer, ...)
#   target_port    vLLM target server port (default: 8000)
#   target_gpu     GPU for target vLLM (default: 0)
#   attacker_gpu   GPU for attacker vLLM (default: 1)
#   attacker_port  vLLM attacker server port (default: 8001)
#   num_samples    Pass@k: samples per case (default: 10)
#
# Examples:
#   bash scripts/eval_piarena.sh \
#       checkpoints/piarena/checkpoint-500 secalign
#   bash scripts/eval_piarena.sh \
#       checkpoints/piarena_none/checkpoint-500 none 8000 0 1 8001 10
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/..:${PYTHONPATH:-}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"


# ── Arguments ────────────────────────────────────────────────
CHECKPOINT=${1:-"checkpoints/piarena/checkpoint-"}
DEFENSE=${2:-secalign}
TARGET_PORT=${3:-8000}
TARGET_GPU=${4:-0}
ATTACKER_GPU=${5:-1}
ATTACKER_PORT=${6:-8001}
NUM_SAMPLES=${7:-10}

JUDGE_CONFIG="configs/judge.yaml"
OUTPUT_DIR="eval_results/piarena_${DEFENSE}_$(basename $CHECKPOINT)"
DATA_PATH="data/piarena"

# ── Model name for each defense (align with train_piarena.sh) ─
SECALIGN_MODEL="checkpoints/Meta-SecAlign-8B-merged"
QWEN_MODEL="Qwen/Qwen3-4B-Instruct-2507"
case "$DEFENSE" in
  secalign)
    TARGET_MODEL="$SECALIGN_MODEL"
    ;;
  none | sandwich | instructional | pisanitizer | promptguard | \
  datasentinel | datafilter | attentiontracker | piguard | promptarmor)
    TARGET_MODEL="$QWEN_MODEL"
    ;;
  *)
    echo "Unknown defense: $DEFENSE"
    exit 1
    ;;
esac

VLLM_TARGET_PID=""
VLLM_ATTACKER_PID=""

# ── Cleanup handler ──────────────────────────────────────────
cleanup() {
    echo ""
    echo "Shutting down vLLM servers..."
    if [[ -n "$VLLM_ATTACKER_PID" ]] && kill -0 "$VLLM_ATTACKER_PID" 2>/dev/null; then
        kill "$VLLM_ATTACKER_PID" 2>/dev/null || true
        echo "  Attacker vLLM (PID $VLLM_ATTACKER_PID) stopped."
    fi
    if [[ -n "$VLLM_TARGET_PID" ]] && kill -0 "$VLLM_TARGET_PID" 2>/dev/null; then
        kill "$VLLM_TARGET_PID" 2>/dev/null || true
        echo "  Target vLLM (PID $VLLM_TARGET_PID) stopped."
    fi
}
trap cleanup EXIT

echo "============================================================"
echo "  Checkpoint     : $CHECKPOINT"
echo "  Defense        : $DEFENSE"
echo "  Target model   : $TARGET_MODEL"
echo "  Target GPU     : $TARGET_GPU   port: $TARGET_PORT"
echo "  Attacker GPU   : $ATTACKER_GPU port: $ATTACKER_PORT"
echo "  Pass@k         : $NUM_SAMPLES"
echo "  Output dir     : $OUTPUT_DIR"
echo "============================================================"

# ── Start target vLLM ────────────────────────────────────────
mkdir -p logs
LOG_TARGET="logs/vllm_target_eval_gpu${TARGET_GPU}_port${TARGET_PORT}.log"
echo "Starting target vLLM → $LOG_TARGET"
CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" \
    --port "$TARGET_PORT" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.4 \
    --dtype bfloat16 \
    --trust-remote-code \
    > "$LOG_TARGET" 2>&1 &
VLLM_TARGET_PID=$!
echo "  Target vLLM PID: $VLLM_TARGET_PID"

# ── Start attacker vLLM ──────────────────────────────────────
LOG_ATTACKER="logs/vllm_attacker_eval_gpu${ATTACKER_GPU}_port${ATTACKER_PORT}.log"
echo "Starting attacker vLLM → $LOG_ATTACKER"
CUDA_VISIBLE_DEVICES="$ATTACKER_GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$CHECKPOINT" \
    --port "$ATTACKER_PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --dtype bfloat16 \
    --trust-remote-code \
    > "$LOG_ATTACKER" 2>&1 &
VLLM_ATTACKER_PID=$!
echo "  Attacker vLLM PID: $VLLM_ATTACKER_PID"

# ── Wait for target vLLM ─────────────────────────────────────
TARGET_URL="http://localhost:${TARGET_PORT}/v1/models"
echo "Waiting for target vLLM at $TARGET_URL ..."
for i in $(seq 1 120); do
    if curl -sf "$TARGET_URL" > /dev/null 2>&1; then
        echo "  Target server ready."
        break
    fi
    if [[ -n "$VLLM_TARGET_PID" ]] && ! kill -0 "$VLLM_TARGET_PID" 2>/dev/null; then
        echo "ERROR: Target vLLM died. Check $LOG_TARGET" >&2
        exit 1
    fi
    echo "  Attempt $i/120 — not ready, sleeping 10s ..."
    sleep 10
done

# ── Wait for attacker vLLM ────────────────────────────────────
ATTACKER_URL="http://localhost:${ATTACKER_PORT}/v1/models"
echo "Waiting for attacker vLLM at $ATTACKER_URL ..."
for i in $(seq 1 120); do
    if curl -sf "$ATTACKER_URL" > /dev/null 2>&1; then
        echo "  Attacker server ready."
        break
    fi
    if [[ -n "$VLLM_ATTACKER_PID" ]] && ! kill -0 "$VLLM_ATTACKER_PID" 2>/dev/null; then
        echo "ERROR: Attacker vLLM died. Check $LOG_ATTACKER" >&2
        exit 1
    fi
    echo "  Attempt $i/120 — not ready, sleeping 10s ..."
    sleep 10
done

# ── Run evaluation (use vLLM attacker server) ─────────────────
echo ""
echo "Running evaluation..."
python -m eval.eval_piarena \
    --attacker_model "$CHECKPOINT" \
    --attacker_server_url "http://localhost:${ATTACKER_PORT}/v1" \
    --target_model_url "http://localhost:${TARGET_PORT}/v1" \
    --target_model_name_or_path "$TARGET_MODEL" \
    --defense_method "$DEFENSE" \
    --judge_model_config "$JUDGE_CONFIG" \
    --data_path "$DATA_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16

echo "============================================================"
echo "Evaluation complete. Results: $OUTPUT_DIR/eval_results.json"
echo "============================================================"
