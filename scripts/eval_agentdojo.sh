#!/usr/bin/env bash
# ============================================================
# eval_agentdojo.sh — AgentDojo RL Attacker Evaluation
#
# Evaluates a trained attacker checkpoint on the AgentDojo benchmark.
# Reports ASR (attack success) and Utility per suite and per injection task.
#
# Usage:
#   bash scripts/eval_agentdojo.sh \
#       <checkpoint> [target_type] [eval_suites] [num_samples] [target_defense]
#
# Arguments:
#   checkpoint     Path to trained attacker checkpoint (required)
#   target_type    gpt4o-mini | gpt4o | local (default: gpt4o-mini)
#   eval_suites    comma-separated suites (default: workspace,banking,travel,slack)
#   num_samples    Pass@k: samples per task pair (default: 1)
#   target_defense AgentDojo defense (default: none)
#
# Env vars:
#   TARGET_GPU     GPU index for target vLLM server (default: 0, only for local target)
#   TARGET_PORT    Port for target vLLM server (default: 8000, only for local target)
#   ATTACKER_GPU   GPU index for attacker vLLM server (default: 1)
#   ATTACKER_PORT  Port for attacker vLLM server (default: 8001)
#   ATTACKER_URL   Use external attacker vLLM server (skips launching a new one)
#
# Examples:
#   # Eval on all suites, GPT-4o-mini target
#   bash scripts/eval_agentdojo.sh \
#       checkpoints/agentdojo/checkpoint-500 gpt4o-mini
#
#   # Eval on banking suite only
#   bash scripts/eval_agentdojo.sh \
#       checkpoints/agentdojo/checkpoint-500 gpt4o-mini banking
#
#   # Eval with defense
#   bash scripts/eval_agentdojo.sh \
#       checkpoints/agentdojo/checkpoint-500 gpt4o-mini workspace 1 tool_filter
#
#   # Eval with local vLLM target (script starts the server automatically)
#   TARGET_GPU=0 TARGET_PORT=8000 \
#   bash scripts/eval_agentdojo.sh \
#       checkpoints/agentdojo/checkpoint-500 local workspace
#
#   # Eval with external attacker vLLM server (faster), pass@5
#   ATTACKER_URL=http://localhost:8001/v1 \
#   bash scripts/eval_agentdojo.sh \
#       checkpoints/agentdojo/checkpoint-500 gpt4o-mini workspace 5
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/..:${PYTHONPATH:-}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# ── Arguments ────────────────────────────────────────────────
CHECKPOINT=${1:-"checkpoints/agentdojo/checkpoint-"}
TARGET_TYPE=${2:-gpt4o-mini}
EVAL_SUITES=${3:-"workspace,banking,travel,slack"}
NUM_SAMPLES=${4:-10}
TARGET_DEFENSE=${5:-}

# Optional fine-grained task filtering (env vars)
EVAL_INJ=${EVAL_INJ:-}    # e.g. "injection_task_0,injection_task_1"
EVAL_USER=${EVAL_USER:-}  # e.g. "user_task_0,user_task_1"

TARGET_GPU=${TARGET_GPU:-0}
TARGET_PORT=${TARGET_PORT:-8000}
ATTACKER_GPU=${ATTACKER_GPU:-1}
ATTACKER_PORT=${ATTACKER_PORT:-8001}
VLLM_TARGET_PID=""
VLLM_ATTACKER_PID=""

OUTPUT_DIR="eval_results/agentdojo_${TARGET_TYPE}_$(basename "$CHECKPOINT")"
if [ -n "$TARGET_DEFENSE" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_${TARGET_DEFENSE}"
fi

# ── Target model config ───────────────────────────────────────
case "$TARGET_TYPE" in
  gpt4o-mini)
    TARGET_MODEL="gpt-4o-mini-2024-07-18"
    TARGET_MODEL_ID=""
    TARGET_MODEL_URL=""
    NEEDS_VLLM=0
    ;;
  gpt4o)
    TARGET_MODEL="gpt-4o-2024-05-13"
    TARGET_MODEL_ID=""
    TARGET_MODEL_URL=""
    NEEDS_VLLM=0
    ;;
  local)
    TARGET_MODEL="local"
    TARGET_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
    TARGET_MODEL_URL="http://localhost:${TARGET_PORT}/v1"
    NEEDS_VLLM=1
    ;;
  *)
    echo "Unknown target_type: $TARGET_TYPE"
    echo "Available: gpt4o-mini, gpt4o, local"
    exit 1
    ;;
esac

# ── Validate API key for cloud targets ────────────────────────
if [ "$NEEDS_VLLM" -eq 0 ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set but target '$TARGET_TYPE' requires OpenAI API." >&2
    echo "  Export it before running: export OPENAI_API_KEY=sk-..." >&2
    exit 1
fi

echo "============================================================"
echo "  Checkpoint  : $CHECKPOINT"
echo "  Target      : $TARGET_MODEL"
[ "$NEEDS_VLLM" -eq 1 ] && echo "  Target GPU  : $TARGET_GPU   port: $TARGET_PORT"
echo "  Attacker GPU: $ATTACKER_GPU port: $ATTACKER_PORT"
echo "  Suites      : $EVAL_SUITES"
echo "  Pass@k      : $NUM_SAMPLES"
echo "  Defense     : ${TARGET_DEFENSE:-none}"
[ -n "$EVAL_INJ"  ] && echo "  Inj tasks   : $EVAL_INJ"
[ -n "$EVAL_USER" ] && echo "  User tasks  : $EVAL_USER"
echo "  Output dir  : $OUTPUT_DIR"
echo "============================================================"

# ── Cleanup handler ───────────────────────────────────────────
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

mkdir -p logs
ulimit -n 65536 2>/dev/null || true

# ── Start target vLLM server (only for local target) ─────────
if [ "$NEEDS_VLLM" -eq 1 ]; then
    LOG_TARGET="logs/vllm_target_eval_gpu${TARGET_GPU}_port${TARGET_PORT}.log"
    echo "Starting target vLLM → $LOG_TARGET"

    CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m vllm.entrypoints.openai.api_server \
        --model "$TARGET_MODEL_ID" \
        --port "$TARGET_PORT" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.8 \
        --dtype bfloat16 \
        --trust-remote-code \
        --disable-frontend-multiprocessing \
        > "$LOG_TARGET" 2>&1 &
    VLLM_TARGET_PID=$!
    echo "  Target vLLM PID: $VLLM_TARGET_PID"

    TARGET_CHECK_URL="http://localhost:${TARGET_PORT}/v1/models"
    echo "Waiting for target vLLM at $TARGET_CHECK_URL ..."
    for i in $(seq 1 120); do
        if curl -sf "$TARGET_CHECK_URL" > /dev/null 2>&1; then
            echo "  Target server ready."
            break
        fi
        if ! kill -0 "$VLLM_TARGET_PID" 2>/dev/null; then
            echo "ERROR: Target vLLM process died. Check $LOG_TARGET" >&2
            exit 1
        fi
        echo "  Attempt $i/120 — not ready, sleeping 10s ..."
        sleep 10
    done
fi

# ── Start attacker vLLM server ────────────────────────────────
if [ -n "${ATTACKER_URL:-}" ]; then
    echo "Using external attacker server: $ATTACKER_URL"
    ATTACKER_ARGS="--attacker_server_url ${ATTACKER_URL}"
else
    LOG_ATTACKER="logs/vllm_attacker_eval_gpu${ATTACKER_GPU}_port${ATTACKER_PORT}.log"
    echo "Starting attacker vLLM → $LOG_ATTACKER"

    CUDA_VISIBLE_DEVICES="$ATTACKER_GPU" python -m vllm.entrypoints.openai.api_server \
        --model "$CHECKPOINT" \
        --port "$ATTACKER_PORT" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.8 \
        --dtype bfloat16 \
        --trust-remote-code \
        --disable-frontend-multiprocessing \
        > "$LOG_ATTACKER" 2>&1 &
    VLLM_ATTACKER_PID=$!
    echo "  Attacker vLLM PID: $VLLM_ATTACKER_PID"

    ATTACKER_CHECK_URL="http://localhost:${ATTACKER_PORT}/v1/models"
    echo "Waiting for attacker vLLM at $ATTACKER_CHECK_URL ..."
    for i in $(seq 1 120); do
        if curl -sf "$ATTACKER_CHECK_URL" > /dev/null 2>&1; then
            echo "  Attacker server ready."
            break
        fi
        if ! kill -0 "$VLLM_ATTACKER_PID" 2>/dev/null; then
            echo "ERROR: Attacker vLLM process died. Check $LOG_ATTACKER" >&2
            exit 1
        fi
        echo "  Attempt $i/120 — not ready, sleeping 10s ..."
        sleep 10
    done

    ATTACKER_ARGS="--attacker_server_url http://localhost:${ATTACKER_PORT}/v1"
fi

# ── Target model args ─────────────────────────────────────────
TARGET_ARGS="--target_model $TARGET_MODEL"
if [ -n "$TARGET_MODEL_ID" ]; then
    TARGET_ARGS="$TARGET_ARGS --target_model_id $TARGET_MODEL_ID"
fi
if [ -n "$TARGET_MODEL_URL" ]; then
    TARGET_ARGS="$TARGET_ARGS --target_model_url $TARGET_MODEL_URL"
fi
if [ -n "$TARGET_DEFENSE" ]; then
    TARGET_ARGS="$TARGET_ARGS --target_defense $TARGET_DEFENSE"
fi

# ── Build optional task filter args ──────────────────────────
FILTER_ARGS=""
[ -n "$EVAL_INJ"  ] && FILTER_ARGS="$FILTER_ARGS --eval_injection_tasks $EVAL_INJ"
[ -n "$EVAL_USER" ] && FILTER_ARGS="$FILTER_ARGS --eval_user_tasks $EVAL_USER"

# ── Run evaluation ────────────────────────────────────────────
echo ""
echo "Running evaluation..."
python -m eval.eval_agentdojo \
    --attacker_model "$CHECKPOINT" \
    $ATTACKER_ARGS \
    $TARGET_ARGS \
    --eval_suites "$EVAL_SUITES" \
    --num_samples "$NUM_SAMPLES" \
    --max_workers 16 \
    --output_dir "$OUTPUT_DIR" \
    $FILTER_ARGS

echo "============================================================"
echo "Evaluation complete."
echo "  Summary : $OUTPUT_DIR/eval_results.json"
echo "  Detailed: $OUTPUT_DIR/eval_detailed.jsonl"
echo "============================================================"
