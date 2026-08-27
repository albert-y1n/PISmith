#!/usr/bin/env bash
# ============================================================
# eval_agentdyn.sh — AgentDyn RL Attacker Evaluation
#
# Usage:
#   bash scripts/eval_agentdyn.sh \
#       <checkpoint> [target_type] [eval_suites] [num_samples] [target_defense]
#
# target_type:
#   gpt4o-mini | gpt4o | gpt5-nano | gpt5.6-luna | gpt5.6-terra |
#   gemini-3.7-flash | openrouter | local
#
# Env vars:
#   TARGET_GPU       GPU index for target vLLM server (default: 0, only for local target)
#   TARGET_PORT      Port for target vLLM server (default: 8000, only for local target)
#   ATTACKER_GPU     GPU index for attacker vLLM server (default: 1)
#   ATTACKER_GPUS    Comma-separated GPUs for attacker vLLM server (overrides ATTACKER_GPU)
#   ATTACKER_DP_SIZE Data parallel replicas for attacker vLLM server (default: 1)
#   ATTACKER_PORT    Port for attacker vLLM server (default: 8001)
#   ATTACKER_URL     Use external attacker vLLM server (skips launching a new one)
#   MAX_WORKERS      Concurrent target evaluations (default: 16)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/..:${PYTHONPATH:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

CHECKPOINT=${1:-"checkpoints/agentdyn/checkpoint-"}
TARGET_TYPE=${2:-gpt4o-mini}
EVAL_SUITES=${3:-"github,dailylife,shopping"}
NUM_SAMPLES=${4:-10}
ATTACKER_MAX_TOKENS=${ATTACKER_MAX_TOKENS:-4096}
TARGET_DEFENSE=${5:-}

EVAL_INJ=${EVAL_INJ:-}
EVAL_USER=${EVAL_USER:-}

TARGET_GPU=${TARGET_GPU:-0}
TARGET_PORT=${TARGET_PORT:-8000}
TARGET_MAX_MODEL_LEN=${TARGET_MAX_MODEL_LEN:-131072}
TARGET_MAX_TOKENS=${TARGET_MAX_TOKENS:-32768}
TARGET_PROVIDER_OVERRIDE=${TARGET_PROVIDER:-}
TARGET_API_KEY_ENV_OVERRIDE=${TARGET_API_KEY_ENV:-}
TARGET_BASE_URL=${TARGET_BASE_URL:-}
OPENROUTER_MODEL=${OPENROUTER_MODEL:-google/gemini-3.7-flash}
ATTACKER_GPU=${ATTACKER_GPU:-1}
ATTACKER_GPUS=${ATTACKER_GPUS:-$ATTACKER_GPU}
ATTACKER_DP_SIZE=${ATTACKER_DP_SIZE:-1}
ATTACKER_PORT=${ATTACKER_PORT:-8001}
MAX_WORKERS=${MAX_WORKERS:-16}
VLLM_TARGET_PID=""
VLLM_ATTACKER_PID=""

SUITES_TAG=$(echo "$EVAL_SUITES" | tr ',' '-')
OUTPUT_DIR=${OUTPUT_DIR:-"eval_results/agentdyn_${TARGET_TYPE}_${SUITES_TAG}_pass${NUM_SAMPLES}_$(basename "$CHECKPOINT")"}
[ -n "$TARGET_DEFENSE" ] && OUTPUT_DIR="${OUTPUT_DIR}_${TARGET_DEFENSE}"

case "$TARGET_TYPE" in
  gpt4o-mini)
    TARGET_MODEL="gpt-4o-mini-2024-07-18"
    TARGET_PROVIDER=${TARGET_PROVIDER_OVERRIDE:-openai}
    TARGET_API_KEY_ENV=${TARGET_API_KEY_ENV_OVERRIDE:-OPENAI_API_KEY}
    TARGET_MODEL_ID=""
    TARGET_MODEL_URL=""
    NEEDS_VLLM=0
    ;;
  gpt4o)
    TARGET_MODEL="gpt-4o-2024-05-13"
    TARGET_PROVIDER=${TARGET_PROVIDER_OVERRIDE:-openai}
    TARGET_API_KEY_ENV=${TARGET_API_KEY_ENV_OVERRIDE:-OPENAI_API_KEY}
    TARGET_MODEL_ID=""
    TARGET_MODEL_URL=""
    NEEDS_VLLM=0
    ;;
  gpt5-nano)
    TARGET_MODEL="gpt-5-nano"
    TARGET_PROVIDER=${TARGET_PROVIDER_OVERRIDE:-openai}
    TARGET_API_KEY_ENV=${TARGET_API_KEY_ENV_OVERRIDE:-OPENAI_API_KEY}
    TARGET_MODEL_ID=""
    TARGET_MODEL_URL=""
    NEEDS_VLLM=0
    ;;
  gpt5.6-luna|gpt-5.6-luna)
    TARGET_MODEL="gpt-5.6-luna"
    TARGET_PROVIDER=${TARGET_PROVIDER_OVERRIDE:-openai}
    TARGET_API_KEY_ENV=${TARGET_API_KEY_ENV_OVERRIDE:-OPENAI_API_KEY}
    TARGET_MODEL_ID=""
    TARGET_MODEL_URL=""
    NEEDS_VLLM=0
    ;;
  gpt5.6-terra|gpt-5.6-terra)
    TARGET_MODEL="gpt-5.6-terra"
    TARGET_PROVIDER=${TARGET_PROVIDER_OVERRIDE:-openai}
    TARGET_API_KEY_ENV=${TARGET_API_KEY_ENV_OVERRIDE:-OPENAI_API_KEY}
    TARGET_MODEL_ID=""
    TARGET_MODEL_URL=""
    NEEDS_VLLM=0
    ;;
  gemini-3.7-flash|openrouter)
    TARGET_MODEL="$OPENROUTER_MODEL"
    TARGET_PROVIDER=${TARGET_PROVIDER_OVERRIDE:-openrouter}
    TARGET_API_KEY_ENV=${TARGET_API_KEY_ENV_OVERRIDE:-OPENROUTER_API_KEY}
    TARGET_MODEL_ID=""
    TARGET_MODEL_URL=""
    NEEDS_VLLM=0
    ;;
  local)
    TARGET_MODEL="local"
    TARGET_PROVIDER="vllm"
    TARGET_API_KEY_ENV=""
    TARGET_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
    TARGET_MODEL_URL="http://localhost:${TARGET_PORT}/v1"
    NEEDS_VLLM=1
    ;;
  *)
    echo "Unknown target_type: $TARGET_TYPE"
    echo "Available: gpt4o-mini, gpt4o, gpt5-nano, gpt5.6-luna, gpt5.6-terra, gemini-3.7-flash, openrouter, local"
    exit 1
    ;;
esac

if [ "$NEEDS_VLLM" -eq 0 ] && [ -z "${!TARGET_API_KEY_ENV:-}" ]; then
    echo "ERROR: $TARGET_API_KEY_ENV is not set for provider '$TARGET_PROVIDER'." >&2
    exit 1
fi

echo "============================================================"
echo "  Checkpoint      : $CHECKPOINT"
echo "  Target          : $TARGET_MODEL"
echo "  Provider        : $TARGET_PROVIDER"
if [ "$NEEDS_VLLM" -eq 1 ]; then
    echo "  Target GPU      : $TARGET_GPU port: $TARGET_PORT"
    echo "  Target context/output: $TARGET_MAX_MODEL_LEN / $TARGET_MAX_TOKENS"
fi
echo "  Attacker GPU(s) : $ATTACKER_GPUS port: $ATTACKER_PORT"
echo "  Attacker DP     : $ATTACKER_DP_SIZE"
echo "  Suites          : $EVAL_SUITES"
echo "  Pass@k          : $NUM_SAMPLES"
echo "  Max workers     : $MAX_WORKERS"
echo "  Defense         : ${TARGET_DEFENSE:-none}"
[ -n "$EVAL_INJ" ] && echo "  Inj tasks       : $EVAL_INJ"
[ -n "$EVAL_USER" ] && echo "  User tasks      : $EVAL_USER"
echo "  Output dir      : $OUTPUT_DIR"
echo "============================================================"

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

if [ "$NEEDS_VLLM" -eq 1 ]; then
    LOG_TARGET="logs/vllm_target_eval_gpu${TARGET_GPU}_port${TARGET_PORT}.log"
    echo "Starting target vLLM -> $LOG_TARGET"
    CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m vllm.entrypoints.openai.api_server \
        --model "$TARGET_MODEL_ID" \
        --port "$TARGET_PORT" \
        --max-model-len "$TARGET_MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.8 \
        --dtype bfloat16 \
        --trust-remote-code \
        --disable-frontend-multiprocessing \
        > "$LOG_TARGET" 2>&1 &
    VLLM_TARGET_PID=$!

    TARGET_CHECK_URL="http://localhost:${TARGET_PORT}/v1/models"
    for i in $(seq 1 120); do
        if curl -sf "$TARGET_CHECK_URL" > /dev/null 2>&1; then
            echo "  Target server ready."
            break
        fi
        if ! kill -0 "$VLLM_TARGET_PID" 2>/dev/null; then
            echo "ERROR: Target vLLM process died. Check $LOG_TARGET" >&2
            exit 1
        fi
        echo "  Attempt $i/120 - not ready, sleeping 10s ..."
        sleep 10
    done
fi

if [ -n "${ATTACKER_URL:-}" ]; then
    echo "Using external attacker server: $ATTACKER_URL"
    ATTACKER_ARGS="--attacker_server_url ${ATTACKER_URL}"
else
    LOG_ATTACKER="logs/vllm_attacker_eval_gpu${ATTACKER_GPU}_port${ATTACKER_PORT}.log"
    echo "Starting attacker vLLM -> $LOG_ATTACKER"

    ATTACKER_VLLM_ARGS=()
    if [ "$ATTACKER_DP_SIZE" != "1" ]; then
        ATTACKER_VLLM_ARGS+=(--data-parallel-size "$ATTACKER_DP_SIZE")
    fi

    CUDA_VISIBLE_DEVICES="$ATTACKER_GPUS" python -m vllm.entrypoints.openai.api_server \
        --model "$CHECKPOINT" \
        --port "$ATTACKER_PORT" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.8 \
        --dtype bfloat16 \
        --trust-remote-code \
        --disable-frontend-multiprocessing \
        "${ATTACKER_VLLM_ARGS[@]}" \
        > "$LOG_ATTACKER" 2>&1 &
    VLLM_ATTACKER_PID=$!

    ATTACKER_CHECK_URL="http://localhost:${ATTACKER_PORT}/v1/models"
    for i in $(seq 1 120); do
        if curl -sf "$ATTACKER_CHECK_URL" > /dev/null 2>&1; then
            echo "  Attacker server ready."
            break
        fi
        if ! kill -0 "$VLLM_ATTACKER_PID" 2>/dev/null; then
            echo "ERROR: Attacker vLLM process died. Check $LOG_ATTACKER" >&2
            exit 1
        fi
        echo "  Attempt $i/120 - not ready, sleeping 10s ..."
        sleep 10
    done

    ATTACKER_ARGS="--attacker_server_url http://localhost:${ATTACKER_PORT}/v1"
fi

TARGET_ARGS="--target_model $TARGET_MODEL --target_provider $TARGET_PROVIDER"
[ -n "$TARGET_API_KEY_ENV" ] && TARGET_ARGS="$TARGET_ARGS --target_api_key_env $TARGET_API_KEY_ENV"
[ -n "$TARGET_BASE_URL" ] && TARGET_ARGS="$TARGET_ARGS --target_base_url $TARGET_BASE_URL"
[ -n "$TARGET_MODEL_ID" ] && TARGET_ARGS="$TARGET_ARGS --target_model_id $TARGET_MODEL_ID"
[ -n "$TARGET_MODEL_URL" ] && TARGET_ARGS="$TARGET_ARGS --target_model_url $TARGET_MODEL_URL"
[ "$NEEDS_VLLM" -eq 1 ] && TARGET_ARGS="$TARGET_ARGS --target_max_tokens $TARGET_MAX_TOKENS"
[ -n "$TARGET_DEFENSE" ] && TARGET_ARGS="$TARGET_ARGS --target_defense $TARGET_DEFENSE"

FILTER_ARGS=""
[ -n "$EVAL_INJ" ] && FILTER_ARGS="$FILTER_ARGS --eval_injection_tasks $EVAL_INJ"
[ -n "$EVAL_USER" ] && FILTER_ARGS="$FILTER_ARGS --eval_user_tasks $EVAL_USER"

echo ""
echo "Running AgentDyn evaluation..."
python -m eval.eval_agentdyn \
    --attacker_model "$CHECKPOINT" \
    $ATTACKER_ARGS \
    $TARGET_ARGS \
    --eval_suites "$EVAL_SUITES" \
    --num_samples "$NUM_SAMPLES" \
    --max_tokens "$ATTACKER_MAX_TOKENS" \
    --max_workers "$MAX_WORKERS" \
    --output_dir "$OUTPUT_DIR" \
    $FILTER_ARGS

echo "============================================================"
echo "Evaluation complete."
echo "  Summary : $OUTPUT_DIR/eval_results.json"
echo "  Detailed: $OUTPUT_DIR/eval_detailed.jsonl"
echo "============================================================"
