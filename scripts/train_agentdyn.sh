#!/usr/bin/env bash
# ============================================================
# train_agentdyn.sh — AgentDyn RL Attacker Training
#
# Usage:
#   bash scripts/train_agentdyn.sh [target_type] [suites] [train_gpus]
#
# target_type:
#   gpt4o-mini | gpt4o | gpt5-nano | gpt5.6-luna | gpt5.6-terra | local
#
# suites:
#   workspace | github | dailylife | shopping | all | comma-separated suites
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/..:${PYTHONPATH:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

TARGET_TYPE=${1:-gpt4o-mini}
SUITES=${2:-workspace}
TRAIN_GPUS=${3:-"0,1,2,3"}

TARGET_PORT=${TARGET_PORT:-8000}
TARGET_DEFENSE=${TARGET_DEFENSE:-}
ATTACKER_MODEL=${ATTACKER_MODEL:-"Qwen/Qwen3-4B-Instruct-2507"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/agentdyn"}
RUN_NAME=${RUN_NAME:-}

TRAIN_INJ=${TRAIN_INJ:-}
EVAL_SUITES=${EVAL_SUITES:-}
EVAL_INJ=${EVAL_INJ:-}
TRAIN_USER=${TRAIN_USER:-}
EVAL_USER=${EVAL_USER:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

# Optional training overrides. When unset, values come from configs/agentdyn.yaml.
LEARNING_RATE=${LEARNING_RATE:-}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-}
SAVE_STEPS=${SAVE_STEPS:-}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-}

ACCEL_CONFIG="configs/accelerate.yaml"
NUM_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)

if [ "$SUITES" = "all" ]; then
    SUITES="github,dailylife,shopping"
fi
SUITES_TAG=$(echo "$SUITES" | tr ',' '-')

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
  gpt5-nano)
    TARGET_MODEL="gpt-5-nano"
    TARGET_MODEL_ID=""
    TARGET_MODEL_URL=""
    NEEDS_VLLM=0
    ;;
  gpt5.6-luna|gpt-5.6-luna)
    TARGET_MODEL="gpt-5.6-luna"
    TARGET_MODEL_ID=""
    TARGET_MODEL_URL=""
    NEEDS_VLLM=0
    ;;
  gpt5.6-terra|gpt-5.6-terra)
    TARGET_MODEL="gpt-5.6-terra"
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
    echo "Available: gpt4o-mini, gpt4o, gpt5-nano, gpt5.6-luna, gpt5.6-terra, local"
    exit 1
    ;;
esac

if [ "$NEEDS_VLLM" -eq 0 ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set but target '$TARGET_TYPE' requires OpenAI API." >&2
    echo "  Export it before running: export OPENAI_API_KEY=sk-..." >&2
    exit 1
fi

RUN_NAME=${RUN_NAME:-"agentdyn_${SUITES_TAG}_${TARGET_TYPE}"}

echo "============================================================"
echo "  Target model  : $TARGET_MODEL"
echo "  Suites        : $SUITES"
echo "  Train GPUs    : $TRAIN_GPUS ($NUM_GPUS GPU(s))"
echo "  Attacker      : $ATTACKER_MODEL"
echo "  Run name      : $RUN_NAME"
[ -n "$TARGET_DEFENSE" ] && echo "  Target defense: $TARGET_DEFENSE"
[ -n "$TRAIN_INJ" ] && echo "  Train inj tasks: $TRAIN_INJ"
[ -n "$EVAL_SUITES" ] && echo "  Eval suites   : $EVAL_SUITES"
[ -n "$EVAL_INJ" ] && echo "  Eval inj tasks : $EVAL_INJ"
[ -n "$TRAIN_USER" ] && echo "  Train user tasks: $TRAIN_USER"
[ -n "$EVAL_USER" ] && echo "  Eval user tasks : $EVAL_USER"
[ -n "$LEARNING_RATE" ] && echo "  Learning rate : $LEARNING_RATE"
[ -n "$NUM_TRAIN_EPOCHS" ] && echo "  Epochs        : $NUM_TRAIN_EPOCHS"
[ -n "$SAVE_STEPS" ] && echo "  Save steps    : $SAVE_STEPS"
[ -n "$SAVE_TOTAL_LIMIT" ] && echo "  Checkpoint cap: $SAVE_TOTAL_LIMIT"
echo "  Output dir    : $OUTPUT_DIR"
echo "============================================================"

if [ "$NEEDS_VLLM" -eq 1 ]; then
    TARGET_CHECK_URL="http://localhost:${TARGET_PORT}/v1/models"
    echo "Waiting for target vLLM at $TARGET_CHECK_URL ..."
    for i in $(seq 1 30); do
        if curl -sf "$TARGET_CHECK_URL" > /dev/null 2>&1; then
            echo "Target server ready."
            break
        fi
        echo "  Attempt $i/30 - not ready, sleeping 10s ..."
        sleep 10
    done
fi

EXTRA_ARGS=(
    --target_model "$TARGET_MODEL"
    --train_suites "$SUITES"
    --attacker_model_name_or_path "$ATTACKER_MODEL"
    --output_dir "$OUTPUT_DIR"
    --run_name "$RUN_NAME"
)
[ -n "$TARGET_MODEL_ID" ] && EXTRA_ARGS+=(--target_model_id "$TARGET_MODEL_ID")
[ -n "$TARGET_MODEL_URL" ] && EXTRA_ARGS+=(--target_model_url "$TARGET_MODEL_URL")
[ -n "$TARGET_DEFENSE" ] && EXTRA_ARGS+=(--target_defense "$TARGET_DEFENSE")
[ -n "$TRAIN_INJ" ] && EXTRA_ARGS+=(--train_injection_tasks "$TRAIN_INJ")
[ -n "$EVAL_SUITES" ] && EXTRA_ARGS+=(--eval_suites "$EVAL_SUITES")
[ -n "$EVAL_INJ" ] && EXTRA_ARGS+=(--eval_injection_tasks "$EVAL_INJ")
[ -n "$TRAIN_USER" ] && EXTRA_ARGS+=(--train_user_tasks "$TRAIN_USER")
[ -n "$EVAL_USER" ] && EXTRA_ARGS+=(--eval_user_tasks "$EVAL_USER")
[ -n "$RESUME_FROM_CHECKPOINT" ] && EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
[ -n "$LEARNING_RATE" ] && EXTRA_ARGS+=(--learning_rate "$LEARNING_RATE")
[ -n "$NUM_TRAIN_EPOCHS" ] && EXTRA_ARGS+=(--num_train_epochs "$NUM_TRAIN_EPOCHS")
[ -n "$SAVE_STEPS" ] && EXTRA_ARGS+=(--save_steps "$SAVE_STEPS")
[ -n "$SAVE_TOTAL_LIMIT" ] && EXTRA_ARGS+=(--save_total_limit "$SAVE_TOTAL_LIMIT")

TRAIN_CMD=(
    -m train
    --benchmark agentdyn
    --config_file configs/agentdyn.yaml
    "${EXTRA_ARGS[@]}"
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
echo "Training complete. Checkpoints: $OUTPUT_DIR"
echo "============================================================"
