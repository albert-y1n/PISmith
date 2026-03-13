#!/usr/bin/env bash
# ============================================================
# train_agentdojo.sh — AgentDojo RL Attacker Training
#
# Trains an attacker to inject malicious instructions into
# AgentDojo agent pipelines.
#
# Usage:
#   bash scripts/train_agentdojo.sh [target_type] [suites] [train_gpus]
#
# target_type:
#   gpt4o-mini   GPT-4o-mini via Azure API (default)
#   gpt4o        GPT-4o via Azure API
#   local        Local vLLM model (needs GPU 0 for vLLM server)
#
# suites:
#   workspace    Single workspace suite (default)
#   all          workspace,banking,travel,slack
#   banking      Single banking suite
#
# ── Basic Examples ────────────────────────────────────────────
#   # GPT-4o-mini target on workspace
#   bash scripts/train_agentdojo.sh gpt4o-mini workspace "1,2,3"
#
#   # Train on all suites
#   bash scripts/train_agentdojo.sh gpt4o-mini all "1,2,3"
#
#   # Local vLLM target
#   TARGET_PORT=8000 \
#   bash scripts/train_agentdojo.sh local workspace "1,2,3"
#
#   # With AgentDojo native defense
#   TARGET_DEFENSE=tool_filter \
#   bash scripts/train_agentdojo.sh gpt4o-mini workspace "1,2,3"
#
# ── Partial Injection Task Selection ─────────────────────────
#   Train on injection_task_0 only, hold out the rest for eval:
#   TRAIN_INJ=injection_task_0 \
#   EVAL_SUITES=workspace \
#   EVAL_INJ=injection_task_1,injection_task_2,injection_task_3 \
#   bash scripts/train_agentdojo.sh gpt4o-mini workspace "1,2,3"
#
#   Train on workspace, eval on banking (cross-suite):
#   EVAL_SUITES=banking \
#   bash scripts/train_agentdojo.sh gpt4o-mini workspace "1,2,3"
#
#   Train on subset of user tasks:
#   TRAIN_USER=user_task_0,user_task_1,user_task_2 \
#   EVAL_SUITES=workspace \
#   EVAL_USER=user_task_3,user_task_4,user_task_5 \
#   bash scripts/train_agentdojo.sh gpt4o-mini workspace "1,2,3"
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: ${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/..:${PYTHONPATH:-}"

TARGET_TYPE=${1:-gpt4o-mini}
SUITES=${2:-workspace}
TRAIN_GPUS=${3:-"0,1,2,3"}

TARGET_PORT=${TARGET_PORT:-8000}
TARGET_DEFENSE=${TARGET_DEFENSE:-}      # Optional AgentDojo defense: tool_filter, etc.

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# ── Partial task selection (optional) ────────────────────────
# Set any of these env vars to restrict training / add a held-out eval set.
TRAIN_INJ=${TRAIN_INJ:-}      # e.g. "injection_task_0"
EVAL_SUITES=${EVAL_SUITES:-}  # e.g. "banking" or "workspace"
EVAL_INJ=${EVAL_INJ:-}        # e.g. "injection_task_1,injection_task_2"
TRAIN_USER=${TRAIN_USER:-}    # e.g. "user_task_0,user_task_1"
EVAL_USER=${EVAL_USER:-}      # e.g. "user_task_2,user_task_3"

ACCEL_CONFIG="configs/accelerate.yaml"
NUM_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)

# Expand "all" shorthand
if [ "$SUITES" = "all" ]; then
    SUITES="workspace,banking,travel,slack"
fi

# ── Target model config ───────────────────────────────────────
case "$TARGET_TYPE" in
  gpt4o-mini)
    TARGET_MODEL="gpt-4o-mini-2024-07-18"
    TARGET_MODEL_ID=""
    TARGET_MODEL_URL=""
    NEEDS_VLLM=0
    echo "Target: GPT-4o-mini-2024-07-18 (Azure/OpenAI API)"
    ;;
  gpt4o)
    TARGET_MODEL="gpt-4o-2024-05-13"
    TARGET_MODEL_ID=""
    TARGET_MODEL_URL=""
    NEEDS_VLLM=0
    echo "Target: GPT-4o-2024-05-13 (Azure/OpenAI API)"
    ;;
  local)
    TARGET_MODEL="local"
    TARGET_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
    TARGET_MODEL_URL="http://localhost:${TARGET_PORT}/v1"
    NEEDS_VLLM=1
    echo "Target: local vLLM (meta-llama/Llama-3.1-8B-Instruct)"
    ;;
  *)
    echo "Unknown target_type: $TARGET_TYPE"
    echo "Available: gpt4o-mini, gpt4o, local"
    exit 1
    ;;
esac

echo "============================================================"
echo "  Target model  : $TARGET_MODEL"
echo "  Suites        : $SUITES"
echo "  Train GPUs    : $TRAIN_GPUS ($NUM_GPUS GPU(s))"
[ -n "$TARGET_DEFENSE" ] && echo "  Target defense: $TARGET_DEFENSE"
[ -n "$TRAIN_INJ"  ]     && echo "  Train inj tasks: $TRAIN_INJ"
[ -n "$EVAL_SUITES" ]    && echo "  Eval suites   : $EVAL_SUITES"
[ -n "$EVAL_INJ"   ]     && echo "  Eval inj tasks : $EVAL_INJ"
[ -n "$TRAIN_USER" ]     && echo "  Train user tasks: $TRAIN_USER"
[ -n "$EVAL_USER"  ]     && echo "  Eval user tasks : $EVAL_USER"
echo "  Output dir    : checkpoints/agentdojo"
echo "============================================================"

# ── Wait for vLLM if using local target ──────────────────────
if [ "$NEEDS_VLLM" -eq 1 ]; then
    TARGET_CHECK_URL="http://localhost:${TARGET_PORT}/v1/models"
    echo "Waiting for target vLLM at $TARGET_CHECK_URL ..."
    for i in $(seq 1 30); do
        if curl -sf "$TARGET_CHECK_URL" > /dev/null 2>&1; then
            echo "Target server ready."
            break
        fi
        echo "  Attempt $i/30 — not ready, sleeping 10s ..."
        sleep 10
    done
fi

# ── Build CLI overrides ───────────────────────────────────────
EXTRA_ARGS=(
    --target_model "$TARGET_MODEL"
    --train_suites "$SUITES"
)
[ -n "$TARGET_MODEL_ID"  ] && EXTRA_ARGS+=(--target_model_id  "$TARGET_MODEL_ID")
[ -n "$TARGET_MODEL_URL" ] && EXTRA_ARGS+=(--target_model_url "$TARGET_MODEL_URL")
[ -n "$TARGET_DEFENSE"   ] && EXTRA_ARGS+=(--target_defense   "$TARGET_DEFENSE")
[ -n "$TRAIN_INJ"   ] && EXTRA_ARGS+=(--train_injection_tasks "$TRAIN_INJ")
[ -n "$EVAL_SUITES" ] && EXTRA_ARGS+=(--eval_suites           "$EVAL_SUITES")
[ -n "$EVAL_INJ"    ] && EXTRA_ARGS+=(--eval_injection_tasks  "$EVAL_INJ")
[ -n "$TRAIN_USER"  ] && EXTRA_ARGS+=(--train_user_tasks      "$TRAIN_USER")
[ -n "$EVAL_USER"   ] && EXTRA_ARGS+=(--eval_user_tasks       "$EVAL_USER")

# ── Launch training ───────────────────────────────────────────
TRAIN_CMD=(
    -m train
    --benchmark agentdojo
    --config_file configs/agentdojo.yaml
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
echo "Training complete. Checkpoints: checkpoints/agentdojo"
echo "============================================================"
