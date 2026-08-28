#!/usr/bin/env bash
# Train an RL attacker on IPI Arena OS.
# Usage: bash scripts/train_ipi_arena_os.sh [target_model] [categories] [train_gpus]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/..:${PYTHONPATH:-}"

TARGET_MODEL=${1:-gpt-5.6-luna}
CATEGORIES=${2:-tool,coding,browser}
TRAIN_GPUS=${3:-0,1,2,3}

infer_provider() {
    [[ "$1" == */* ]] && echo openrouter || echo openai
}

default_key_env() {
    [[ "$1" == "openrouter" ]] && echo OPENROUTER_API_KEY || echo OPENAI_API_KEY
}

TARGET_PROVIDER=${TARGET_PROVIDER:-$(infer_provider "$TARGET_MODEL")}
TARGET_API_KEY_ENV=${TARGET_API_KEY_ENV:-$(default_key_env "$TARGET_PROVIDER")}
TARGET_BASE_URL=${TARGET_BASE_URL:-}
TARGET_REASONING_EFFORT=${TARGET_REASONING_EFFORT:-medium}

JUDGE_MODEL=${JUDGE_MODEL:-gpt-5.6-luna}
JUDGE_PROVIDER=${JUDGE_PROVIDER:-$(infer_provider "$JUDGE_MODEL")}
JUDGE_API_KEY_ENV=${JUDGE_API_KEY_ENV:-$(default_key_env "$JUDGE_PROVIDER")}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-}
JUDGE_REASONING_EFFORT=${JUDGE_REASONING_EFFORT:-medium}

WORLDSIM_MODEL=${WORLDSIM_MODEL:-gpt-5.6-luna}
WORLDSIM_PROVIDER=${WORLDSIM_PROVIDER:-$(infer_provider "$WORLDSIM_MODEL")}
WORLDSIM_API_KEY_ENV=${WORLDSIM_API_KEY_ENV:-$(default_key_env "$WORLDSIM_PROVIDER")}
WORLDSIM_BASE_URL=${WORLDSIM_BASE_URL:-}
WORLDSIM_REASONING_EFFORT=${WORLDSIM_REASONING_EFFORT:-medium}

BEHAVIOR_IDS=${BEHAVIOR_IDS:-}
WAVES=${WAVES:-}
ATTACKER_MODEL=${ATTACKER_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/ipi_arena_os}
RUN_NAME=${RUN_NAME:-ipi_arena_os}
MAX_WORKERS=${MAX_WORKERS:-8}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
LEARNING_RATE=${LEARNING_RATE:-}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-}
SAVE_STEPS=${SAVE_STEPS:-}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-}

for provider_and_key in \
    "${TARGET_PROVIDER}:${TARGET_API_KEY_ENV}" \
    "${JUDGE_PROVIDER}:${JUDGE_API_KEY_ENV}" \
    "${WORLDSIM_PROVIDER}:${WORLDSIM_API_KEY_ENV}"; do
    provider=${provider_and_key%%:*}
    key_env=${provider_and_key#*:}
    if [[ "${provider}" != "vllm" && -z "${!key_env:-}" ]]; then
        echo "ERROR: ${key_env} is required by provider '${provider}'." >&2
        exit 1
    fi
done

if [[ "${TARGET_PROVIDER}" == "vllm" && -z "${TARGET_BASE_URL}" ]]; then
    echo "ERROR: Set TARGET_BASE_URL for a vLLM target." >&2
    exit 1
fi

NUM_GPUS=$(tr ',' '\n' <<<"${TRAIN_GPUS}" | wc -l)
ARGS=(
    -m train
    --benchmark ipi_arena_os
    --config_file configs/ipi_arena_os.yaml
    --categories "${CATEGORIES}"
    --attacker_model_name_or_path "${ATTACKER_MODEL}"
    --target_provider "${TARGET_PROVIDER}"
    --target_model "${TARGET_MODEL}"
    --target_api_key_env "${TARGET_API_KEY_ENV}"
    --target_reasoning_effort "${TARGET_REASONING_EFFORT}"
    --judge_provider "${JUDGE_PROVIDER}"
    --judge_model "${JUDGE_MODEL}"
    --judge_api_key_env "${JUDGE_API_KEY_ENV}"
    --judge_reasoning_effort "${JUDGE_REASONING_EFFORT}"
    --worldsim_provider "${WORLDSIM_PROVIDER}"
    --worldsim_model "${WORLDSIM_MODEL}"
    --worldsim_api_key_env "${WORLDSIM_API_KEY_ENV}"
    --worldsim_reasoning_effort "${WORLDSIM_REASONING_EFFORT}"
    --eval_max_workers "${MAX_WORKERS}"
    --output_dir "${OUTPUT_DIR}"
    --run_name "${RUN_NAME}"
)

[[ -n "${TARGET_BASE_URL}" ]] && ARGS+=(--target_base_url "${TARGET_BASE_URL}")
[[ -n "${JUDGE_BASE_URL}" ]] && ARGS+=(--judge_base_url "${JUDGE_BASE_URL}")
[[ -n "${WORLDSIM_BASE_URL}" ]] && ARGS+=(--worldsim_base_url "${WORLDSIM_BASE_URL}")
[[ -n "${BEHAVIOR_IDS}" ]] && ARGS+=(--behavior_ids "${BEHAVIOR_IDS}")
[[ -n "${WAVES}" ]] && ARGS+=(--waves "${WAVES}")
[[ -n "${RESUME_FROM_CHECKPOINT}" ]] && ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
[[ -n "${LEARNING_RATE}" ]] && ARGS+=(--learning_rate "${LEARNING_RATE}")
[[ -n "${NUM_TRAIN_EPOCHS}" ]] && ARGS+=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")
[[ -n "${SAVE_STEPS}" ]] && ARGS+=(--save_steps "${SAVE_STEPS}")
[[ -n "${SAVE_TOTAL_LIMIT}" ]] && ARGS+=(--save_total_limit "${SAVE_TOTAL_LIMIT}")

echo "============================================================"
echo "  Benchmark : IPI Arena OS (${CATEGORIES})"
echo "  Target    : ${TARGET_MODEL} (${TARGET_PROVIDER})"
echo "  Judge     : ${JUDGE_MODEL} (${JUDGE_PROVIDER})"
echo "  WorldSim  : ${WORLDSIM_MODEL} (${WORLDSIM_PROVIDER})"
echo "  Attacker  : ${ATTACKER_MODEL}"
echo "  GPUs      : ${TRAIN_GPUS}"
echo "  Output    : ${OUTPUT_DIR}"
echo "============================================================"

if [[ "${NUM_GPUS}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" python "${ARGS[@]}"
else
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" accelerate launch \
        --config_file configs/accelerate.yaml \
        --num_processes "${NUM_GPUS}" \
        "${ARGS[@]}"
fi
