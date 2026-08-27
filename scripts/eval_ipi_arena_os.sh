#!/usr/bin/env bash
# Evaluate an RL attacker on IPI Arena OS.
# Usage: bash scripts/eval_ipi_arena_os.sh <checkpoint> [target_model] [categories] [num_samples]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/..:${PYTHONPATH:-}"

CHECKPOINT=${1:?Usage: $0 <checkpoint> [target_model] [categories] [num_samples]}
TARGET_MODEL=${2:-gpt-5.6-luna}
CATEGORIES=${3:-tool,coding,browser}
NUM_SAMPLES=${4:-10}
ATTACKER_MAX_TOKENS=${ATTACKER_MAX_TOKENS:-4096}

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
ATTACKER_GPU=${ATTACKER_GPU:-0}
ATTACKER_GPUS=${ATTACKER_GPUS:-${ATTACKER_GPU}}
ATTACKER_DP_SIZE=${ATTACKER_DP_SIZE:-1}
ATTACKER_PORT=${ATTACKER_PORT:-8001}
ATTACKER_URL=${ATTACKER_URL:-}
MAX_WORKERS=${MAX_WORKERS:-8}
GENERATION_WORKERS=${GENERATION_WORKERS:-32}
MAX_STEPS=${MAX_STEPS:-5}
OUTPUT_DIR=${OUTPUT_DIR:-eval_results/ipi_arena_os_${TARGET_MODEL}_pass${NUM_SAMPLES}_$(basename "${CHECKPOINT}")}
VLLM_ATTACKER_PID=""

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

cleanup() {
    if [[ -n "${VLLM_ATTACKER_PID}" ]] && kill -0 "${VLLM_ATTACKER_PID}" 2>/dev/null; then
        kill "${VLLM_ATTACKER_PID}" 2>/dev/null || true
        echo "Attacker vLLM stopped."
    fi
}
trap cleanup EXIT

mkdir -p logs
if [[ -z "${ATTACKER_URL}" ]]; then
    ATTACKER_URL="http://localhost:${ATTACKER_PORT}/v1"
    ATTACKER_LOG="logs/vllm_ipi_arena_os_$(basename "${CHECKPOINT}")_${ATTACKER_PORT}.log"
    VLLM_ARGS=()
    [[ "${ATTACKER_DP_SIZE}" != "1" ]] && VLLM_ARGS+=(--data-parallel-size "${ATTACKER_DP_SIZE}")

    CUDA_VISIBLE_DEVICES="${ATTACKER_GPUS}" python -m vllm.entrypoints.openai.api_server \
        --model "${CHECKPOINT}" \
        --port "${ATTACKER_PORT}" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.8 \
        --dtype bfloat16 \
        --trust-remote-code \
        "${VLLM_ARGS[@]}" >"${ATTACKER_LOG}" 2>&1 &
    VLLM_ATTACKER_PID=$!

    for attempt in $(seq 1 120); do
        if curl -sf "${ATTACKER_URL}/models" >/dev/null 2>&1; then
            break
        fi
        if ! kill -0 "${VLLM_ATTACKER_PID}" 2>/dev/null; then
            echo "ERROR: Attacker vLLM exited; see ${ATTACKER_LOG}." >&2
            exit 1
        fi
        [[ "${attempt}" -eq 120 ]] && { echo "ERROR: Attacker vLLM timed out." >&2; exit 1; }
        sleep 10
    done
fi

ARGS=(
    -m eval.eval_ipi_arena_os
    --attacker_model "${CHECKPOINT}"
    --attacker_server_url "${ATTACKER_URL}"
    --categories "${CATEGORIES}"
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
    --num_samples "${NUM_SAMPLES}"
    --max_tokens "${ATTACKER_MAX_TOKENS}"
    --max_workers "${MAX_WORKERS}"
    --generation_workers "${GENERATION_WORKERS}"
    --max_steps "${MAX_STEPS}"
    --output_dir "${OUTPUT_DIR}"
)

[[ -n "${TARGET_BASE_URL}" ]] && ARGS+=(--target_base_url "${TARGET_BASE_URL}")
[[ -n "${JUDGE_BASE_URL}" ]] && ARGS+=(--judge_base_url "${JUDGE_BASE_URL}")
[[ -n "${WORLDSIM_BASE_URL}" ]] && ARGS+=(--worldsim_base_url "${WORLDSIM_BASE_URL}")
[[ -n "${BEHAVIOR_IDS}" ]] && ARGS+=(--behavior_ids "${BEHAVIOR_IDS}")
[[ -n "${WAVES}" ]] && ARGS+=(--waves "${WAVES}")

echo "Target=${TARGET_MODEL}; Judge=${JUDGE_MODEL}; WorldSim=${WORLDSIM_MODEL}"
python "${ARGS[@]}"
