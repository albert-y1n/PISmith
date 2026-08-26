#!/usr/bin/env bash
# Evaluate an attacker checkpoint against Qwen3.6-27B-SecOPD on AgentDyn.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENTDYN_ROOT=${AGENTDYN_ROOT:-/data/home/cmy5428/AgentDyn}
EVAL_PYTHON=${EVAL_PYTHON:-/data/home/cmy5428/anaconda3/envs/PISmith/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-/data/home/cmy5428/PISmith_Pro/envs/molt/bin/python}
VLLM_BIN_DIR="$(dirname "${VLLM_PYTHON}")"

CHECKPOINT=${1:-/data/home/cmy5428/PISmith/checkpoints/agentdyn_github_gpt56_terra_qwen3_4b_56_luna_lr5e6_add5ep/checkpoint-180}
CHECKPOINT_NAME=${CHECKPOINT##*/}
CHECKPOINT_PARENT=${CHECKPOINT%/*}
CHECKPOINT_PARENT=${CHECKPOINT_PARENT##*/}
RUN_LABEL=${RUN_LABEL:-${CHECKPOINT_PARENT}_${CHECKPOINT_NAME}}
TARGET_MODEL=${TARGET_MODEL:-pybbb/Qwen3.6-27B-SecOPD}
EVAL_SUITES=${EVAL_SUITES:-github,shopping,dailylife}
NUM_SAMPLES=${NUM_SAMPLES:-10}
ATTACKER_MAX_TOKENS=${ATTACKER_MAX_TOKENS:-4096}
TARGET_MAX_TOKENS=${TARGET_MAX_TOKENS:-32768}
TARGET_MAX_MODEL_LEN=${TARGET_MAX_MODEL_LEN:-65536}
TARGET_GPUS=${TARGET_GPUS:-0,1,2,3}
TARGET_TP_SIZE=${TARGET_TP_SIZE:-2}
TARGET_DP_SIZE=${TARGET_DP_SIZE:-2}
TARGET_PORT=${TARGET_PORT:-8010}
ATTACKER_GPU=${ATTACKER_GPU:-0}
ATTACKER_PORT=${ATTACKER_PORT:-8011}
MAX_WORKERS=${MAX_WORKERS:-16}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/eval_results/agentdyn_secopd_github-shopping-dailylife_pass10_${RUN_LABEL}}
INJECTIONS_CACHE=${INJECTIONS_CACHE:-${OUTPUT_DIR}/injections_cache.json}

LOG_DIR=${PROJECT_ROOT}/logs
TARGET_LOG=${LOG_DIR}/vllm_target_qwen36_27b_secopd_${RUN_LABEL}_port${TARGET_PORT}.log
ATTACKER_LOG=${LOG_DIR}/vllm_attacker_${RUN_LABEL}_port${ATTACKER_PORT}.log
RUN_LOG=${LOG_DIR}/eval_agentdyn_secopd_${RUN_LABEL}.log
AUDIT_FILE=${OUTPUT_DIR}/secopd_response_audit.jsonl

export PYTHONPATH="${PROJECT_ROOT}:${AGENTDYN_ROOT}/src:${PYTHONPATH:-}"
export PATH="${VLLM_BIN_DIR}:${PATH}"
export SECOPD_AUDIT_FILE="${AUDIT_FILE}"
export SECOPD_AUDIT_SAMPLE_LIMIT=${SECOPD_AUDIT_SAMPLE_LIMIT:-20}
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
ulimit -n 65536 2>/dev/null || true

target_pid=""
attacker_pid=""

cleanup() {
    if [[ -n "${attacker_pid}" ]] && kill -0 "${attacker_pid}" 2>/dev/null; then
        kill "${attacker_pid}" 2>/dev/null || true
    fi
    if [[ -n "${target_pid}" ]] && kill -0 "${target_pid}" 2>/dev/null; then
        kill "${target_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local name=$1
    local url=$2
    local pid=$3
    local logfile=$4
    for attempt in $(seq 1 360); do
        if curl -sf "${url}/models" >/dev/null 2>&1; then
            echo "${name} server ready (attempt ${attempt})."
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "${name} server exited; see ${logfile}." >&2
            return 1
        fi
        if (( attempt % 6 == 0 )); then
            echo "Waiting for ${name} server (${attempt}/360)..."
        fi
        sleep 10
    done
    echo "Timed out waiting for ${name} server; see ${logfile}." >&2
    return 1
}

if [[ ! -f "${INJECTIONS_CACHE}" ]]; then
    echo "Starting ${CHECKPOINT_NAME} attacker on GPU ${ATTACKER_GPU}..."
    CUDA_VISIBLE_DEVICES="${ATTACKER_GPU}" "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
        --model "${CHECKPOINT}" \
        --served-model-name "${CHECKPOINT_NAME}" \
        --port "${ATTACKER_PORT}" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
        --dtype bfloat16 \
        --trust-remote-code \
        >"${ATTACKER_LOG}" 2>&1 &
    attacker_pid=$!
    wait_for_server "attacker" "http://127.0.0.1:${ATTACKER_PORT}/v1" "${attacker_pid}" "${ATTACKER_LOG}"

    echo "Generating and caching ${EVAL_SUITES} attacker prompts..."
    "${EVAL_PYTHON}" -u -m eval.eval_agentdyn \
        --attacker_model "${CHECKPOINT}" \
        --attacker_server_url "http://127.0.0.1:${ATTACKER_PORT}/v1" \
        --eval_suites "${EVAL_SUITES}" \
        --num_samples "${NUM_SAMPLES}" \
        --max_tokens "${ATTACKER_MAX_TOKENS}" \
        --injections_cache "${INJECTIONS_CACHE}" \
        --generate_only

    kill "${attacker_pid}" 2>/dev/null || true
    wait "${attacker_pid}" 2>/dev/null || true
    attacker_pid=""
fi

echo "Starting two SecOPD TP=${TARGET_TP_SIZE} replicas on GPUs ${TARGET_GPUS} (DP=${TARGET_DP_SIZE})..."
CUDA_VISIBLE_DEVICES="${TARGET_GPUS}" "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${TARGET_MODEL}" \
    --served-model-name "${TARGET_MODEL}" \
    --port "${TARGET_PORT}" \
    --tensor-parallel-size "${TARGET_TP_SIZE}" \
    --data-parallel-size "${TARGET_DP_SIZE}" \
    --max-model-len "${TARGET_MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_WORKERS}" \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --trust-remote-code \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    >"${TARGET_LOG}" 2>&1 &
target_pid=$!
wait_for_server "target" "http://127.0.0.1:${TARGET_PORT}/v1" "${target_pid}" "${TARGET_LOG}"

echo "Smoke-testing SecOPD's custom input role..."
TARGET_BASE_URL="http://127.0.0.1:${TARGET_PORT}/v1" \
    TARGET_MODEL_ID="${TARGET_MODEL}" \
    TARGET_MAX_TOKENS="${TARGET_MAX_TOKENS}" \
    "${EVAL_PYTHON}" - <<'PY'
import os
from openai import OpenAI
from transformers import AutoTokenizer

messages = [
    {"role": "user", "content": "Reply with OK after reading the external result."},
    {"role": "input", "content": "External environment result: smoke-test."},
]
tokenizer = AutoTokenizer.from_pretrained(
    os.environ["TARGET_MODEL_ID"], trust_remote_code=True
)
rendered = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
assert "<|im_start|>input\n" in rendered, "input role missing from rendered template"
assert rendered.endswith("<think>\n"), "thinking generation prompt was not applied"

client = OpenAI(base_url=os.environ["TARGET_BASE_URL"], api_key="EMPTY", timeout=600)
response = client.chat.completions.create(
    model=os.environ["TARGET_MODEL_ID"],
    messages=messages,
    temperature=0,
    max_tokens=int(os.environ["TARGET_MAX_TOKENS"]),
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
choice = response.choices[0]
content = choice.message.content
assert content is not None
print("Smoke output:", content[:1000].replace("\n", " "))
assert choice.finish_reason != "length", "SecOPD smoke test was truncated"
assert "</think>" in content, "Thinking response did not close correctly"
print("SecOPD input-role/thinking smoke test passed.")
PY

echo "Starting AgentDyn ${EVAL_SUITES} pass@${NUM_SAMPLES} evaluation..."
"${EVAL_PYTHON}" -u -m eval.eval_agentdyn \
    --attacker_model "${CHECKPOINT}" \
    --target_model local \
    --target_model_id "${TARGET_MODEL}" \
    --target_model_url "http://127.0.0.1:${TARGET_PORT}/v1" \
    --target_adapter secopd \
    --target_max_tokens "${TARGET_MAX_TOKENS}" \
    --eval_suites "${EVAL_SUITES}" \
    --num_samples "${NUM_SAMPLES}" \
    --injections_cache "${INJECTIONS_CACHE}" \
    --max_workers "${MAX_WORKERS}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${RUN_LOG}"

echo "Results: ${OUTPUT_DIR}/eval_results.json"
echo "Detailed cases: ${OUTPUT_DIR}/eval_detailed.jsonl"
