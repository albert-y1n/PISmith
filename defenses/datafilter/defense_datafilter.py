import os
from ...llm import Model

DEFAULT_CONFIG = {
    "max_new_tokens": 2048,   # filter output length (lower = faster decode)
    "gpu_memory_utilization": 0.2,  # vLLM GPU memory fraction (local mode only; 0.2 ≈ 19 GiB, leaves room for other processes)
    "max_model_len": 20480,  # vLLM maximum sequence length
}

# Environment variable overrides:
# DATAFILTER_SERVER_URL=http://localhost:8011/v1  use a remote vLLM server (recommended during training to avoid VRAM conflicts)
# DATAFILTER_GPU_MEMORY=0.2   GPU memory fraction (local mode only, default 0.2 ≈ 19 GiB)
# DATAFILTER_MAX_MODEL_LEN=20480  maximum sequence length
# DATAFILTER_BATCH_SIZE=32  cross-sample batch size (max strings per vLLM call; larger = faster but more VRAM)

FILTER_MODEL = None        # vLLM LLM instance (local mode only)
FILTER_TOKENIZER = None    # tokenizer (both modes)
FILTER_MAX_MODEL_LEN = None
FILTER_SERVER_URL = None   # remote vLLM server URL
FILTER_MODE = None         # "local" or "remote"


def _ensure_model(config=None):
    """Ensure the filter model is ready (lazy-loaded on first call).

    Priority:
      1. DATAFILTER_SERVER_URL env var → remote mode (load tokenizer only, call via HTTP)
      2. Otherwise → local mode (load vLLM LLM instance)
    """
    global FILTER_MODEL, FILTER_TOKENIZER, FILTER_MAX_MODEL_LEN
    global FILTER_SERVER_URL, FILTER_MODE

    if FILTER_MODE is not None:
        return

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    max_len = int(os.getenv("DATAFILTER_MAX_MODEL_LEN", cfg.get("max_model_len", 20480)))
    server_url = os.getenv("DATAFILTER_SERVER_URL")

    if server_url:
        # ========== Remote mode: load tokenizer only, call remote vLLM server via HTTP ==========
        from transformers import AutoTokenizer

        FILTER_SERVER_URL = server_url.rstrip("/")
        hf_cache_dir = os.path.join(os.getenv("HF_HOME"), "hub") if os.getenv("HF_HOME") else None
        FILTER_TOKENIZER = AutoTokenizer.from_pretrained(
            "JoyYizhu/DataFilter",
            trust_remote_code=True,
            cache_dir=hf_cache_dir,
        )
        FILTER_MAX_MODEL_LEN = max_len
        FILTER_MODE = "remote"
        print(f"✅ DataFilter using remote vLLM server: {FILTER_SERVER_URL}")
    else:
        # ========== Local mode: load vLLM LLM instance ==========
        from vllm import LLM

        gpu_mem = float(os.getenv("DATAFILTER_GPU_MEMORY", str(cfg.get("gpu_memory_utilization", 0.2))))
        enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "1").lower() in ("1", "true", "yes")

        print(f"⏳ Loading DataFilter model with vLLM "
              f"(gpu_memory={gpu_mem:.0%}, max_model_len={max_len})...")
        FILTER_MODEL = LLM(
            model="JoyYizhu/DataFilter",
            dtype="bfloat16",
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_len,
            enforce_eager=enforce_eager,
        )
        FILTER_TOKENIZER = FILTER_MODEL.get_tokenizer()
        FILTER_MAX_MODEL_LEN = max_len
        FILTER_MODE = "local"
        print("✅ DataFilter model loaded (vLLM local)")


def datafilter(
    target_inst,
    context=None,
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG,
):
    _ensure_model(config)
    from .inference_utils import recursive_filter_vllm, recursive_filter_remote, parse

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    max_tokens = cfg.get("max_new_tokens", 512)

    if FILTER_MODE == "remote":
        filtered_context = recursive_filter_remote(
            parse(context), FILTER_SERVER_URL, FILTER_TOKENIZER, target_inst,
            max_tokens=max_tokens, max_model_len=FILTER_MAX_MODEL_LEN,
        )
    else:
        filtered_context = recursive_filter_vllm(
            parse(context), FILTER_MODEL, FILTER_TOKENIZER, target_inst,
            max_tokens=max_tokens, max_model_len=FILTER_MAX_MODEL_LEN,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{target_inst}\n\n{filtered_context}"},
    ]
    if llm is not None:
        response = llm.query(messages)
    else:
        response = "[PIArena] No LLM provided, response is not available."

    return {
        "response": response,
        "cleaned_context": filtered_context,
    }


def datafilter_batch(
    target_insts: list,
    contexts: list,
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG,
):
    """
    Batch version of the datafilter defense.

    Args:
        target_insts: list of user instructions
        contexts: list of contexts (one-to-one with target_insts)
        system_prompt: system prompt
        llm: target LLM
        config: configuration dict

    Returns:
        list of dict, each containing response and cleaned_context
    """
    if len(target_insts) != len(contexts):
        raise ValueError("target_insts and contexts must have the same length")

    _ensure_model(config)
    from .inference_utils import batch_filter_vllm, batch_filter_remote, parse

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    max_tokens = cfg.get("max_new_tokens", 512)
    batch_size = int(os.getenv("DATAFILTER_BATCH_SIZE", cfg.get("filter_batch_size", 32)))

    contexts_parsed = [parse(ctx) for ctx in contexts]
    if FILTER_MODE == "remote":
        filtered_contexts = batch_filter_remote(
            contexts_parsed, target_insts,
            FILTER_SERVER_URL, FILTER_TOKENIZER,
            max_tokens=max_tokens, max_model_len=FILTER_MAX_MODEL_LEN,
            batch_size=batch_size,
        )
    else:
        filtered_contexts = batch_filter_vllm(
            contexts_parsed, target_insts,
            FILTER_MODEL, FILTER_TOKENIZER,
            max_tokens=max_tokens, max_model_len=FILTER_MAX_MODEL_LEN,
            batch_size=batch_size,
        )

    # Batch query the target LLM
    results = []
    if llm is not None:
        messages_list = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{inst}\n\n{fctx}"},
            ]
            for inst, fctx in zip(target_insts, filtered_contexts)
        ]

        if hasattr(llm, 'batch_query'):
            responses = llm.batch_query(messages_list)
        else:
            responses = [llm.query(msgs) for msgs in messages_list]

        for resp, fctx in zip(responses, filtered_contexts):
            results.append({
                "response": resp,
                "cleaned_context": fctx,
            })
    else:
        for fctx in filtered_contexts:
            results.append({
                "response": "[PIArena] No LLM provided, response is not available.",
                "cleaned_context": fctx,
            })

    return results
