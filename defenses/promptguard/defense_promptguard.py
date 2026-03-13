from ...llm import Model
from transformers import pipeline
import os
import threading

DEFAULT_CONFIG = {
    "detector_batch_size": int(os.getenv("PROMPTGUARD_DETECTOR_BATCH_SIZE", 2)),
}

DETECTOR = None
_DETECTOR_LOCK = threading.Lock()


def get_detector():
    """Load Prompt-Guard via from_config + load_state_dict to avoid meta tensor bug (same as PIGuard)."""
    global DETECTOR
    with _DETECTOR_LOCK:
        if DETECTOR is not None:
            return DETECTOR
        import torch
        import logging
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download

        _prev_hf = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            for _logger in ("accelerate", "accelerate.accelerator", "accelerate.utils.modeling", "transformers.pipelines.base"):
                logging.getLogger(_logger).setLevel(logging.ERROR)

            gpu_id = int(os.getenv("PROMPTGUARD_DEVICE", "0"))
            device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            cache_dir = os.getenv("HF_HOME")
            model_id = "meta-llama/Prompt-Guard-86M"

            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir, local_files_only=True)
            model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)
            path = snapshot_download(model_id, cache_dir=cache_dir, local_files_only=True)
            state_dict = load_file(os.path.join(path, "model.safetensors"))
            model.load_state_dict(state_dict, strict=True)
            model = model.to(device)

            max_len = int(os.getenv("PROMPTGUARD_MAX_LENGTH", "2048"))
            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=gpu_id if torch.cuda.is_available() else -1,
                truncation=True,
                max_length=max_len,
            )
            DETECTOR = classifier
            return classifier
        finally:
            if _prev_hf is not None:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = _prev_hf
            else:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)


def promptguard(
    target_inst,
    context=None,
    system_prompt:str="You are a helpful assistant.",
    llm:Model=None,
    config=DEFAULT_CONFIG,
):
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = get_detector()

    max_len = int(os.getenv("PROMPTGUARD_MAX_LENGTH", "2048"))
    original_context = context
    truncated_context = context
    if context is not None and len(context) > max_len:
        truncated_context = context[:max_len]

    try:
        with _DETECTOR_LOCK:
            prediction = DETECTOR(truncated_context)[0]["label"]
        detect_flag = ("benign" not in prediction.lower())
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            if truncated_context is not None and len(truncated_context) > max_len // 2:
                truncated_context = truncated_context[:max_len // 2]
                try:
                    with _DETECTOR_LOCK:
                        prediction = DETECTOR(truncated_context)[0]["label"]
                    detect_flag = ("benign" not in prediction.lower())
                except RuntimeError:
                    detect_flag = True
            else:
                detect_flag = True
        else:
            raise

    if llm is not None:
        if detect_flag:
            response = "[Warning] PromptGuard detected injected prompt in the context."
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{target_inst}\n\n{original_context}"},
            ]
            response = llm.query(messages)
    else:
        response = "[PIArena] No LLM provided, response is not available."

    return {
        "response": response,
        "detect_flag": detect_flag,
    }


def promptguard_batch(
    target_insts: list,
    contexts: list,
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG,
):
    """
    Batch version of promptguard defense.
    Uses detector_batch_size for chunked inference (thread-safe via lock).
    """
    global DETECTOR

    if len(target_insts) != len(contexts):
        raise ValueError("target_insts and contexts must have the same length")

    if DETECTOR is None:
        DETECTOR = get_detector()

    max_len = int(os.getenv("PROMPTGUARD_MAX_LENGTH", "2048"))
    raw_contexts = []
    for ctx in contexts:
        if ctx is None:
            raw_contexts.append("")
        elif len(ctx) > max_len:
            raw_contexts.append(ctx[:max_len])
        else:
            raw_contexts.append(ctx)

    detector_batch_size = config.get("detector_batch_size", DEFAULT_CONFIG["detector_batch_size"])
    batch_preds = []
    for start in range(0, len(raw_contexts), detector_batch_size):
        chunk = raw_contexts[start : start + detector_batch_size]
        try:
            with _DETECTOR_LOCK:
                batch_preds.extend(DETECTOR(chunk))
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                for ctx in chunk:
                    ctx_short = ctx[:max_len // 2] if len(ctx) > max_len // 2 else ctx
                    try:
                        with _DETECTOR_LOCK:
                            pred = DETECTOR(ctx_short)
                        batch_preds.append(pred[0] if isinstance(pred, list) else pred)
                    except RuntimeError:
                        batch_preds.append({"label": "injection"})
            else:
                raise

    detect_flags = []
    for pred in batch_preds:
        if isinstance(pred, list):
            label = pred[0]["label"]
        else:
            label = pred["label"]
        detect_flags.append("benign" not in label.lower())

    results = []
    if llm is not None:
        detected_indices = [i for i, flag in enumerate(detect_flags) if flag]
        not_detected_indices = [i for i, flag in enumerate(detect_flags) if not flag]
        responses = [""] * len(target_insts)

        for i in detected_indices:
            responses[i] = "[Warning] PromptGuard detected injected prompt in the context."

        if not_detected_indices:
            messages_batch = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{target_insts[i]}\n\n{contexts[i]}"},
                ]
                for i in not_detected_indices
            ]
            if hasattr(llm, "batch_query"):
                batch_responses = llm.batch_query(messages_batch)
            else:
                batch_responses = [llm.query(msgs) for msgs in messages_batch]
            for idx, resp in zip(not_detected_indices, batch_responses):
                responses[idx] = resp

        for resp, flag in zip(responses, detect_flags):
            results.append({"response": resp, "detect_flag": flag})
    else:
        for flag in detect_flags:
            results.append(
                {
                    "response": "[PIArena] No LLM provided, response is not available.",
                    "detect_flag": flag,
                }
            )

    return results
