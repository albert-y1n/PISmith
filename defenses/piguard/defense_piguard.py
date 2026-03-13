from ...llm import Model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import threading

DEFAULT_CONFIG = {
    # Limit forward pass batch size; large batches + long sequences cause DeBERTa attention OOM
    "detector_batch_size": int(os.getenv("PIGUARD_DETECTOR_BATCH_SIZE", 2)),
}

DETECTOR = None
_DETECTOR_LOCK = threading.Lock()
# PIGuard uses a DeBERTa encoder classification model; vLLM only supports decoder LMs.
# Run inference via transformers on the specified GPU (set via PIGUARD_DEVICE, default 0).
def get_detector():
    global DETECTOR
    with _DETECTOR_LOCK:
        if DETECTOR is not None:
            return DETECTOR
        import torch
        import logging
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download

        # Suppress HuggingFace Hub progress bars
        _prev_hf = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            # Suppress repeated warnings (tie_weights, device placement, etc.)
            for _logger in ("accelerate", "accelerate.accelerator", "accelerate.utils.modeling", "transformers.pipelines.base"):
                logging.getLogger(_logger).setLevel(logging.ERROR)

            gpu_id = int(os.getenv("PIGUARD_DEVICE", "0"))
            device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            cache_dir = os.getenv("HF_HOME")
            model_id = "leolee99/PIGuard"

            # Avoid meta tensor bug: use from_config + load_state_dict to bypass
            # accelerate's meta loading path. local_files_only=True avoids Hub fetches
            # when the model is already cached.
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir, local_files_only=True)
            model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True)
            path = snapshot_download(model_id, cache_dir=cache_dir, local_files_only=True)
            state_dict = load_file(os.path.join(path, "model.safetensors"))
            model.load_state_dict(state_dict, strict=True)
            model = model.to(device)

            # Large max_length (e.g. 10240) causes O(n²) DeBERTa attention OOM; 2048 is safe
            max_len = int(os.getenv("PIGUARD_MAX_LENGTH", "2048"))
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

def piguard(
    target_inst,
    context=None,
    system_prompt:str="You are a helpful assistant.",
    llm:Model=None,
    config=DEFAULT_CONFIG,
):
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = get_detector()

    result = DETECTOR(context)[0]
    prediction = result["label"]
    score = result["score"]

    if prediction == "benign":
        detect_flag = False
    elif prediction == "injection":
        detect_flag = True
    else:
        raise ValueError(f"Unknown prediction: {prediction}")

    if llm is not None:
        if detect_flag:
            response = "[Warning] PIGuard detected injected prompt in the context."
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{target_inst}\n\n{context}"},
            ]
            response = llm.query(messages)
    else:
        response = "[PIArena] No LLM provided, response is not available."

    return {
        "response": response,
        "detect_flag": detect_flag,
        "score": score,
    }


def piguard_batch(
    target_insts: list,
    contexts: list,
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG,
):
    """
    Batch version of the piguard defense.

    Args:
        target_insts: list of user instructions
        contexts: list of contexts
        system_prompt: system prompt
        llm: target LLM (should support batch_query)
        config: configuration dict (reserved for future use)

    Returns:
        list of dict, each containing response, detect_flag, and score
    """
    global DETECTOR

    if len(target_insts) != len(contexts):
        raise ValueError("target_insts and contexts must have the same length")

    # Initialize detector once
    if DETECTOR is None:
        DETECTOR = get_detector()

    # Replace None contexts with empty strings
    raw_contexts = [ctx if ctx is not None else "" for ctx in contexts]

    # Run in small batches to avoid DeBERTa CUDA OOM on large/long inputs
    detector_batch_size = config.get("detector_batch_size", DEFAULT_CONFIG["detector_batch_size"])
    batch_preds = []
    for start in range(0, len(raw_contexts), detector_batch_size):
        chunk = raw_contexts[start : start + detector_batch_size]
        batch_preds.extend(DETECTOR(chunk))

    detect_flags = []
    scores = []
    for pred in batch_preds:
        # Handle both single-label and multi-candidate output formats
        if isinstance(pred, list):
            label = pred[0]["label"]
            score = pred[0]["score"]
        else:
            label = pred["label"]
            score = pred["score"]

        if label == "benign":
            detect_flags.append(False)
        elif label == "injection":
            detect_flags.append(True)
        else:
            raise ValueError(f"Unknown prediction: {label}")
        scores.append(score)

    # Generate responses
    results = []
    if llm is not None:
        detected_indices = [i for i, flag in enumerate(detect_flags) if flag]
        not_detected_indices = [i for i, flag in enumerate(detect_flags) if not flag]

        responses = [""] * len(target_insts)

        # Return warning for detected injections
        for i in detected_indices:
            responses[i] = "[Warning] PIGuard detected injected prompt in the context."

        # Batch query the underlying LLM for non-detected samples
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

        for resp, flag, score in zip(responses, detect_flags, scores):
            results.append({"response": resp, "detect_flag": flag, "score": score})
    else:
        # No underlying LLM; return detection results only
        for flag, score in zip(detect_flags, scores):
            results.append(
                {
                    "response": "[PIArena] No LLM provided, response is not available.",
                    "detect_flag": flag,
                    "score": score,
                }
            )

    return results
