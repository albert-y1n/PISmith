import gc
import torch
from ...llm import Model
from .AttentionTracker.utils import create_model
from .AttentionTracker.detector.attn import AttentionDetector


def _clear_cuda_cache():
    """Clear the CUDA cache to free memory."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


DEFAULT_CONFIG = {
    "model_info": {
        "provider": "attn-hf",
        "name": "qwen-attn",
        "model_id": "Qwen/Qwen2-1.5B-Instruct",
    },
    "params": {
        "temperature": 0.1,
        "max_output_tokens": 32,
        "important_heads": [
            [10, 6], [11, 0], [11, 2], [11, 8], [11, 9], [11, 11],
            [12, 8], [13, 10], [14, 8], [15, 7], [15, 11],
            [17, 0], [18, 9], [19, 7],
        ],
    },
}

DETECTOR = None
def get_detector(model_config=DEFAULT_CONFIG):
    model = create_model(config=model_config)
    model.print_model_info()
    detector = AttentionDetector(model)
    print("===================")
    print(f"Using detector: {detector.name}")
    return detector

def attentiontracker(
    target_inst,
    context=None,
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG,
):
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = get_detector(model_config=config)

    detect_flag, _ = DETECTOR.detect(context)

    if llm is not None:
        if detect_flag:
            response = "[Warning] AttentionTracker detected injected prompt in the context."
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
    }


def attentiontracker_batch(
    target_insts: list,
    contexts: list,
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG,
):
    """
    Batch version of the attentiontracker defense.

    Args:
        target_insts: list of user instructions
        contexts: list of contexts
        system_prompt: system prompt
        llm: target LLM (should support batch_query)
        config: model configuration

    Returns:
        list of dict, each containing response and detect_flag
    """
    global DETECTOR

    if len(target_insts) != len(contexts):
        raise ValueError("target_insts and contexts must have the same length")

    # Initialize detector once
    if DETECTOR is None:
        DETECTOR = get_detector(model_config=config)

    # AttentionDetector currently only supports single-sample detect; iterate here
    # but expose a batch interface so the training pipeline can use it uniformly.
    detect_flags = []
    for i, ctx in enumerate(contexts):
        flag, _ = DETECTOR.detect(ctx)
        detect_flags.append(flag)
        # Clear CUDA cache every 10 samples to prevent memory accumulation
        if (i + 1) % 10 == 0:
            _clear_cuda_cache()

    results = []
    if llm is not None:
        detected_indices = [i for i, flag in enumerate(detect_flags) if flag]
        not_detected_indices = [i for i, flag in enumerate(detect_flags) if not flag]

        responses = [""] * len(target_insts)

        # Return warning for detected injections
        for i in detected_indices:
            responses[i] = "[Warning] AttentionTracker detected injected prompt in the context."

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

    # Clear cache after batch processing
    _clear_cuda_cache()

    return results
