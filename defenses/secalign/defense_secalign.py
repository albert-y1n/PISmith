import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Union

from ...llm import Model

DEFAULT_CONFIG_SECALIGN = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",  # base model
    "lora_adapter": "facebook/Meta-SecAlign-8B",        # LoRA adapter
}

# Global SecAlign model instance (supports vLLM or transformers)
SECALIGN_MODEL = None
SECALIGN_USE_VLLM = None  # None = auto-detect, True = force vLLM, False = force transformers


class SecAlignVLLMModel:
    """
    SecAlign vLLM model wrapper.
    Loads the base model + LoRA adapter via vLLM for inference.
    """

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        lora_adapter: str = "facebook/Meta-SecAlign-8B",
        tensor_parallel_size: int = None,
        gpu_memory_utilization: float = 0.35,
        max_model_len: int = 20480,
        max_num_seqs: Optional[int] = None,
    ):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.base_model = base_model
        self.lora_adapter = lora_adapter
        self.max_model_len = max_model_len
        self.SamplingParams = SamplingParams
        self.LoRARequest = LoRARequest

        # Determine tensor parallel size from visible GPUs
        if tensor_parallel_size is None:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                tensor_parallel_size = len([g.strip() for g in cuda_visible.split(",") if g.strip()])
            else:
                import torch
                tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

        hf_cache_dir = os.path.join(os.getenv("HF_HOME"), "hub") if os.getenv("HF_HOME") else None

        print(f"Loading SecAlign with vLLM...")
        print(f"   Base model: {base_model}")
        print(f"   LoRA adapter: {lora_adapter}")
        print(f"   Tensor parallel size: {tensor_parallel_size} GPU(s)")
        print(f"   GPU memory utilization: {gpu_memory_utilization:.0%}")
        print(f"   Max model length: {max_model_len}")
        if max_num_seqs is not None:
            print(f"   Max number of sequences: {max_num_seqs}")

        enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "1").lower() in ("1", "true", "yes")

        # When called inside an accelerate distributed training process, vLLM v1's LLM()
        # spawns an EngineCore subprocess that inherits MASTER_ADDR/MASTER_PORT etc.,
        # causing TCP connection timeouts.
        # Fix: clear distributed env vars + disable vLLM V1 multi-processing mode.
        dist_env_keys = [
            "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE",
            "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
            "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
            "TORCHELASTIC_RUN_ID", "OMP_NUM_THREADS",
        ]
        saved_env = {}
        for key in dist_env_keys:
            if key in os.environ:
                saved_env[key] = os.environ.pop(key)
        saved_v1_mp = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        # Load vLLM model with LoRA support
        try:
            self.model = LLM(
                model=base_model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                enable_lora=True,
                max_lora_rank=64,  # LoRA rank used by SecAlign
                download_dir=hf_cache_dir,
                trust_remote_code=True,
                enforce_eager=enforce_eager,
            )
        finally:
            os.environ.update(saved_env)
            if saved_v1_mp is not None:
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = saved_v1_mp
            else:
                os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)

        self.tokenizer = self.model.get_tokenizer()

        # Create LoRA request (reused for every inference call)
        self.lora_request = LoRARequest(
            lora_name="secalign",
            lora_int_id=1,
            lora_path=lora_adapter,
        )

        print(f"SecAlign loaded (vLLM + LoRA, supports batch inference)")

    def _format_messages(self, messages: Union[str, List[dict]]) -> str:
        """Format a messages list into a prompt string."""
        if isinstance(messages, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages}
            ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple concatenation
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "input":
                    prompt += f"Input: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "

        return prompt

    def query(
        self,
        messages: Union[str, List[dict]],
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """Single-sample query."""
        results = self.batch_query(
            [messages],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            **kwargs
        )
        return results[0]

    def batch_query(
        self,
        messages_list: List[Union[str, List[dict]]],
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """Batch query."""
        if not messages_list:
            return []

        # Format all messages into prompts
        prompts = [self._format_messages(m) for m in messages_list]

        # Truncate prompts that exceed the model's context window
        max_input_tokens = self.max_model_len - max_new_tokens - 100  # safety margin
        truncated_prompts = []
        for prompt in prompts:
            prompt_tokens = self.tokenizer.encode(prompt)
            if len(prompt_tokens) > max_input_tokens:
                print(f"SecAlign: Truncating input from {len(prompt_tokens)} to {max_input_tokens} tokens")
                prompt_tokens = prompt_tokens[:max_input_tokens]
                prompt = self.tokenizer.decode(prompt_tokens)
            truncated_prompts.append(prompt)

        effective_temp = temperature if do_sample else 0.0
        sampling_params = self.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=effective_temp,
            top_p=top_p if do_sample else 1.0,
        )

        # Run inference with LoRA
        outputs = self.model.generate(
            truncated_prompts,
            sampling_params,
            lora_request=self.lora_request,
        )

        results = [output.outputs[0].text for output in outputs]
        return results


class SecAlignTransformersModel:
    """
    SecAlign transformers model wrapper.
    Loads the base model + LoRA adapter via PEFT for inference.
    """

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        lora_adapter: str = "facebook/Meta-SecAlign-8B",
    ):
        from peft import PeftModel

        self.base_model_name = base_model
        self.lora_adapter = lora_adapter

        hf_cache_dir = os.path.join(os.getenv("HF_HOME"), "hub") if os.getenv("HF_HOME") else None

        print(f"Loading SecAlign with transformers + PEFT...")
        print(f"   Base model: {base_model}")
        print(f"   LoRA adapter: {lora_adapter}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
            cache_dir=hf_cache_dir,
        )

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
            cache_dir=hf_cache_dir,
        )

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(
            base,
            lora_adapter,
            token=os.getenv("HF_TOKEN"),
            cache_dir=hf_cache_dir,
        )
        self.model.eval()

        print(f"SecAlign loaded (transformers + PEFT)")

    def query(
        self,
        messages: Union[str, List[dict]],
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """Single-sample query."""
        if isinstance(messages, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages}
            ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        attention_mask = torch.ones_like(input_ids).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=do_sample if temperature > 0 else False,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][len(input_ids[0]):],
            skip_special_tokens=True
        )
        return generated_text

    def batch_query(
        self,
        messages_list: List[Union[str, List[dict]]],
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """Batch query using left padding."""
        if not messages_list:
            return []

        # Format messages
        formatted_messages = []
        for msgs in messages_list:
            if isinstance(msgs, str):
                msgs = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msgs}
                ]
            formatted_messages.append(msgs)

        # Use left padding for batch generation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            # Apply chat template
            input_texts = [
                self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                for msgs in formatted_messages
            ]

            # Batch tokenize
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.model.device)

            # Batch generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=do_sample if temperature > 0 else False,
                    top_p=top_p if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode outputs
            results = []
            for i, output in enumerate(outputs):
                generated_ids = output[inputs.input_ids.shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                results.append(generated_text)

            return results

        finally:
            self.tokenizer.padding_side = original_padding_side


def _get_secalign_model(config=DEFAULT_CONFIG_SECALIGN, use_vllm: bool = None):
    """
    Get or initialize the SecAlign model.

    SecAlign is a LoRA adapter that requires the Llama-3.1-8B-Instruct base model.

    Args:
        config: configuration dict containing base_model and lora_adapter
        use_vllm: whether to use vLLM. None = auto-detect (prefer vLLM)

    Returns:
        model instance (SecAlignVLLMModel or SecAlignTransformersModel)
    """
    global SECALIGN_MODEL, SECALIGN_USE_VLLM

    if SECALIGN_MODEL is not None:
        return SECALIGN_MODEL

    base_model = config.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
    lora_adapter = config.get("lora_adapter", "facebook/Meta-SecAlign-8B")

    # Backwards-compatibility with old config format
    if "model_name_or_path" in config and "lora_adapter" not in config:
        lora_adapter = config["model_name_or_path"]

    env_use_vllm = os.getenv("SECALIGN_USE_VLLM", "1")
    if use_vllm is None:
        use_vllm = env_use_vllm.lower() in ("1", "true", "yes")

    SECALIGN_USE_VLLM = use_vllm

    if use_vllm:
        try:
            gpu_memory = float(os.getenv("SECALIGN_VLLM_GPU_MEMORY", "0.35"))
            max_model_len = int(os.getenv("SECALIGN_VLLM_MAX_MODEL_LEN", "20480"))
            tensor_parallel_size = os.getenv("SECALIGN_VLLM_TP_SIZE")
            tensor_parallel_size = int(tensor_parallel_size) if tensor_parallel_size else None

            # Conservative max_num_seqs to reduce VRAM pressure
            max_num_seqs_env = os.getenv("SECALIGN_VLLM_MAX_NUM_SEQS")
            if max_num_seqs_env:
                try:
                    max_num_seqs = int(max_num_seqs_env)
                except ValueError:
                    print(f"Could not parse SECALIGN_VLLM_MAX_NUM_SEQS={max_num_seqs_env!r}, using default 256")
                    max_num_seqs = 256
            else:
                # More conservative than vLLM's default of 1024 to reduce VRAM usage
                max_num_seqs = 256

            SECALIGN_MODEL = SecAlignVLLMModel(
                base_model=base_model,
                lora_adapter=lora_adapter,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
            )
            return SECALIGN_MODEL
        except Exception as e:
            print(f"Failed to load SecAlign with vLLM: {e}")
            print(f"   Falling back to transformers + PEFT...")

    # Fallback to transformers + PEFT
    SECALIGN_MODEL = SecAlignTransformersModel(
        base_model=base_model,
        lora_adapter=lora_adapter,
    )
    SECALIGN_USE_VLLM = False
    return SECALIGN_MODEL


def _build_secalign_messages(target_inst: str, context: Optional[str], system_prompt: str) -> List[dict]:
    """Build the message format expected by SecAlign."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": target_inst},
    ]
    if context is not None:
        messages.append({"role": "input", "content": context})
    return messages


def secalign(
    target_inst,
    context=None,
    system_prompt=None,
    llm: Model = None,
    config=DEFAULT_CONFIG_SECALIGN,
):
    """
    SecAlign defense: runs inference through the defense-aligned LoRA model.

    Args:
        target_inst: user instruction
        context: context (may contain injected content)
        system_prompt: system prompt
        llm: unused; SecAlign uses its own model
        config: configuration dict

    Returns:
        dict: {"response": generated response}
    """
    system_prompt = system_prompt if system_prompt is not None else "You are a helpful assistant."

    secalign_model = _get_secalign_model(config)
    secalign_messages = _build_secalign_messages(target_inst, context, system_prompt)
    secalign_response = secalign_model.query(secalign_messages)

    return {
        "response": secalign_response,
    }


def secalign_batch(
    target_insts: List[str],
    contexts: List[Optional[str]],
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG_SECALIGN,
) -> List[dict]:
    """
    Batch version of the SecAlign defense: uses vLLM for parallel inference.

    Args:
        target_insts: list of user instructions
        contexts: list of contexts (may contain injected content)
        system_prompt: system prompt
        llm: unused; SecAlign uses its own model
        config: configuration dict

    Returns:
        List[dict]: [{"response": response1}, {"response": response2}, ...]
    """
    global SECALIGN_MODEL, SECALIGN_USE_VLLM

    if not target_insts:
        return []

    if contexts is None:
        contexts = [None] * len(target_insts)
    elif len(contexts) != len(target_insts):
        raise ValueError(f"Length mismatch: target_insts={len(target_insts)}, contexts={len(contexts)}")

    secalign_model = _get_secalign_model(config)

    # Build batch messages
    messages_list = [
        _build_secalign_messages(inst, ctx, system_prompt)
        for inst, ctx in zip(target_insts, contexts)
    ]

    if hasattr(secalign_model, 'batch_query'):
        try:
            responses = secalign_model.batch_query(messages_list)
        except Exception as e:
            error_msg = str(e).lower()
            # Detect fatal vLLM errors (CUDA crash, process death, etc.) and fall back
            if any(kw in error_msg for kw in ['cuda', 'enginecore', 'nccl', 'illegal instruction', 'worker', 'cancelled']):
                print(f"SecAlign vLLM fatal error: {e}")
                print(f"   Attempting fallback to transformers mode...")

                # Clean up the crashed vLLM model
                try:
                    if hasattr(SECALIGN_MODEL, 'model'):
                        del SECALIGN_MODEL.model
                    del SECALIGN_MODEL
                except:
                    pass
                SECALIGN_MODEL = None

                # Free GPU memory
                try:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass

                # Force transformers on retry
                SECALIGN_USE_VLLM = False
                os.environ["SECALIGN_USE_VLLM"] = "0"

                secalign_model = _get_secalign_model(config, use_vllm=False)

                if hasattr(secalign_model, 'batch_query'):
                    responses = secalign_model.batch_query(messages_list)
                else:
                    responses = [secalign_model.query(msgs) for msgs in messages_list]
            else:
                raise
    else:
        responses = [secalign_model.query(msgs) for msgs in messages_list]

    return [{"response": resp} for resp in responses]


def reset_secalign_model(force_transformers: bool = False):
    """
    Reset the SecAlign model (for testing or configuration changes).

    Args:
        force_transformers: if True, force transformers backend on next load
    """
    global SECALIGN_MODEL, SECALIGN_USE_VLLM

    if SECALIGN_MODEL is not None:
        try:
            if hasattr(SECALIGN_MODEL, 'model'):
                del SECALIGN_MODEL.model
            del SECALIGN_MODEL
        except:
            pass

        # Free GPU memory
        try:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

    SECALIGN_MODEL = None

    if force_transformers:
        SECALIGN_USE_VLLM = False
        os.environ["SECALIGN_USE_VLLM"] = "0"
    else:
        SECALIGN_USE_VLLM = None
