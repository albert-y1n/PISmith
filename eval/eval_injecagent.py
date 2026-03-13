#!/usr/bin/env python3
"""
Evaluation Script for InjecAgent RL Attacker (RL-trained models only).

Evaluates a trained attacker model on the InjecAgent benchmark by:
1. Loading attack prompts from the trained attacker model
2. Injecting them into target agent's tool observation
3. Checking if the target agent calls the attacker's tool

Supports vLLM and tool-calling (GPT/Azure/Gemini) target models.

Usage:
    # Evaluate with vLLM target
    python -m eval.eval_injecagent \\
        --attacker_model checkpoints/injecagent/checkpoint-500 \\
        --attacker_server_url http://localhost:8001/v1 \\
        --target_model_url http://localhost:8000/v1 \\
        --target_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \\
        --data_path data/InjecAgent/test.json \\
        --output_dir eval_results/injecagent

    # Evaluate with OpenAI API target
    python -m eval.eval_injecagent \\
        --attacker_model checkpoints/injecagent/checkpoint-500 \\
        --attacker_server_url http://localhost:8001/v1 \\
        --use_openai_target \\
        --target_model_name_or_path gpt-4o-mini \\
        --target_openai_config configs/openai.yaml \\
        --data_path data/InjecAgent/test.json \\
        --output_dir eval_results/injecagent
"""

import os
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI, AzureOpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from core.utils import (
    injecagent_get_tool_dict,
    INJECAGENT_SYS_PROMPT,
    INJECAGENT_USER_PROMPT,
    load_openai_client,
    set_random_seed,
    to_anthropic_tools,
    to_anthropic_tool_call,
    to_anthropic_tool_result,
    anthropic_completion_to_openai,
)
from benchmarks.injecagent.output_parsing import (
    evaluate_output_finetuned,
    evaluate_output_prompted,
)
from benchmarks.injecagent.dataset import InjecAgentDataset


def generate_attacks_vllm(
    client: OpenAI,
    model_name: str,
    tokenizer,
    prompts: List[List[Dict]],
    max_tokens: int = 512,
    temperature: float = 0.7,
    num_samples: int = 1,
) -> List[List[str]]:
    """Generate attack prompts from a vLLM-served attacker model."""
    input_texts = [
        tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]

    try:
        resp = client.completions.create(
            model=model_name,
            prompt=input_texts,
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            n=num_samples,
        )
        results = [[] for _ in prompts]
        for i in range(len(prompts)):
            for j in range(num_samples):
                idx = i * num_samples + j
                if idx < len(resp.choices):
                    text = resp.choices[idx].text
                    # Extract from <prompt> tags if present
                    if "<prompt>" in text and "</prompt>" in text:
                        text = text.split("<prompt>", 1)[1].split("</prompt>", 1)[0].strip()
                    results[i].append(text)
                else:
                    results[i].append("")
        return results
    except Exception as e:
        print(f"Attacker vLLM error: {e}")
        return [[""] * num_samples for _ in prompts]


def generate_attacks_local(
    model,
    tokenizer,
    prompts: List[List[Dict]],
    max_tokens: int = 512,
    temperature: float = 0.7,
    num_samples: int = 1,
    batch_size: int = 4,
) -> List[List[str]]:
    """Generate attack prompts from a local model."""
    input_texts = [
        tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]

    device = model.device
    tok = tokenizer
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    results = [[] for _ in prompts]
    for _ in range(num_samples):
        for start in range(0, len(input_texts), batch_size):
            chunk = input_texts[start : start + batch_size]
            old_side = getattr(tok, "padding_side", "right")
            tok.padding_side = "left"
            encoded = tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(device)
            tok.padding_side = old_side

            with torch.no_grad():
                outputs = model.generate(**encoded, **gen_kwargs)

            prompt_len = encoded["input_ids"].size(1)
            for i, output in enumerate(outputs):
                text = tok.decode(output[prompt_len:], skip_special_tokens=True)
                if "<prompt>" in text and "</prompt>" in text:
                    text = text.split("<prompt>", 1)[1].split("</prompt>", 1)[0].strip()
                results[start + i].append(text)

    return results


def fetch_with_retries(client, messages, tools, model_name, max_retries=5, reasoning_effort="minimal"):
    """Fetch API completion with retries."""
    try:
        from anthropic import RateLimitError
    except ImportError:
        RateLimitError = Exception

    if to_anthropic_tools and "anthropic" in model_name.lower():
        messages = list(messages)
        messages[2] = to_anthropic_tool_call(messages[2])
        messages[3] = to_anthropic_tool_result(messages[3])

    for attempt in range(max_retries):
        try:
            if to_anthropic_tools and "anthropic" in model_name.lower():
                completion = client.messages.create(
                    model=model_name,
                    system=messages[0]["content"],
                    messages=messages[1:],
                    tools=to_anthropic_tools(tools),
                    max_tokens=1024,
                )
                return anthropic_completion_to_openai(completion)
            else:
                kwargs = {"model": model_name, "messages": messages, "tools": tools}
                if "gpt-5" in model_name.lower():
                    kwargs["reasoning_effort"] = reasoning_effort
                return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                time.sleep(10**attempt)
            else:
                raise
        except Exception as e:
            if attempt >= max_retries - 1:
                print(f"API error: {e}")
                return None


def evaluate_single_sample(
    sample: Dict,
    attack_prompt: str,
    target_client,
    target_model_name: str,
    target_tokenizer,
    tool_dict: Dict,
    tool_dict_gpt: Dict,
    use_openai_target: bool,
    target_max_tokens: int,
    reasoning_effort: str = "minimal",
) -> bool:
    """Evaluate a single attack prompt on one sample. Returns True if attack succeeded."""
    attacker_tools = sample["Attacker Tools"]
    user_tool = sample["User Tool"]
    tool_resp_template = sample["Tool Response Template"]
    observation = tool_resp_template.replace("<Attacker Instruction>", attack_prompt)

    if use_openai_target:
        available_tool_names = [user_tool] + attacker_tools
        available_tools = [tool_dict_gpt[t] for t in available_tool_names]
        messages = [
            {"role": "system", "content": INJECAGENT_SYS_PROMPT},
            {"role": "user", "content": sample["User Instruction"]},
        ]
        call_id = "call_injecagent_eval"
        messages.append(
            ChatCompletionMessage(
                content=None, role="assistant",
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id=call_id,
                        function=Function(arguments=sample["Tool Parameters"], name=user_tool),
                        type="function",
                    )
                ],
            )
        )
        messages.append(
            {"tool_call_id": call_id, "role": "tool", "name": user_tool, "content": observation}
        )
        completion = fetch_with_retries(
            target_client, messages, available_tools, target_model_name,
            reasoning_effort=reasoning_effort,
        )
        if completion is None:
            return False
        msg = completion.choices[0].message
        text = msg.content or ""
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls and hasattr(tool_calls[0], "model_dump"):
            tool_calls = [t.model_dump() for t in tool_calls]
        result = evaluate_output_finetuned(text, attacker_tools[0], user_tool, tool_calls)
    else:
        available_tool_names = [user_tool] + attacker_tools
        available_tool_descriptions = json.dumps(
            [tool_dict[t] for t in available_tool_names], indent=True
        )
        agent_scratchpad = (
            f"\n{sample['Thought']}\nAction: {user_tool}\n"
            f"Action Input: {sample['Tool Parameters']}\nObservation: {observation}\n"
        )
        user_prompt = INJECAGENT_USER_PROMPT.format(
            available_tool_descriptions=available_tool_descriptions,
            input=sample["User Instruction"],
            available_tool_names=available_tool_names,
            agent_scratchpad=agent_scratchpad,
        )

        is_secalign = "secalign" in target_model_name.lower()
        if is_secalign:
            msgs = [[{"role": "user", "content": INJECAGENT_SYS_PROMPT}, {"role": "input", "content": user_prompt}]]
        else:
            msgs = [[{"role": "system", "content": INJECAGENT_SYS_PROMPT}, {"role": "user", "content": user_prompt}]]

        prompts = target_tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        prompt_list = prompts if isinstance(prompts, list) else [prompts]

        try:
            resp = target_client.completions.create(
                model=target_model_name,
                prompt=prompt_list,
                max_tokens=target_max_tokens,
                temperature=0.0,
            )
            text = resp.choices[0].text if resp.choices else ""
        except Exception as e:
            print(f"Target query error: {e}")
            return False

        result = evaluate_output_prompted(text, attacker_tools[0], attacker_tools, user_tool, attack_prompt)

    return result.get("eval") == "succ"


def main():
    parser = argparse.ArgumentParser(description="InjecAgent RL Attacker Evaluation")

    # Attacker model
    parser.add_argument("--attacker_model", required=True, help="Path to trained attacker model")
    parser.add_argument("--attacker_base_model", default=None, help="Base model for LoRA checkpoints")
    parser.add_argument("--attacker_server_url", default=None, help="vLLM URL for attacker model")

    # Target model
    parser.add_argument("--target_model_url", default="http://localhost:8000/v1")
    parser.add_argument("--target_model_name_or_path", default=None)
    parser.add_argument("--use_openai_target", action="store_true")
    parser.add_argument("--target_openai_config", default=None, help="YAML config for OpenAI target")
    parser.add_argument("--target_max_tokens", type=int, default=512)
    parser.add_argument("--reasoning_effort", default="minimal")

    # Dataset
    parser.add_argument("--data_path", required=True, help="Path to InjecAgent JSON test file")
    parser.add_argument("--prompt_strategy", default="recent_update",
                        choices=["default", "recent_update"])
    parser.add_argument("--max_samples", type=int, default=None)

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_samples", type=int, default=1, help="Pass@k")
    parser.add_argument("--batch_size", type=int, default=16)

    # Output
    parser.add_argument("--output_dir", default="eval_results/injecagent")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tools_path", default="data/injecagent/tools.json")

    args = parser.parse_args()
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    with open(args.data_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    if args.max_samples:
        raw_data = raw_data[: args.max_samples]

    # Build InjecAgentDataset for prompt formatting
    injecagent_ds = InjecAgentDataset(
        data_path=args.data_path,
    )
    if args.max_samples:
        injecagent_ds.dataset = injecagent_ds.dataset[: args.max_samples]

    n_total = len(injecagent_ds)
    print(f"InjecAgent test set: {args.data_path} ({n_total} samples)")

    # Load tool dicts
    tool_dict = injecagent_get_tool_dict(args.tools_path)
    tool_dict_gpt = injecagent_get_tool_dict(args.tools_path, gpt_format=True)

    # Load attacker model
    if args.attacker_server_url:
        attacker_client = OpenAI(base_url=args.attacker_server_url, api_key="EMPTY")
        try:
            models = attacker_client.models.list()
            attacker_model_name = models.data[0].id
        except Exception:
            attacker_model_name = args.attacker_model
        base = args.attacker_base_model or "Qwen/Qwen3-4B-Instruct-2507"
        attacker_tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if attacker_tokenizer.pad_token is None:
            attacker_tokenizer.pad_token = attacker_tokenizer.eos_token
        attacker_model = None
        print(f"Attacker (vLLM): {attacker_model_name}")
    else:
        attacker_client = None
        attacker_model_name = args.attacker_model
        is_lora = os.path.exists(os.path.join(args.attacker_model, "adapter_config.json"))
        base = args.attacker_base_model or "Qwen/Qwen3-4B-Instruct-2507"
        attacker_tokenizer = AutoTokenizer.from_pretrained(
            base if is_lora else args.attacker_model, trust_remote_code=True
        )
        if attacker_tokenizer.pad_token is None:
            attacker_tokenizer.pad_token = attacker_tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base if is_lora else args.attacker_model,
            torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        if is_lora and HAS_PEFT:
            base_model = PeftModel.from_pretrained(base_model, args.attacker_model)
            base_model = base_model.merge_and_unload()
        attacker_model = base_model.eval()
        print(f"Attacker (local): {args.attacker_model}")

    # Load target model
    if args.use_openai_target:
        if args.target_openai_config and os.path.isfile(args.target_openai_config):
            target_client, target_model_name = load_openai_client(
                args.target_openai_config, model_name=args.target_model_name_or_path
            )
        else:
            model_name = args.target_model_name_or_path or "gpt-4o-mini"
            if "gpt" in model_name.lower():
                env_name = model_name.upper().replace("-", "_")
                target_client = AzureOpenAI(
                    api_version=os.environ[f"{env_name}_AZURE_API_VERSION"],
                    api_key=os.environ[f"{env_name}_API_KEY"],
                    azure_endpoint=f"https://{os.environ[f'{env_name}_ENDPOINT']}",
                )
                target_model_name = model_name
            else:
                raise ValueError(f"Cannot build OpenAI target for {model_name}")
        target_tokenizer = None
        print(f"Target (OpenAI API): {target_model_name}")
    else:
        if not args.target_model_name_or_path:
            raise ValueError("--target_model_name_or_path required for vLLM target")
        target_client = OpenAI(base_url=args.target_model_url, api_key="EMPTY")
        try:
            models = target_client.models.list()
            target_model_name = models.data[0].id
        except Exception:
            target_model_name = args.target_model_name_or_path
        try:
            target_tokenizer = AutoTokenizer.from_pretrained(
                args.target_model_name_or_path, trust_remote_code=True
            )
        except Exception:
            target_tokenizer = None
        print(f"Target (vLLM): {target_model_name}")

    # Generate all attack prompts
    print(f"\nGenerating attack prompts ({n_total} samples x {args.num_samples} per sample)...")
    all_prompts = [injecagent_ds[i]["prompt"] for i in range(n_total)]

    if attacker_client:
        all_attack_prompts = generate_attacks_vllm(
            attacker_client, attacker_model_name, attacker_tokenizer,
            all_prompts, args.max_new_tokens, args.temperature, args.num_samples,
        )
    else:
        all_attack_prompts = generate_attacks_local(
            attacker_model, attacker_tokenizer, all_prompts,
            args.max_new_tokens, args.temperature, args.num_samples,
        )

    # Evaluate
    print(f"\nEvaluating ({n_total} samples)...")
    successes = 0
    all_results = []

    def process_one(idx):
        sample = injecagent_ds.dataset[idx]
        attack_prompts = all_attack_prompts[idx]

        success = False
        for ap in attack_prompts:
            if not ap.strip():
                continue
            if evaluate_single_sample(
                sample=sample,
                attack_prompt=ap,
                target_client=target_client,
                target_model_name=target_model_name,
                target_tokenizer=target_tokenizer,
                tool_dict=tool_dict,
                tool_dict_gpt=tool_dict_gpt,
                use_openai_target=args.use_openai_target,
                target_max_tokens=args.target_max_tokens,
                reasoning_effort=args.reasoning_effort,
            ):
                success = True
                break

        return {
            "index": idx,
            "attacker_tools": sample.get("Attacker Tools", []),
            "user_tool": sample.get("User Tool", ""),
            "attack_prompts": [ap[:300] for ap in attack_prompts],
            "success": success,
        }, success

    if args.num_workers > 1:
        results_out = [None] * n_total
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(process_one, i): i for i in range(n_total)}
            for f in tqdm(as_completed(futures), total=n_total, desc="Evaluating"):
                result, success = f.result()
                if success:
                    successes += 1
                results_out[futures[f]] = result
        all_results = [r for r in results_out if r is not None]
    else:
        for idx in tqdm(range(n_total), desc="Evaluating"):
            result, success = process_one(idx)
            if success:
                successes += 1
            all_results.append(result)

    asr = successes / n_total if n_total > 0 else 0
    print(f"\nASR: {asr:.1%} ({successes}/{n_total})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.output_dir, f"eval_injecagent_{timestamp}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config": vars(args),
                "summary": {"n_total": n_total, "successes": successes, "asr": asr},
                "samples": all_results,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
