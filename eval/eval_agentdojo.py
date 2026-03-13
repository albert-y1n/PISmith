#!/usr/bin/env python3
"""
Evaluation Script for AgentDojo RL Attacker (RL-trained models only).

Evaluates a trained attacker model on the AgentDojo benchmark.
Generates injection prompts using the trained model and runs them through
the AgentDojo agent pipeline to measure Attack Success Rate (ASR).

Usage:
    # Evaluate on banking suite with vLLM attacker
    python -m eval.eval_agentdojo \\
        --attacker_model checkpoints/agentdojo/checkpoint-500 \\
        --attacker_server_url http://localhost:8001/v1 \\
        --target_model gpt-4o-mini-2024-07-18 \\
        --eval_suites banking \\
        --output_dir eval_results/agentdojo

    # Evaluate with local target model via vLLM
    python -m eval.eval_agentdojo \\
        --attacker_model checkpoints/agentdojo/checkpoint-500 \\
        --attacker_server_url http://localhost:8001/v1 \\
        --target_model local \\
        --target_model_id meta-llama/Llama-3.1-8B-Instruct \\
        --target_model_url http://localhost:8000/v1 \\
        --eval_suites workspace \\
        --output_dir eval_results/agentdojo
"""

import os
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from agentdojo.task_suite.load_suites import get_suite
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.logging import NullLogger, TraceLogger, LOGGER_STACK

from benchmarks.agentdojo.reward import _resolve_agentdojo_model, extract_injection_text
from benchmarks.agentdojo.dataset import (
    AgentDojoDataset,
    compute_injection_candidates,
    format_attacker_prompt,
    get_injection_vector_descriptions,
)
from core.utils import set_random_seed


def generate_injections_vllm(
    client: OpenAI,
    model_name: str,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.7,
    num_samples: int = 1,
    batch_size: int = 32,
) -> List[List[str]]:
    """Generate injection prompts using vLLM server."""
    all_outputs = [None] * len(prompts)

    def _generate_one(idx):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompts[idx]}],
                max_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                n=num_samples,
            )
            return idx, [choice.message.content or "" for choice in resp.choices]
        except Exception as e:
            print(f"  Warning: Generation failed for prompt {idx}: {e}")
            return idx, [""] * num_samples

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(_generate_one, i) for i in range(len(prompts))]
        completed = 0
        for future in as_completed(futures):
            idx, outputs = future.result()
            all_outputs[idx] = outputs
            completed += 1
            if completed % 50 == 0 or completed == len(prompts):
                print(f"  Generated {completed}/{len(prompts)} prompts...")

    return all_outputs


def generate_injections_local(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.7,
    num_samples: int = 1,
    batch_size: int = 4,
) -> List[List[str]]:
    """Generate injection prompts using local transformers model."""
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

    all_outputs = [[] for _ in prompts]

    for _ in range(num_samples):
        for start in range(0, len(prompts), batch_size):
            chunk_prompts = prompts[start : start + batch_size]
            messages_batch = [[{"role": "user", "content": p}] for p in chunk_prompts]
            texts = [
                tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_batch
            ]

            old_side = getattr(tok, "padding_side", "right")
            tok.padding_side = "left"
            encoded = tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)
            tok.padding_side = old_side

            with torch.no_grad():
                outputs = model.generate(**encoded, **gen_kwargs)

            prompt_len = encoded["input_ids"].size(1)
            for i, output in enumerate(outputs):
                text = tok.decode(output[prompt_len:], skip_special_tokens=True)
                all_outputs[start + i].append(text)

    return all_outputs


def evaluate_agentdojo(
    attacker_model: str,
    target_model: str,
    eval_suites: str,
    benchmark_version: str = "v1.2.2",
    eval_user_tasks: Optional[List[str]] = None,
    eval_injection_tasks: Optional[List[str]] = None,
    target_model_id: Optional[str] = None,
    target_model_url: Optional[str] = None,
    target_defense: Optional[str] = None,
    attacker_server_url: Optional[str] = None,
    attacker_base_model: Optional[str] = None,
    format_prompt: bool = True,
    max_tokens: int = 512,
    temperature: float = 0.7,
    num_samples: int = 1,
    max_workers: int = 8,
    output_dir: str = "eval_results",
    logdir: Optional[str] = None,
    seed: int = 42,
):
    """Evaluate attacker model on AgentDojo benchmark."""
    set_random_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("AgentDojo Evaluation")
    print("=" * 70)
    print(f"  Attacker model: {attacker_model}")
    print(f"  Target model: {target_model}")
    if target_model_id:
        print(f"  Target model ID: {target_model_id}")
    if target_defense:
        print(f"  Target defense: {target_defense}")
    print(f"  Eval suites: {eval_suites}")
    print(f"  Benchmark version: {benchmark_version}")
    print(f"  Num samples per task pair: {num_samples}")

    # Set up environment variables for local target models
    if target_model_id:
        os.environ["AGENTDOJO_VLLM_MODEL"] = target_model_id
    if target_model_url:
        os.environ["AGENTDOJO_VLLM_URL"] = target_model_url

    # Create target pipeline
    print("\nCreating target pipeline...")
    llm_resolved = _resolve_agentdojo_model(target_model)
    pipeline_config = PipelineConfig(
        llm=llm_resolved,
        model_id=target_model_id,
        defense=target_defense,
        system_message_name=None,
        system_message=None,
    )
    pipeline = AgentPipeline.from_config(pipeline_config)
    print(f"  Pipeline name: {pipeline.name}")

    # Load evaluation dataset
    print("\nLoading evaluation dataset...")
    eval_dataset = AgentDojoDataset(
        suites=eval_suites,
        benchmark_version=benchmark_version,
        user_tasks=eval_user_tasks,
        injection_tasks=eval_injection_tasks,
    )
    print(f"  Total eval task pairs: {len(eval_dataset)}")

    if len(eval_dataset) == 0:
        print("ERROR: No evaluation samples found!")
        return

    # Collect prompts and metadata
    prompts = []
    metadata = []
    for i in range(len(eval_dataset)):
        sample = eval_dataset[i]
        prompts.append(sample["prompt"][0]["content"])
        metadata.append(
            {
                "suite_name": sample["suite_name"],
                "user_task_id": sample["user_task_id"],
                "injection_task_id": sample["injection_task_id"],
                "injection_candidates": json.loads(sample["injection_candidates"]),
                "injection_task_goal": sample["injection_task_goal"],
                "user_task_prompt": sample["user_task_prompt"],
            }
        )

    # Load attacker model
    if attacker_server_url:
        attacker_client = OpenAI(base_url=attacker_server_url, api_key="EMPTY")
        try:
            models = attacker_client.models.list()
            attacker_model_name = models.data[0].id
            print(f"\nvLLM serving model: {attacker_model_name}")
        except Exception:
            attacker_model_name = attacker_model
        local_attacker = None
        local_tokenizer = None
    else:
        print(f"\nLoading attacker model: {attacker_model}")
        base = attacker_base_model or "Qwen/Qwen3-4B-Instruct-2507"
        is_lora = os.path.exists(os.path.join(attacker_model, "adapter_config.json"))
        local_tokenizer = AutoTokenizer.from_pretrained(
            base if is_lora else attacker_model, trust_remote_code=True
        )
        if local_tokenizer.pad_token is None:
            local_tokenizer.pad_token = local_tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base if is_lora else attacker_model,
            torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        if is_lora and HAS_PEFT:
            base_model = PeftModel.from_pretrained(base_model, attacker_model)
            base_model = base_model.merge_and_unload()
        local_attacker = base_model.eval()
        attacker_client = None
        attacker_model_name = attacker_model

    # Generate injection prompts
    print(f"\nGenerating injection prompts ({len(prompts)} task pairs x {num_samples} samples)...")
    start_time = time.time()

    if attacker_server_url:
        all_outputs = generate_injections_vllm(
            attacker_client, attacker_model_name, prompts,
            max_tokens=max_tokens, temperature=temperature, num_samples=num_samples,
        )
    else:
        all_outputs = generate_injections_local(
            local_attacker, local_tokenizer, prompts,
            max_tokens=max_tokens, temperature=temperature, num_samples=num_samples,
        )

    print(f"  Generation took {time.time() - start_time:.1f}s")

    # Run evaluations
    print(f"\nRunning AgentDojo evaluations...")
    all_results = []
    suite_results = defaultdict(lambda: {"total": 0, "success": 0, "utility": 0})
    injection_task_results = defaultdict(lambda: {"total": 0, "success": 0})
    # pass@k tracking: per task-pair whether any sample has succeeded so far
    pair_ever_succeeded: Dict[str, bool] = {}   # key=(suite/user/inj) -> bool
    pair_seen: set = set()                       # all task pairs encountered so far

    attacker_basename = os.path.basename(attacker_model.rstrip("/"))
    attack_type = f"rl_attacker_{attacker_basename}"

    trace_logdir = logdir or os.path.join(output_dir, "runs")
    os.makedirs(trace_logdir, exist_ok=True)
    print(f"  Trace logs: {trace_logdir}")

    def eval_single(task_idx, sample_idx, injection_text, meta):
        LOGGER_STACK.set([])
        suite = eval_dataset.get_suite(meta["suite_name"])
        user_task = suite.get_user_task_by_id(meta["user_task_id"])
        injection_task = suite.get_injection_task_by_id(meta["injection_task_id"])

        candidates = meta["injection_candidates"]
        if not candidates:
            return task_idx, sample_idx, False, False, injection_text

        injections = {c: injection_text for c in candidates}

        try:
            _null_logger = NullLogger()
            if not hasattr(_null_logger, "logdir"):
                _null_logger.logdir = None  # agentdojo >=0.1.35 NullLogger missing logdir
            with TraceLogger(
                delegate=_null_logger,
                suite_name=meta["suite_name"],
                user_task_id=meta["user_task_id"],
                injection_task_id=meta["injection_task_id"],
                injections=injections,
                attack_type=attack_type,
                pipeline_name=pipeline.name,
                benchmark_version=benchmark_version,
            ) as logger:
                try:
                    utility, security = suite.run_task_with_pipeline(
                        agent_pipeline=pipeline,
                        user_task=user_task,
                        injection_task=injection_task,
                        injections=injections,
                    )
                except Exception as e:
                    logger.log_error(f"Eval failed: {e}")
                    utility, security = False, False
                logger.set_contextarg("utility", utility)
                logger.set_contextarg("security", security)
            return task_idx, sample_idx, utility, security, injection_text
        except Exception as e:
            print(f"  Warning: Eval failed for {meta['user_task_id']}+{meta['injection_task_id']}: {e}")
            return task_idx, sample_idx, False, False, injection_text

    eval_start = time.time()
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for task_idx in range(len(metadata)):
            meta = metadata[task_idx]
            for sample_idx, output in enumerate(all_outputs[task_idx]):
                injection_text = extract_injection_text(output, format_prompt)
                futures.append(
                    executor.submit(eval_single, task_idx, sample_idx, injection_text, meta)
                )

        completed = 0
        total_futures = len(futures)
        for future in as_completed(futures):
            completed += 1
            try:
                task_idx, sample_idx, utility, security, injection_text = future.result()
                meta = metadata[task_idx]
                attack_success = security

                result = {
                    "suite_name": meta["suite_name"],
                    "user_task_id": meta["user_task_id"],
                    "injection_task_id": meta["injection_task_id"],
                    "injection_task_goal": meta["injection_task_goal"],
                    "sample_idx": sample_idx,
                    "injection_text": injection_text[:500],
                    "utility": utility,
                    "security": security,
                    "attack_success": attack_success,
                }
                all_results.append(result)

                sn = meta["suite_name"]
                suite_results[sn]["total"] += 1
                if attack_success:
                    suite_results[sn]["success"] += 1
                if utility:
                    suite_results[sn]["utility"] += 1

                inj_key = f"{sn}/{meta['injection_task_id']}"
                injection_task_results[inj_key]["total"] += 1
                if attack_success:
                    injection_task_results[inj_key]["success"] += 1

                # Update pass@k tracking
                pair_key = f"{sn}/{meta['user_task_id']}/{meta['injection_task_id']}"
                pair_seen.add(pair_key)
                if attack_success:
                    pair_ever_succeeded[pair_key] = True
                elif pair_key not in pair_ever_succeeded:
                    pair_ever_succeeded[pair_key] = False

                if completed % 10 == 0 or completed == total_futures:
                    total_samples = max(sum(r["total"] for r in suite_results.values()), 1)
                    total_success_samples = sum(r["success"] for r in suite_results.values())
                    sample_asr = total_success_samples / total_samples

                    n_pairs_seen = len(pair_seen)
                    n_pairs_passed = sum(1 for v in pair_ever_succeeded.values() if v)
                    passk_running = n_pairs_passed / n_pairs_seen if n_pairs_seen > 0 else 0.0

                    print(f"  Progress: {completed}/{total_futures} | "
                          f"Sample ASR: {sample_asr:.1%} | "
                          f"Pass@{num_samples}: {passk_running:.1%} "
                          f"({n_pairs_passed}/{n_pairs_seen} pairs)")
            except Exception as e:
                print(f"  Warning: Future failed: {e}")

    eval_time = time.time() - eval_start

    # ── Pass@k aggregation ────────────────────────────────────────────────
    # Group all sample results by task pair; a pair is "passed" if any sample
    # succeeded (security=True).  This is the correct pass@k metric.
    pair_security: Dict[str, list] = defaultdict(list)   # key -> [security, ...]
    pair_utility:  Dict[str, list] = defaultdict(list)   # key -> [utility, ...]
    pair_meta:     Dict[str, dict] = {}
    for r in all_results:
        key = f"{r['suite_name']}/{r['user_task_id']}/{r['injection_task_id']}"
        pair_security[key].append(r["security"])
        pair_utility[key].append(r["utility"])
        pair_meta[key] = {"suite_name": r["suite_name"],
                          "injection_task_id": r["injection_task_id"]}

    passk_suite: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "pass": 0, "utility": 0})
    passk_inj:   Dict[str, Dict] = defaultdict(lambda: {"total": 0, "pass": 0})
    for key, sec_list in pair_security.items():
        meta_k = pair_meta[key]
        sn = meta_k["suite_name"]
        ik = f"{sn}/{meta_k['injection_task_id']}"
        passed   = any(sec_list)
        utility  = any(pair_utility[key])
        passk_suite[sn]["total"]   += 1
        passk_suite[sn]["utility"] += int(utility)
        if passed:
            passk_suite[sn]["pass"] += 1
        passk_inj[ik]["total"] += 1
        if passed:
            passk_inj[ik]["pass"] += 1

    total_pairs   = sum(s["total"]   for s in passk_suite.values())
    total_passed  = sum(s["pass"]    for s in passk_suite.values())
    total_util_pk = sum(s["utility"] for s in passk_suite.values())
    passk_overall = total_passed  / total_pairs   if total_pairs   > 0 else 0
    util_overall  = total_util_pk / total_pairs   if total_pairs   > 0 else 0

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Eval time : {eval_time:.1f}s")
    print(f"  Pass@{num_samples} (primary metric): at least 1 of {num_samples} samples succeeds per pair")

    print(f"\n--- Per-Suite Pass@{num_samples} ---")
    for suite_name, stats in sorted(passk_suite.items()):
        asr  = stats["pass"]    / stats["total"] if stats["total"] > 0 else 0
        util = stats["utility"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {suite_name}: Pass@{num_samples} {asr:.1%} ({stats['pass']}/{stats['total']} pairs), "
              f"Utility {util:.1%}")

    print(f"\n--- Per-Injection-Task Pass@{num_samples} ---")
    for inj_key, stats in sorted(passk_inj.items()):
        asr = stats["pass"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {inj_key}: Pass@{num_samples} {asr:.1%} ({stats['pass']}/{stats['total']} pairs)")

    print(f"\n--- Overall ---")
    print(f"  Pass@{num_samples} ASR : {passk_overall:.1%} ({total_passed}/{total_pairs} pairs)")
    print(f"  Utility      : {util_overall:.1%} ({total_util_pk}/{total_pairs} pairs)")

    # ── Save results ──────────────────────────────────────────────────────
    results_file = os.path.join(output_dir, "eval_results.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "attacker_model": attacker_model,
            "target_model": target_model,
            "target_model_id": target_model_id,
            "target_defense": target_defense,
            "eval_suites": eval_suites,
            "benchmark_version": benchmark_version,
            "format_prompt": format_prompt,
            "num_samples": num_samples,
        },
        "overall": {
            f"pass_at_{num_samples}": passk_overall,
            "utility": util_overall,
            "total_pairs": total_pairs,
            "passed_pairs": total_passed,
        },
        "per_suite": {
            sn: {
                f"pass_at_{num_samples}": stats["pass"] / stats["total"] if stats["total"] > 0 else 0,
                "utility": stats["utility"] / stats["total"] if stats["total"] > 0 else 0,
                **stats,
            }
            for sn, stats in passk_suite.items()
        },
        "per_injection_task": {
            ik: {
                f"pass_at_{num_samples}": stats["pass"] / stats["total"] if stats["total"] > 0 else 0,
                **stats,
            }
            for ik, stats in passk_inj.items()
        },
    }
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    detailed_file = os.path.join(output_dir, "eval_detailed.jsonl")
    with open(detailed_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Detailed results saved to: {detailed_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="AgentDojo RL Attacker Evaluation")

    # Attacker model
    parser.add_argument("--attacker_model", required=True)
    parser.add_argument("--attacker_base_model", default=None)
    parser.add_argument("--attacker_server_url", default=None,
                        help="vLLM URL for attacker (if externally served)")

    # Target model
    parser.add_argument("--target_model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--target_model_id", default=None)
    parser.add_argument("--target_model_url", default=None)
    parser.add_argument("--target_defense", default=None)

    # Evaluation config
    parser.add_argument("--eval_suites", default="workspace")
    parser.add_argument("--eval_user_tasks", default=None)
    parser.add_argument("--eval_injection_tasks", default=None)
    parser.add_argument("--benchmark_version", default="v1.2.2")

    parser.add_argument("--format_prompt", action="store_true", default=True)
    parser.add_argument("--no_format_prompt", action="store_false", dest="format_prompt")

    # Generation
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_samples", type=int, default=1)

    # Other
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--output_dir", default="eval_results/agentdojo")
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    eval_user_tasks = (
        [t.strip() for t in args.eval_user_tasks.split(",")]
        if args.eval_user_tasks else None
    )
    eval_injection_tasks = (
        [t.strip() for t in args.eval_injection_tasks.split(",")]
        if args.eval_injection_tasks else None
    )

    evaluate_agentdojo(
        attacker_model=args.attacker_model,
        target_model=args.target_model,
        eval_suites=args.eval_suites,
        benchmark_version=args.benchmark_version,
        eval_user_tasks=eval_user_tasks,
        eval_injection_tasks=eval_injection_tasks,
        target_model_id=args.target_model_id,
        target_model_url=args.target_model_url,
        target_defense=args.target_defense,
        attacker_server_url=args.attacker_server_url,
        attacker_base_model=args.attacker_base_model,
        format_prompt=args.format_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        logdir=args.logdir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
