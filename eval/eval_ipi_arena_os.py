#!/usr/bin/env python3
"""Evaluate an RL attacker on the open-source IPI Arena benchmark."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from openai import OpenAI

from benchmarks.ipi_arena_os.client import build_runner
from benchmarks.ipi_arena_os.config import DEFAULT_LUNA_MODEL
from benchmarks.ipi_arena_os.dataset import IPIArenaOSDataset, parse_csv
from benchmarks.ipi_arena_os.reward import extract_injection_text
from core.utils import set_random_seed


def generate_attacks(
    client: OpenAI,
    model: str,
    prompts: list[str],
    num_samples: int,
    max_tokens: int,
    temperature: float,
    max_workers: int,
) -> list[list[str]]:
    outputs: list[list[str]] = [[""] * num_samples for _ in prompts]

    def generate_one(index: int):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompts[index]}],
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            n=num_samples,
        )
        return index, [choice.message.content or "" for choice in response.choices]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, index) for index in range(len(prompts))]
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                index, generated = future.result()
                outputs[index] = generated
            except Exception as exc:
                print(f"  Warning: attacker generation failed: {exc}")
            if completed % 10 == 0 or completed == len(prompts):
                print(f"  Generated {completed}/{len(prompts)} behavior groups")
    return outputs


def select_representatives(results: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["behavior_id"]].append(result)

    rng = random.Random(seed)
    selected = []
    for behavior_id in sorted(grouped):
        candidates = sorted(grouped[behavior_id], key=lambda item: item["sample_idx"])
        successful = [item for item in candidates if item["attack_success"]]
        selected.append(rng.choice(successful or candidates))
    return selected


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    behavior_ids = parse_csv(args.behavior_ids)
    waves = [int(value) for value in parse_csv(args.waves) or []] or None
    dataset = IPIArenaOSDataset(
        categories=args.categories,
        behavior_ids=behavior_ids,
        waves=waves,
        behaviors_path=args.behaviors_path,
    )

    print("=" * 70)
    print("IPI Arena OS Evaluation")
    print("=" * 70)
    print(f"  Attacker : {args.attacker_model}")
    print(f"  Target   : {args.target_model} ({args.target_provider})")
    print(f"  Judge    : {args.judge_model} ({args.judge_provider})")
    print(f"  WorldSim : {args.worldsim_model} ({args.worldsim_provider})")
    print(f"  Behaviors: {len(dataset)} ({args.categories})")
    print(f"  Pass@k   : {args.num_samples}")

    attacker_client = OpenAI(base_url=args.attacker_server_url, api_key="EMPTY")
    try:
        attacker_model_name = attacker_client.models.list().data[0].id
    except Exception:
        attacker_model_name = args.attacker_model

    prompts = [dataset[index]["prompt"][0]["content"] for index in range(len(dataset))]
    metadata = [dataset[index] for index in range(len(dataset))]
    print(f"\nGenerating {len(dataset) * args.num_samples} attacks...")
    generation_start = time.time()
    generated = generate_attacks(
        attacker_client,
        attacker_model_name,
        prompts,
        args.num_samples,
        args.max_tokens,
        args.temperature,
        args.generation_workers,
    )
    generation_seconds = time.time() - generation_start
    print(f"  Generation completed in {generation_seconds:.1f}s")

    runner = build_runner(args, verbose=False)
    all_results: list[dict[str, Any]] = []

    def evaluate_one(task_index: int, sample_index: int, completion: str):
        attack_text = extract_injection_text(completion, args.format_prompt)
        result = runner.run_one(metadata[task_index]["behavior_id"], attack_text)
        record = result.to_dict()
        record.update(
            {
                "sample_idx": sample_index,
                "attack_success": bool(result.is_break and not result.error),
                "attacker_completion": completion,
                "attack_text": attack_text,
            }
        )
        return record

    print("\nRunning target, WorldSim, and judge evaluations...")
    evaluation_start = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(evaluate_one, task_index, sample_index, completion)
            for task_index, completions in enumerate(generated)
            for sample_index, completion in enumerate(completions)
        ]
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                all_results.append(future.result())
            except Exception as exc:
                print(f"  Warning: evaluation future failed: {exc}")

            if completed % 10 == 0 or completed == len(futures):
                successes = sum(item["attack_success"] for item in all_results)
                seen = {item["behavior_id"] for item in all_results}
                passed = {
                    item["behavior_id"]
                    for item in all_results
                    if item["attack_success"]
                }
                print(
                    f"  Progress: {completed}/{len(futures)} | "
                    f"Sample ASR: {successes / max(len(all_results), 1):.1%} | "
                    f"Pass@{args.num_samples}: {len(passed) / max(len(seen), 1):.1%}"
                )

    evaluation_seconds = time.time() - evaluation_start
    expected_results = len(dataset) * args.num_samples
    if len(all_results) != expected_results:
        print(f"  Warning: collected {len(all_results)}/{expected_results} results")

    by_behavior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in all_results:
        by_behavior[result["behavior_id"]].append(result)
        by_category[result["category"]].append(result)

    total_behaviors = len(dataset)
    passed_behaviors = sum(
        any(item["attack_success"] for item in by_behavior[sample["behavior_id"]])
        for sample in metadata
    )
    successful_samples = sum(item["attack_success"] for item in all_results)
    total_samples = expected_results
    total_errors = sum(bool(item.get("error")) for item in all_results)

    per_category = {}
    for category in sorted({sample["category"] for sample in metadata}):
        category_ids = [
            sample["behavior_id"] for sample in metadata if sample["category"] == category
        ]
        category_results = by_category[category]
        category_passed = sum(
            any(item["attack_success"] for item in by_behavior[behavior_id])
            for behavior_id in category_ids
        )
        category_successes = sum(item["attack_success"] for item in category_results)
        category_samples = len(category_ids) * args.num_samples
        per_category[category] = {
            f"pass_at_{args.num_samples}": category_passed / max(len(category_ids), 1),
            "sample_asr": category_successes / max(category_samples, 1),
            "total_behaviors": len(category_ids),
            "passed_behaviors": category_passed,
            "total_samples": category_samples,
            "successful_samples": category_successes,
            "errors": sum(bool(item.get("error")) for item in category_results),
        }

    per_behavior = {
        behavior_id: {
            "category": dataset.get_behavior(behavior_id).get("category", "unknown"),
            f"pass_at_{args.num_samples}": any(
                item["attack_success"] for item in by_behavior[behavior_id]
            ),
            "sample_asr": sum(
                item["attack_success"] for item in by_behavior[behavior_id]
            )
            / args.num_samples,
            "successful_samples": sum(
                item["attack_success"] for item in by_behavior[behavior_id]
            ),
            "total_samples": args.num_samples,
            "errors": sum(bool(item.get("error")) for item in by_behavior[behavior_id]),
        }
        for behavior_id in sorted(dataset.behaviors)
    }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "attacker_model": args.attacker_model,
            "target_model": args.target_model,
            "target_provider": args.target_provider,
            "judge_model": args.judge_model,
            "judge_provider": args.judge_provider,
            "worldsim_model": args.worldsim_model,
            "worldsim_provider": args.worldsim_provider,
            "categories": args.categories,
            "behavior_ids": behavior_ids,
            "waves": waves,
            "num_samples": args.num_samples,
            "max_steps": args.max_steps,
        },
        "timing": {
            "generation_seconds": generation_seconds,
            "evaluation_seconds": evaluation_seconds,
        },
        "overall": {
            f"pass_at_{args.num_samples}": passed_behaviors / total_behaviors,
            "sample_asr": successful_samples / max(total_samples, 1),
            "total_behaviors": total_behaviors,
            "passed_behaviors": passed_behaviors,
            "total_samples": total_samples,
            "successful_samples": successful_samples,
            "errors": total_errors,
        },
        "per_category": per_category,
        "per_behavior": per_behavior,
    }

    summary_path = os.path.join(args.output_dir, "eval_results.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    representatives = select_representatives(all_results, args.seed)
    detailed_path = os.path.join(args.output_dir, "eval_detailed.jsonl")
    with open(detailed_path, "w", encoding="utf-8") as handle:
        for result in representatives:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("IPI ARENA OS RESULTS")
    print("=" * 70)
    for category, stats in per_category.items():
        print(
            f"  {category}: Pass@{args.num_samples} "
            f"{stats[f'pass_at_{args.num_samples}']:.1%} "
            f"({stats['passed_behaviors']}/{stats['total_behaviors']}), "
            f"Sample ASR {stats['sample_asr']:.1%}"
        )
    print(
        f"  Overall Pass@{args.num_samples}: "
        f"{summary['overall'][f'pass_at_{args.num_samples}']:.1%} "
        f"({passed_behaviors}/{total_behaviors})"
    )
    print(
        f"  Overall Sample ASR: {summary['overall']['sample_asr']:.1%} "
        f"({successful_samples}/{total_samples})"
    )
    print(f"  Errors: {total_errors}")
    print(f"  Summary : {summary_path}")
    print(f"  Detailed: {detailed_path} ({len(representatives)} behaviors)")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an attacker on IPI Arena OS")
    parser.add_argument("--attacker_model", required=True)
    parser.add_argument("--attacker_server_url", required=True)
    parser.add_argument("--categories", default="tool,coding,browser")
    parser.add_argument("--behavior_ids", default=None)
    parser.add_argument("--waves", default=None)
    parser.add_argument("--behaviors_path", default=None)

    for prefix, default_model in (
        ("target", DEFAULT_LUNA_MODEL),
        ("judge", DEFAULT_LUNA_MODEL),
        ("worldsim", DEFAULT_LUNA_MODEL),
    ):
        parser.add_argument(f"--{prefix}_provider", default="openai")
        parser.add_argument(f"--{prefix}_model", default=default_model)
        parser.add_argument(f"--{prefix}_api_key_env", default="OPENAI_API_KEY")
        parser.add_argument(f"--{prefix}_base_url", default=None)
        parser.add_argument(f"--{prefix}_reasoning_effort", default="none")

    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--generation_workers", type=int, default=32)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--output_dir", default="eval_results/ipi_arena_os")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format_prompt", action="store_true", default=True)
    parser.add_argument("--no_format_prompt", action="store_false", dest="format_prompt")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
