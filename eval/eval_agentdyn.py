#!/usr/bin/env python3
"""
Evaluation script for AgentDyn RL attacker.
"""

import argparse

from benchmarks.agentdyn.dataset import AgentDynDataset
from eval.eval_agentdojo import evaluate_agentdojo


def main():
    parser = argparse.ArgumentParser(description="AgentDyn RL Attacker Evaluation")

    parser.add_argument("--attacker_model", required=True)
    parser.add_argument("--attacker_base_model", default=None)
    parser.add_argument("--attacker_server_url", default=None,
                        help="vLLM URL for attacker (if externally served)")

    parser.add_argument("--target_model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--target_model_id", default=None)
    parser.add_argument("--target_model_url", default=None)
    parser.add_argument("--target_adapter", choices=["secopd"], default=None)
    parser.add_argument("--target_max_tokens", type=int, default=32768)
    parser.add_argument("--target_defense", default=None)

    parser.add_argument("--eval_suites", default="workspace,github,dailylife,shopping")
    parser.add_argument("--eval_user_tasks", default=None)
    parser.add_argument("--eval_injection_tasks", default=None)
    parser.add_argument("--benchmark_version", default="v1.2.2")

    parser.add_argument("--format_prompt", action="store_true", default=True)
    parser.add_argument("--no_format_prompt", action="store_false", dest="format_prompt")

    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--injections_cache", default=None)
    parser.add_argument("--generate_only", action="store_true")

    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--output_dir", default="eval_results/agentdyn")
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
        target_adapter=args.target_adapter,
        target_max_tokens=args.target_max_tokens,
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
        injections_cache=args.injections_cache,
        generate_only=args.generate_only,
        benchmark_label="AgentDyn",
        benchmark_slug="agentdyn",
        dataset_cls=AgentDynDataset,
    )


if __name__ == "__main__":
    main()
