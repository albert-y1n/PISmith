"""
GRPO configuration for PIArena benchmark.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from trl import GRPOConfig


@dataclass
class PIArenaGRPOConfig(GRPOConfig):
    """Extended GRPO config for PIArena prompt injection attack training."""

    # --- Dataset ---
    dataset_path: str = field(
        default="datasets/PIArena-data/data",
        metadata={"help": "Local parquet dir, HF dataset id, or JSON file path"},
    )
    dataset_split: str = field(default="dolly_closed_qa")
    train_start_idx: int = field(default=0)
    train_end_idx: int = field(default=100)

    # --- Models ---
    attacker_model_name_or_path: str = field(default="Qwen/Qwen3-4B-Instruct-2507")
    target_model_name_or_path: str = field(default="checkpoints/Meta-SecAlign-8B-merged")
    target_model_url: str = field(default="http://localhost:8010/v1")
    defense_method: str = field(
        default="secalign",
        metadata={"help": "Defense: secalign, none, datafilter, datasentinel, promptguard, promptarmor, piguard, sandwich, instructional"},
    )

    # --- Judge ---
    judge_model_config: str = field(default="configs/openai_configs/gpt-4o-mini.yaml")

    # --- Attack strategy ---
    attack_template: str = field(default="recent_update")
    inject_position: str = field(default="mid", metadata={"help": "'mid', 'end', or 'start'"})
    format_prompt: bool = field(default=True, metadata={"help": "Require <prompt></prompt> tags in output"})

    # --- Joint training (two targets) ---
    joint: bool = field(default=False)
    target_model_name_or_path_2: str = field(default="meta-llama/Llama-3.1-8B-Instruct")
    target_model_url_2: str = field(default="http://localhost:8011/v1")
    joint_reward_llama: float = field(default=0.3)
    joint_reward_secalign: float = field(default=0.7)
    joint_reward_both: float = field(default=1.0)

    # --- Generation ---
    max_completion_length: int = field(default=512)
    target_model_max_tokens: int = field(default=256)
    target_model_temperature: float = field(default=0.0)

    # --- Reward ---
    reward_functions: List[str] = field(
        default_factory=lambda: ["PIArenaAttackReward"]
    )
    soft_rewards: bool = field(default=True)
    format_reward_weight: float = field(default=0.1)

    # --- Training ---
    seed: int = field(default=42)
    save_total_limit: int = field(default=5)
    max_grad_norm: float = field(default=1.0)
    loss_type: str = field(default="grpo")
    beta: float = field(default=0.01, metadata={"help": "KL penalty coefficient"})
    epsilon: float = field(default=0.2, metadata={"help": "GRPO clip parameter"})

    # --- Adaptive trainer ---
    adaptive: bool = field(default=False, metadata={"help": "Enable AdaptiveGRPOTrainer"})
    enable_adaptive_entropy: bool = field(default=True)
    enable_success_boost: bool = field(default=True)
    fixed_entropy_coef: bool = field(default=False, metadata={"help": "Ablation: constant entropy coef"})


@dataclass
class PIArenaEvalConfig:
    """Configuration for PIArena evaluation."""

    # --- Attacker ---
    attacker_model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    attacker_base_model_name_or_path: Optional[str] = None
    use_vllm_attacker: bool = False
    attacker_model_url: str = "http://localhost:8011/v1"

    # --- Target ---
    target_model_name_or_path: str = "checkpoints/Meta-SecAlign-8B-merged"
    target_model_url: str = "http://localhost:8010/v1"
    use_openai_target: bool = False
    target_model_openai_config: Optional[str] = None

    # --- Judge ---
    judge_model_config: str = "configs/openai_configs/gpt-4o-mini.yaml"

    # --- Defense ---
    defense_method: str = "secalign"

    # --- Attack format ---
    attack_template: str = "recent_update"
    inject_position: str = "mid"
    format_prompt: bool = True

    # --- Data ---
    dataset_path: str = "datasets/PIArena-data/data"
    dataset_split: str = "squad_v2"
    eval_start_idx: int = 100
    eval_end_idx: int = 200

    # --- Generation ---
    max_new_tokens: int = 2048
    target_model_max_tokens: int = 8192
    temperature: float = 0.7
    top_p: Optional[float] = 0.8
    do_sample: bool = True

    # --- Pass@k ---
    num_samples_per_case: int = 10
    batch_size: int = 20
    attacker_batch_size: int = 20

    # --- Output ---
    output_dir: str = "results/eval_piarena"
    save_name: str = "eval_results"
