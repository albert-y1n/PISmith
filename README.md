# PISmith: Reinforcement Learning-based Red Teaming for Prompt Injection Defenses

This is an official implementation of [**PISmith: Reinforcement Learning-based Red Teaming for Prompt Injection Defenses**](https://arxiv.org/abs/2603.13026)

---

## Environment Setup

PISmith has been tested using Python 3.10 and CUDA Version: 12.9

**1. Create a Python 3.10 conda environment**

```bash
conda create -n PISmith python=3.10 -y
conda activate PISmith
```

**2. Install dependencies** 

```bash
pip install -r requirements.txt
```

**3. (Optional) Prepare the [Meta-SecAlign](https://github.com/facebookresearch/Meta_SecAlign) model checkpoint**

For experiments targeting the `secalign` defense, run the provided merge script to download and merge the base model with the SecAlign adapter:

```bash
python merge_meta_secalign.py
```

This downloads `meta-llama/Llama-3.1-8B-Instruct` and `facebook/Meta-SecAlign-8B` from HuggingFace, merges them, and saves the result to `checkpoints/Meta-SecAlign-8B-merged/`.

---

## Scripts

### [PIArena](https://github.com/sleeepeer/PIArena)

[PIArena](https://github.com/sleeepeer/PIArena) supports training and evaluation against a range of prompt injection defenses. Use the `defense` argument to select the target defense.

**Supported defenses:**
`secalign`, `none`, `promptguard`, `promptarmor`, `sandwich`, `instructional`, `datasentinel`, `piguard`, `datafilter`

#### Training

```bash
bash scripts/train_piarena.sh <defense> [train_gpus] [target_gpu] [target_port]
```

| Argument | Default | Description |
|---|---|---|
| `defense` | `secalign` | Defense to train against |
| `train_gpus` | `"1,2,3"` | GPU indices for RL training |
| `target_gpu` | `0` | GPU for the target vLLM server |
| `target_port` | `8010` | Port for the target vLLM server |

Examples:

```bash
# Train against SecAlign defense
bash scripts/train_piarena.sh secalign

# Train against no defense (plain LLM)
bash scripts/train_piarena.sh none
```

#### Evaluation

```bash
bash scripts/eval_piarena.sh <checkpoint> <defense> [target_port] [target_gpu] [attacker_gpu] [attacker_port] [num_samples]
```

| Argument | Default | Description |
|---|---|---|
| `checkpoint` | — | Path to trained attacker checkpoint |
| `defense` | `secalign` | Defense to evaluate against |
| `target_port` | `8000` | Port for the target vLLM server |
| `target_gpu` | `0` | GPU for the target vLLM server |
| `attacker_gpu` | `1` | GPU for the attacker vLLM server |
| `attacker_port` | `8001` | Port for the attacker vLLM server |
| `num_samples` | `10` | Pass@k: number of samples per test case |

Examples:

```bash
# Evaluate against SecAlign (default settings)
bash scripts/eval_piarena.sh checkpoints/piarena/checkpoint-500 secalign

# Evaluate against no piguard, pass@10
bash scripts/eval_piarena.sh checkpoints/piarena_none/checkpoint-500 piguard
```
---

### [AgentDojo](https://github.com/ethz-spylab/agentdojo)

Supports GPT-4o-mini, GPT-4o, and local vLLM targets.

```bash
bash scripts/train_agentdojo.sh [target_type] [suites] [train_gpus]
```

```bash
# Default: GPT-4o-mini target on the firsr 7 injected task of workspace suite
bash scripts/train_agentdojo.sh

# Train on all suites (workspace, banking, travel, slack)
bash scripts/train_agentdojo.sh gpt4o-mini all
```

Evaluation:

```bash
bash scripts/eval_agentdojo.sh <checkpoint> [target_type] [eval_suites] [num_samples] [target_defense]

# Example
bash scripts/eval_agentdojo.sh checkpoints/agentdojo/checkpoint-500 gpt4o-mini
```

---

### [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent)

Supports a local vLLM target, GPT-4o-mini, or multi-target mode.

```bash
bash scripts/train_injecagent.sh [target_type] [train_gpus] [target_gpu] [target_port]
```

```bash
# Default: local vLLM target (Meta-SecAlign-8B)
bash scripts/train_injecagent.sh

# GPT-4o-mini API target
bash scripts/train_injecagent.sh gpt4o-mini
```

Evaluation:

```bash
bash scripts/eval_injecagent.sh <checkpoint> [target_type] [target_gpu] [target_port] [eval_gpu] [num_samples]

# Example
bash scripts/eval_injecagent.sh checkpoints/injecagent/checkpoint-500
```

---

## Experiment Results

### Main Results (vs. Meta-SecAlign-8B, 13 Benchmarks)

PISmith is evaluated against 7 baselines spanning static, search-based, and RL-based attack categories. All RL-based methods report ASR@10 / ASR@1; static and search-based methods report ASR@1.

| Method | Category | Avg. ASR@10 | Avg. ASR@1 |
|---|---|---|---|
| Direct | Static | — | 0.04 |
| Combined | Static | — | 0.07 |
| TAP | Search-based | — | 0.11 |
| PAIR | Search-based | — | 0.16 |
| Strategy | Search-based | — | 0.21 |
| Vanilla GRPO | RL-based | 0.13 | 0.05 |
| RL-Hammer | RL-based | 0.70 | 0.48 |
| **PISmith (Ours)** | **RL-based** | **1.00** | **0.87** |

### Utility–Robustness Trade-off (8 Defenses, Qwen3-4B-Instruct-2507)

PISmith ASR@1 averaged over 13 benchmarks. Utility measures task accuracy without attack.

| Defense | Type | Utility | PISmith ASR@1 |
|---|---|---|---|
| No Defense | — | 0.74 | 0.92 |
| Sandwich | Prevention | 0.74 | 0.91 |
| Instructional | Prevention | 0.73 | 0.92 |
| PromptArmor | Prevention | 0.74 | 0.92 |
| DataFilter | Prevention | 0.63 | 0.49 |
| PIGuard | Filter | 0.72 | 0.82 |
| PromptGuard | Filter | 0.66 | 0.89 |
| DataSentinel | Filter | 0.55 | 0.52 |

> It remains challenging for state-of-the-art defenses to simultaneously achieve high utility (≥0.70) and low ASR (≤0.60), revealing a fundamental utility–robustness trade-off.

### Agentic Settings

#### InjecAgent

| Target Model | Direct ASR@1 | **PISmith ASR@10/1** |
|---|---|---|
| Meta-SecAlign-8B | 0.00 | **1.00 / 0.99** |
| GPT-4o-mini | 0.02 | **1.00 / 0.99** |
| GPT-4.1-nano | 0.01 | **1.00 / 1.00** |
| GPT-5-nano | 0.00 | **1.00 / 0.95** |

#### AgentDojo (best static baseline vs. PISmith)

| Target Model | Best Static ASR@1 | **PISmith ASR@10/1** |
|---|---|---|
| GPT-4o-mini | 0.23 | **0.78 / 0.62** |
| GPT-4.1-nano | 0.20 | **0.81 / 0.64** |
| GPT-5-nano | 0.01 | **0.38 / 0.24** |
