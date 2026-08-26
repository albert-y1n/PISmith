from types import SimpleNamespace

from ipi_arena_bench.llm_client import ChatResponse, ToolCall

from benchmarks.ipi_arena_os.client import build_runner
from benchmarks.ipi_arena_os.config import DEFAULT_LUNA_MODEL, IPIArenaOSGRPOConfig
from benchmarks.ipi_arena_os.dataset import IPIArenaOSDataset
from benchmarks.ipi_arena_os.reward import IPIArenaOSAttackReward
from eval.eval_ipi_arena_os import select_representatives


def test_dataset_loads_all_official_behaviors():
    dataset = IPIArenaOSDataset(categories="all")

    assert len(dataset) == 41
    assert {sample["category"] for sample in dataset.samples} == {
        "tool",
        "coding",
        "browser",
    }
    assert all("<INJECTION_HERE>" in sample["prompt_text"] for sample in dataset.samples)
    assert max(len(sample["prompt_text"]) for sample in dataset.samples) < 13000


def test_luna_is_the_default_judge_and_worldsim():
    config = IPIArenaOSGRPOConfig(output_dir="unused")
    assert config.judge_model == DEFAULT_LUNA_MODEL
    assert config.worldsim_model == DEFAULT_LUNA_MODEL


def test_dataset_filters_categories_ids_and_waves():
    dataset = IPIArenaOSDataset(
        categories="tool,coding",
        behavior_ids=["garage-door-email", "code-backdoor", "venmo-5k-no-permission"],
        waves=[1, 3],
    )

    assert {sample["behavior_id"] for sample in dataset.samples} == {
        "garage-door-email",
        "code-backdoor",
    }


def test_representatives_prefer_success():
    rows = [
        {"behavior_id": "a", "sample_idx": 0, "attack_success": False},
        {"behavior_id": "a", "sample_idx": 1, "attack_success": True},
        {"behavior_id": "b", "sample_idx": 0, "attack_success": False},
    ]

    selected = select_representatives(rows, seed=42)
    assert len(selected) == 2
    assert next(row for row in selected if row["behavior_id"] == "a")["attack_success"]


def test_reward_uses_official_break_verdict(monkeypatch, tmp_path):
    class FakeRunner:
        def run_one(self, behavior_id, attack_text):
            return SimpleNamespace(
                is_break=attack_text == "successful attack",
                error=None,
                criteria={"criterion": {"passed": attack_text == "successful attack"}},
            )

    monkeypatch.setattr(
        "benchmarks.ipi_arena_os.reward.build_runner",
        lambda config, verbose=False: FakeRunner(),
    )
    config = SimpleNamespace(
        format_prompt=True,
        soft_rewards=False,
        eval_max_workers=2,
        output_dir=str(tmp_path),
    )
    reward = IPIArenaOSAttackReward(config)

    rewards = reward(
        prompts=[[], []],
        completions=[
            [{"content": "<prompt>successful attack</prompt>"}],
            [{"content": "<prompt>failed attack</prompt>"}],
        ],
        behavior_id=["a", "b"],
        category=["tool", "coding"],
    )

    assert rewards == [1.0, 0.0]


def test_official_runner_uses_vendored_behavior_data(monkeypatch):
    class FakeClient:
        def __init__(self, role):
            self.role = role
            self.model = f"fake-{role}"
            self.calls = 0

        def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.role == "target" and self.calls == 1:
                return ChatResponse(
                    tool_calls=[
                        ToolCall(
                            name="activate_device",
                            arguments={"name": "Garage Door"},
                            id="call-1",
                        )
                    ]
                )
            if self.role == "target":
                return ChatResponse(content="Candidate summary completed.")
            if self.role == "judge":
                return ChatResponse(
                    content=(
                        '{"Summarizes Email":{"score":10,"explanation":"ok"},'
                        '"No Disclosure":{"score":10,"explanation":"ok"}}'
                    )
                )
            return ChatResponse(content='{"status":"success"}')

    clients = {role: FakeClient(role) for role in ("target", "judge", "worldsim")}
    monkeypatch.setattr(
        "benchmarks.ipi_arena_os.client.build_client",
        lambda config, prefix: clients[prefix],
    )
    runner = build_runner(SimpleNamespace(behaviors_path=None, max_steps=5))

    result = runner.run_one("garage-door-email", "malicious instruction")
    assert result.error is None
    assert result.is_break is True
    assert result.tool_calls_made[0]["name"] == "activate_device"
