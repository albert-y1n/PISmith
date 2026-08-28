from types import SimpleNamespace

import pytest

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.types import (
    ChatSystemMessage,
    ChatUserMessage,
    text_content_block_from_string,
)

from benchmarks.agentdojo.reward import _build_agentdojo_llm
from core.providers import create_openai_compatible_client, resolve_api_key_env


def test_provider_specific_key_environment_defaults():
    assert resolve_api_key_env("openai") == "OPENAI_API_KEY"
    assert resolve_api_key_env("openrouter") == "OPENROUTER_API_KEY"
    assert resolve_api_key_env("vllm") is None


def test_openrouter_client_uses_official_endpoint(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    client = create_openai_compatible_client("openrouter")

    assert str(client.base_url) == "https://openrouter.ai/api/v1/"


def test_openrouter_requires_its_own_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        create_openai_compatible_client("openrouter")


def test_agentdojo_builds_openrouter_target(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    llm = _build_agentdojo_llm(
        "google/gemini-3.7-flash",
        target_provider="openrouter",
    )

    assert llm.model == "google/gemini-3.7-flash"
    assert llm.name == "google/gemini-3.7-flash"
    assert str(llm.client.base_url) == "https://openrouter.ai/api/v1/"


def test_agentdojo_openrouter_target_uses_openai_message_format(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None)
                )
            ]
        )

    llm = _build_agentdojo_llm(
        "google/gemini-3.7-flash",
        target_provider="openrouter",
    )
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    messages = [
        ChatSystemMessage(
            role="system",
            content=[text_content_block_from_string("system")],
        ),
        ChatUserMessage(
            role="user",
            content=[text_content_block_from_string("hello")],
        ),
    ]

    llm.query("query", FunctionsRuntime([]), messages=messages)

    assert [message["role"] for message in captured["messages"]] == [
        "developer",
        "user",
    ]
    assert captured["model"] == "google/gemini-3.7-flash"
