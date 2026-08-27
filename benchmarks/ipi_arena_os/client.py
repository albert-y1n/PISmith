"""OpenAI-compatible client construction for IPI Arena target, judge, and WorldSim."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from openai import OpenAI
from ipi_arena_bench.llm_client import ChatResponse, ToolCall


PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "vllm": "http://localhost:8000/v1",
}


class IPIArenaOSLLMClient:
    """Small provider adapter implementing the official runner's client protocol."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        resolved_url = base_url or PROVIDER_BASE_URLS.get(provider)
        if not resolved_url:
            raise ValueError(f"Unknown provider '{provider}'; provide an explicit base URL.")
        if provider != "vllm" and not api_key:
            raise ValueError(f"An API key is required for provider '{provider}'.")

        self.provider = provider
        self.model = model
        self.reasoning_effort = (
            None if provider == "vllm" and reasoning_effort == "none" else reasoning_effort
        )
        self.client = OpenAI(
            base_url=resolved_url,
            api_key=api_key or "EMPTY",
            max_retries=3,
            timeout=120.0,
        )

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if self.provider == "openai":
            # GPT-5.6 reasoning and visible output share the completion budget.
            # Leave it unset so target, judge, and WorldSim use the provider's
            # full model default rather than an evaluation-side cap.
            if "gpt-5.6" not in self.model:
                kwargs["max_completion_tokens"] = max_tokens
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        tool_calls = []
        for tool_call in message.tool_calls or []:
            raw_arguments = tool_call.function.arguments
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except (json.JSONDecodeError, TypeError):
                arguments = {"_raw": raw_arguments}
            tool_calls.append(
                ToolCall(
                    name=tool_call.function.name,
                    arguments=arguments,
                    id=tool_call.id or "",
                )
            )

        reasoning = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatResponse(
            content=message.content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            raw_response=response,
            model=response.model or self.model,
            usage=usage,
        )


def build_client(config: Any, prefix: str) -> IPIArenaOSLLMClient:
    provider = getattr(config, f"{prefix}_provider")
    model = getattr(config, f"{prefix}_model")
    api_key_env = getattr(config, f"{prefix}_api_key_env")
    api_key = None if provider == "vllm" else os.environ.get(api_key_env)
    if provider != "vllm" and not api_key:
        raise ValueError(
            f"{api_key_env} is not set; it is required by the {prefix} provider '{provider}'."
        )
    return IPIArenaOSLLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=getattr(config, f"{prefix}_base_url"),
        reasoning_effort=getattr(config, f"{prefix}_reasoning_effort"),
    )


def build_runner(config: Any, verbose: bool = False):
    from ipi_arena_bench.runner import BenchmarkRunner
    from .dataset import DATA_PATH

    behaviors_path = getattr(config, "behaviors_path", None) or str(DATA_PATH)
    return BenchmarkRunner(
        target_client=build_client(config, "target"),
        judge_client=build_client(config, "judge"),
        worldsim_client=build_client(config, "worldsim"),
        behaviors_path=behaviors_path,
        verbose=verbose,
        max_steps=getattr(config, "max_steps", 5),
    )
