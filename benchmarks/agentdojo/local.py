"""Bounded-output adapter for ordinary local AgentDojo/AgentDyn targets."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence

from openai import OpenAI

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.local_llm import (
    _make_system_prompt,
    _parse_model_output,
    reformat_message,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str


class BoundedLocalVLLM(BasePipelineElement):
    """Local vLLM target with an explicit per-response token budget."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        max_tokens: int = 32_768,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> None:
        self.client = OpenAI(api_key="EMPTY", base_url=base_url.rstrip("/"))
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.name = f"local-vllm-{model}"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        wire_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system" and content is not None:
                content = _make_system_prompt(
                    get_text_content_as_str(content), runtime.functions.values()
                )
            elif role == "tool":
                if message.get("error") is not None:
                    content = json.dumps({"error": message["error"]})
                else:
                    result = message["content"]
                    if result == "None":
                        result = "Success"
                    content = json.dumps({"result": result})

            wire_messages.append(
                {"role": role, "content": reformat_message({"role": role, "content": content})}
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=wire_messages,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=random.randint(0, 1_000_000),
            max_tokens=self.max_tokens,
        )
        completion = response.choices[0].message.content
        if completion is None:
            raise RuntimeError("Local vLLM target returned an empty completion")
        output = _parse_model_output(completion)
        return query, runtime, env, [*messages, output], extra_args
