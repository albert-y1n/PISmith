"""SecOPD target adapter for AgentDyn.

AgentDyn must retain ``tool`` messages internally so its tool loop can execute.
At the model boundary SecOPD instead requires every untrusted environment/tool
return to use the custom ``input`` role.  This adapter performs that conversion
immediately before the OpenAI-compatible vLLM request.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.local_llm import (
    _make_system_prompt,
    _parse_model_output,
    reformat_message,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str


class SecOPDVLLM(BasePipelineElement):
    """Call SecOPD through vLLM with its required trust-boundary settings."""

    name = "secopd-vllm"
    _audit_lock = threading.Lock()
    _request_count = 0

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        max_tokens: int = 32_768,
        request_timeout: float = 3_600,
        max_retries: int = 4,
    ) -> None:
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url.rstrip("/"),
            timeout=request_timeout,
            max_retries=2,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.audit_file = os.getenv("SECOPD_AUDIT_FILE")
        self.audit_sample_limit = int(os.getenv("SECOPD_AUDIT_SAMPLE_LIMIT", "20"))

    @staticmethod
    def format_wire_messages(
        messages: Sequence[ChatMessage], runtime: FunctionsRuntime
    ) -> list[dict[str, Any]]:
        """Convert AgentDyn messages into SecOPD wire messages.

        In particular, every environment result (AgentDyn role ``tool``) is
        emitted as SecOPD role ``input``.  No tool-return payload is copied into
        a trusted ``system`` or ``user`` message.
        """
        wire_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            content: Any = message["content"]

            if role == "system" and content is not None:
                content = _make_system_prompt(
                    get_text_content_as_str(content), runtime.functions.values()
                )
            elif role == "tool":
                role = "input"
                error = message.get("error")
                if error is not None:
                    content = json.dumps({"error": error}, ensure_ascii=False)
                else:
                    result = message["content"]
                    if result == "None":
                        result = "Success"
                    content = json.dumps(
                        {"result": result}, ensure_ascii=False, default=str
                    )

            wire_messages.append(
                {"role": role, "content": reformat_message({"role": role, "content": content})}
            )

        return wire_messages

    def _complete(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=self.max_tokens,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": True}
                    },
                )
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("SecOPD returned an empty completion")
                choice = response.choices[0]
                usage = response.usage
                return content, {
                    "finish_reason": choice.finish_reason,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                }
            except Exception as error:
                last_error = error
                if attempt == self.max_retries:
                    break
                time.sleep(min(2**attempt, 8))

        raise RuntimeError(
            f"SecOPD vLLM request failed after {self.max_retries + 1} attempts"
        ) from last_error

    def _audit(
        self,
        wire_messages: list[dict[str, Any]],
        completion: str,
        request_meta: dict[str, Any],
        parsed_tool_calls: int,
    ) -> None:
        if not self.audit_file:
            return

        with self._audit_lock:
            type(self)._request_count += 1
            request_index = type(self)._request_count
            has_function_open = "<function" in completion
            has_function_close = "</function>" in completion
            has_native_tool_call = "<tool_call>" in completion
            thinking_closed = "</think>" in completion
            anomaly_reasons = []
            if request_meta["finish_reason"] == "length":
                anomaly_reasons.append("max_tokens_truncation")
            if not completion.strip():
                anomaly_reasons.append("empty_completion")
            if has_function_open != has_function_close:
                anomaly_reasons.append("unclosed_function_call")
            if has_function_open and parsed_tool_calls == 0:
                anomaly_reasons.append("function_call_not_parsed")
            if has_native_tool_call and parsed_tool_calls == 0:
                anomaly_reasons.append("native_tool_call_not_parsed")
            if not thinking_closed:
                anomaly_reasons.append("missing_thinking_close")

            record = {
                "request_index": request_index,
                "wire_roles": [message["role"] for message in wire_messages],
                "input_role_count": sum(
                    message["role"] == "input" for message in wire_messages
                ),
                **request_meta,
                "completion_characters": len(completion),
                "thinking_closed": thinking_closed,
                "parsed_tool_calls": parsed_tool_calls,
                "has_native_tool_call": has_native_tool_call,
                "anomalies": anomaly_reasons,
            }
            if anomaly_reasons or request_index <= self.audit_sample_limit:
                record["completion"] = completion

            os.makedirs(os.path.dirname(os.path.abspath(self.audit_file)), exist_ok=True)
            with open(self.audit_file, "a") as audit_handle:
                audit_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        wire_messages = self.format_wire_messages(messages, runtime)
        completion, request_meta = self._complete(wire_messages)
        output = _parse_model_output(completion)
        self._audit(
            wire_messages,
            completion,
            request_meta,
            len(output.get("tool_calls") or []),
        )
        return query, runtime, env, [*messages, output], extra_args
