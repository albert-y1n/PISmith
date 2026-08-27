from types import SimpleNamespace

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.types import ChatUserMessage, text_content_block_from_string

from benchmarks.agentdojo.local import BoundedLocalVLLM


def test_local_target_sends_32k_output_limit():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done"))]
        )

    llm = BoundedLocalVLLM("http://localhost:8000/v1", "test-model")
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    message = ChatUserMessage(
        role="user",
        content=[text_content_block_from_string("hello")],
    )

    llm.query("query", FunctionsRuntime([]), messages=[message])

    assert captured["max_tokens"] == 32_768
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
