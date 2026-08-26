import json

from benchmarks.agentdyn.secopd import SecOPDVLLM


class _Runtime:
    functions = {}


def test_tool_results_are_sent_only_as_input_messages():
    messages = [
        {"role": "user", "content": [{"type": "text", "content": "trusted task"}]},
        {
            "role": "tool",
            "content": [{"type": "text", "content": "untrusted result"}],
            "error": None,
        },
    ]

    wire = SecOPDVLLM.format_wire_messages(messages, _Runtime())

    assert [message["role"] for message in wire] == ["user", "input"]
    assert "untrusted result" in wire[1]["content"]
    assert "untrusted result" not in wire[0]["content"]


def test_tool_errors_are_also_input_messages():
    messages = [
        {
            "role": "tool",
            "content": [{"type": "text", "content": ""}],
            "error": "external failure",
        }
    ]

    wire = SecOPDVLLM.format_wire_messages(messages, _Runtime())

    assert wire[0]["role"] == "input"
    assert json.loads(wire[0]["content"]) == {"error": "external failure"}
