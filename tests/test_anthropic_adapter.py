"""The real SDK adapter, exercised end to end against a local server speaking the Messages wire format.

This verifies the request shape and response handling without a key or network. It can't verify
model behaviour; `python -m errand_agent.cli --live --model claude-opus-5-5` does that with a real key.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("anthropic")

from errand_agent.agent import Agent  # noqa: E402
from errand_agent.clock import FakeClock  # noqa: E402
from errand_agent.llm.anthropic_client import AnthropicLLM  # noqa: E402
from errand_agent.llm.base import LLMError, LLMRequest  # noqa: E402

USAGE = {"input_tokens": 120, "output_tokens": 30}


def message(content, stop_reason="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": content,
            "stop_reason": stop_reason, "stop_sequence": None, "usage": USAGE}


def client(api):
    return AnthropicLLM(base_url=api.url, api_key="test-key", timeout_s=5.0)


def test_request_shape_and_response_parsing(fake_api):
    api = fake_api([(200, message([{"type": "text", "text": "hello"}]))])
    tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    response = client(api).complete(LLMRequest("sys", [{"role": "user", "content": "hi"}], tools))
    body = api.requests[0]
    assert body["model"] == "claude-opus-5-5"
    assert body["system"] == "sys"
    assert body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default"
    assert body["tools"][0]["name"] == "t"
    assert "server-side-fallback-2026-07-01" in api.headers[0]["anthropic-beta"]
    assert response.text == "hello"
    assert (response.input_tokens, response.output_tokens) == (120, 30)


def test_thinking_blocks_survive_for_replay(fake_api):
    thinking = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    tool = {"type": "tool_use", "id": "toolu_1", "name": "list_bookings", "input": {}}
    api = fake_api([(200, message([thinking, tool], "tool_use"))])
    response = client(api).complete(LLMRequest("s", [{"role": "user", "content": "hi"}], []))
    assert response.content[0]["signature"] == "sig-abc"
    assert response.tool_uses[0]["name"] == "list_bookings"


@pytest.mark.parametrize("status,code", [(400, "http_400"), (401, "auth"), (429, "rate_limited"), (500, "http_500")])
def test_api_errors_become_llm_errors(fake_api, status, code):
    error = {"type": "error", "error": {"type": "api_error", "message": "nope"}}
    api = fake_api([(status, error), (status, error), (status, error)])
    with pytest.raises(LLMError) as exc:
        client(api).complete(LLMRequest("s", [{"role": "user", "content": "hi"}], []))
    assert exc.value.code == code


def test_unreachable_server_becomes_llm_error():
    llm = AnthropicLLM(base_url="http://127.0.0.1:9", api_key="k", timeout_s=1.0)
    with pytest.raises(LLMError):
        llm.complete(LLMRequest("s", [{"role": "user", "content": "hi"}], []))


def test_agent_loop_over_the_real_adapter(fake_api, tmp_path):
    """A full round trip: tool_use blocks in, tool_result blocks out, cost in the trace."""
    def tool(name, **arguments):
        return {"type": "tool_use", "id": f"toolu_{name}", "name": name, "input": arguments}
    thinking = {"type": "thinking", "thinking": "", "signature": "sig-1"}
    api = fake_api([
        (200, message([thinking, tool("update_request", service="plumber", area="indiranagar", date_from="2026-11-03",
                                      date_to="2026-11-06", budget=5000)], "tool_use")),
        (200, message([tool("ask_user", fields=[], question="Any preferred time?")], "tool_use")),
    ])
    agent = Agent(tmp_path, client(api), clock=FakeClock(), trace_dir=tmp_path / "traces")
    reply = agent.handle_message("user-1", "Plumber in Indiranagar this week")
    assert reply.outcome == "awaiting_user" and reply.text == "Any preferred time?"
    second = api.requests[1]["messages"]
    assert second[1]["role"] == "assistant" and second[1]["content"][0]["signature"] == "sig-1"
    assert second[2]["content"][0]["type"] == "tool_result"
    assert second[2]["content"][0]["tool_use_id"] == "toolu_update_request"
    with open(reply.trace_path, encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle]
    calls = [e for e in events if e["type"] == "model_call"]
    assert calls[0]["input_tokens"] == 120 and calls[0]["estimated_cost_usd"] == pytest.approx(0.00108)
    agent.close()
