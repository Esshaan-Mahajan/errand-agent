"""The real SDK adapter, exercised end to end against a local server speaking the Responses wire format.

This verifies the request shape and response handling without a key or network. It can't verify
model behaviour; `python -m errand_agent.cli --live` does that with a real key.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("openai")

from errand_agent.agent import Agent  # noqa: E402
from errand_agent.clock import FakeClock  # noqa: E402
from errand_agent.config import estimate_cost_usd  # noqa: E402
from errand_agent.factory import live_available, make_llm  # noqa: E402
from errand_agent.llm.base import LLMError, LLMRequest  # noqa: E402
from errand_agent.llm.openai_client import OpenAILLM  # noqa: E402

USER = [{"role": "user", "content": "hi"}]
REASONING = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "SECRET-BLOB"}


def response(output, status="completed", incomplete_reason=None, error=None):
    return {"id": "resp_1", "object": "response", "created_at": 1, "model": "gpt-6.1-terra", "status": status,
            "output": output, "error": error, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
            "usage": {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}}


def message(text):
    return {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def call(name, call_id=None, arguments="{}"):
    return {"id": f"fc_{name}", "type": "function_call", "call_id": call_id or f"call_{name}", "name": name,
            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments), "status": "completed"}


def client(api):
    return OpenAILLM(base_url=api.url, api_key="test-key", timeout_s=5.0)


def test_request_shape_and_response_parsing(fake_api):
    api = fake_api([(200, response([message("hello")]))])
    tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    reply = client(api).complete(LLMRequest("sys", USER, tools, max_tokens=500))
    body = api.requests[0]
    assert api.paths[0].endswith("/responses")
    assert body["model"] == "gpt-6.1-terra"
    assert body["instructions"] == "sys"
    assert body["input"] == USER
    assert body["tools"] == [{"type": "function", "name": "t", "description": "d",
                              "parameters": {"type": "object"}, "strict": False}]
    assert body["max_output_tokens"] == 500
    assert body["reasoning"] == {"effort": "low"}
    assert body["store"] is False
    assert reply.text == "hello" and reply.stop_reason == "end_turn"
    assert (reply.input_tokens, reply.output_tokens) == (120, 30)


def test_function_calls_become_tool_use_blocks(fake_api):
    api = fake_api([(200, response([REASONING, call("list_bookings", arguments={"a": 1})]))])
    reply = client(api).complete(LLMRequest("s", USER, []))
    assert reply.stop_reason == "tool_use"
    assert reply.content == [{"type": "tool_use", "id": "call_list_bookings", "name": "list_bookings", "input": {"a": 1}}]


def test_null_arguments_do_not_overwrite_known_fields(fake_api):
    api = fake_api([(200, response([call("update_request", arguments={"area": None, "budget": 900})]))])
    reply = client(api).complete(LLMRequest("s", USER, []))
    assert reply.tool_uses[0]["input"] == {"budget": 900}


@pytest.mark.parametrize("arguments", ['{"area": "indira', "[1, 2]"])
def test_malformed_arguments_become_llm_errors(fake_api, arguments):
    api = fake_api([(200, response([call("update_request", arguments=arguments)]))])
    with pytest.raises(LLMError) as exc:
        client(api).complete(LLMRequest("s", USER, []))
    assert exc.value.code == "invalid_tool_arguments"


def test_truncated_response_never_runs_its_tool_call(fake_api):
    api = fake_api([(200, response([call("hold_slot", arguments='{"provider_id": "p')], "incomplete", "max_output_tokens"))])
    reply = client(api).complete(LLMRequest("s", USER, []))
    assert reply.stop_reason == "max_tokens" and reply.tool_uses == []


def test_refusals_are_reported_as_refusals(fake_api):
    refusal = {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "refusal", "refusal": "no"}]}
    api = fake_api([(200, response([refusal])), (200, response([], "incomplete", "content_filter"))])
    llm = client(api)
    assert llm.complete(LLMRequest("s", USER, [])).stop_reason == "refusal"
    assert llm.complete(LLMRequest("s", USER, [])).stop_reason == "refusal"


def test_failed_response_becomes_llm_error(fake_api):
    error = {"code": "server_error", "message": "broke"}
    api = fake_api([(200, response([], "failed", error=error))])
    with pytest.raises(LLMError) as exc:
        client(api).complete(LLMRequest("s", USER, []))
    assert exc.value.code == "response_failed_server_error"


@pytest.mark.parametrize("status,api_code,code", [
    (400, None, "http_400"), (401, None, "auth"), (404, "model_not_found", "http_404"), (429, None, "rate_limited"),
    (429, "insufficient_quota", "insufficient_quota"), (500, None, "http_500"),
])
def test_api_errors_become_llm_errors(fake_api, status, api_code, code):
    error = {"error": {"message": "nope", "type": "x", "param": None, "code": api_code}}
    api = fake_api([(status, error), (status, error), (status, error)])
    with pytest.raises(LLMError) as exc:
        client(api).complete(LLMRequest("s", USER, []))
    assert exc.value.code == code


def test_unreachable_server_becomes_llm_error():
    llm = OpenAILLM(base_url="http://127.0.0.1:9", api_key="k", timeout_s=1.0)
    with pytest.raises(LLMError):
        llm.complete(LLMRequest("s", USER, []))


def test_history_without_raw_items_is_translated(fake_api):
    api = fake_api([(200, response([message("ok")]))])
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "looking"},
                                          {"type": "tool_use", "id": "c1", "name": "t", "input": {"a": 1}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "bad", "is_error": True}]},
    ]
    client(api).complete(LLMRequest("s", messages, []))
    assert api.requests[0]["input"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "looking"},
        {"type": "function_call", "call_id": "c1", "name": "t", "arguments": '{"a": 1}'},
        {"type": "function_call_output", "call_id": "c1", "output": "ERROR: bad"},
    ]


def test_old_replay_items_are_dropped_once_a_new_run_starts(fake_api):
    api = fake_api([(200, response([REASONING, call("t")])), (200, response([message("ok")]))])
    llm = client(api)
    llm.complete(LLMRequest("s", USER, []))
    assert list(llm._replay) == ["call_t"]
    llm.complete(LLMRequest("s", USER, []))
    assert llm._replay == {}


def test_agent_loop_over_the_real_adapter(fake_api, tmp_path):
    """A full round trip: reasoning goes back unchanged, but never reaches the trace."""
    update = call("update_request", arguments={"service": "plumber", "area": "indiranagar", "date_from": "2026-11-03",
                                               "date_to": "2026-11-06", "budget": 5000})
    ask = call("ask_user", arguments={"fields": [], "question": "Any preferred time?"})
    api = fake_api([(200, response([REASONING, update])), (200, response([ask]))])
    agent = Agent(tmp_path, client(api), clock=FakeClock(), trace_dir=tmp_path / "traces")
    reply = agent.handle_message("user-1", "Plumber in Indiranagar this week")
    assert reply.outcome == "awaiting_user" and reply.text == "Any preferred time?"
    second = api.requests[1]["input"]
    assert second[-3]["encrypted_content"] == "SECRET-BLOB" and second[-3]["id"] == "rs_1"
    assert second[-2]["id"] == "fc_update_request" and second[-2]["call_id"] == "call_update_request"
    assert second[-1]["type"] == "function_call_output" and second[-1]["call_id"] == "call_update_request"
    trace = open(reply.trace_path, encoding="utf-8").read()
    assert "SECRET-BLOB" not in trace
    calls = [e for e in map(json.loads, trace.splitlines()) if e["type"] == "model_call"]
    assert calls[0]["input_tokens"] == 120 and calls[0]["estimated_cost_usd"] == pytest.approx(0.00027)
    agent.close()


def test_cost_estimate_recognises_a_dated_snapshot():
    assert estimate_cost_usd("gpt-6.1-terra-2026-10-01", 1_000_000, 0) == 1.00
    assert estimate_cost_usd("standin-offline", 1_000_000, 0) is None


def test_live_model_routing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert not live_available()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert not live_available() and live_available("claude-opus-5-5")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    llm, name = make_llm(True)
    assert isinstance(llm, OpenAILLM) and name == "gpt-6.1-terra"
    assert live_available()
