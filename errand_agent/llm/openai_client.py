"""The real model, called through the official OpenAI SDK's Responses API.

The agent speaks the content-block shape in base.py, so this adapter translates both ways.
"""
from __future__ import annotations

import json

import openai

from ..config import DEFAULT_MODEL
from .base import LLMError, LLMRequest, LLMResponse, text_block


class OpenAILLM:
    def __init__(self, model: str = DEFAULT_MODEL, *, timeout_s: float = 40.0, effort: str = "low",
                 base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.effort = effort
        # One SDK retry at most: timeout x (retries + 1) has to fit inside the run's wall-clock limit.
        self.client = openai.OpenAI(timeout=timeout_s, max_retries=1, base_url=base_url, api_key=api_key)
        # Each tool-calling response's raw output items, by its first call ID. Reasoning items must go back
        # unchanged, and keeping them here keeps their encrypted content out of traces, which forget can't scrub.
        self._replay: dict[str, list[dict]] = {}

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=request.system,
                input=self._to_input(request.messages),
                tools=[to_function_tool(tool) for tool in request.tools],
                max_output_tokens=request.max_tokens,
                reasoning={"effort": self.effort},
                store=False,
            )
        except openai.RateLimitError as exc:
            # Running out of credit is also a 429, but waiting won't fix it.
            code = "insufficient_quota" if exc.code == "insufficient_quota" else "rate_limited"
            raise LLMError(code, str(exc)) from exc
        except openai.APITimeoutError as exc:
            raise LLMError("timeout", str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise LLMError("connection", str(exc)) from exc
        except openai.AuthenticationError as exc:
            raise LLMError("auth", str(exc)) from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"http_{exc.status_code}", str(exc)) from exc
        return self._parse(response)

    def _to_input(self, messages: list[dict]) -> list[dict]:
        items: list[dict] = []
        replayed: dict[str, list[dict]] = {}
        for message in messages:
            role, content = message["role"], message["content"]
            if isinstance(content, str):
                items.append({"role": role, "content": content})
                continue
            first_call = next((b["id"] for b in content if b.get("type") == "tool_use"), None)
            if first_call in self._replay:
                replayed[first_call] = self._replay[first_call]
                items.extend(replayed[first_call])
                continue
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    items.append({"role": role, "content": block["text"]})
                elif kind == "tool_use":
                    items.append({"type": "function_call", "call_id": block["id"], "name": block["name"],
                                  "arguments": json.dumps(block["input"])})
                elif kind == "tool_result":
                    output = str(block.get("content", ""))
                    items.append({"type": "function_call_output", "call_id": block["tool_use_id"],
                                  "output": f"ERROR: {output}" if block.get("is_error") else output})
        # Runs don't share tool calls, so only this run's items are worth keeping.
        self._replay = replayed
        return items

    def _parse(self, response) -> LLMResponse:
        if response.status == "failed":
            code = response.error.code if response.error else "unknown"
            raise LLMError(f"response_failed_{code}", response.error.message if response.error else "")
        incomplete = response.status == "incomplete"
        content: list[dict] = []
        refused = False
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        content.append(text_block(part.text))
                    elif part.type == "refusal":
                        refused = True
            # A cut-off call may have truncated arguments, so it never reaches the dispatcher.
            elif item.type == "function_call" and not incomplete:
                content.append({"type": "tool_use", "id": item.call_id, "name": item.name,
                                "input": parse_arguments(item.name, item.arguments)})
        calls = [block["id"] for block in content if block["type"] == "tool_use"]
        if calls:
            self._replay[calls[0]] = [item.model_dump(mode="json", exclude_none=True) for item in response.output]
        reason = response.incomplete_details.reason if incomplete and response.incomplete_details else None
        if refused or reason == "content_filter":
            stop_reason = "refusal"
        elif incomplete:
            stop_reason = "max_tokens"
        else:
            stop_reason = "tool_use" if calls else "end_turn"
        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            model=response.model,
        )


def to_function_tool(tool: dict) -> dict:
    # Not strict: strict mode needs every optional field sent as null, and these schemas mean "send only what changed".
    return {"type": "function", "name": tool["name"], "description": tool["description"],
            "parameters": tool["input_schema"], "strict": False}


def parse_arguments(name: str, arguments: str) -> dict:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise LLMError("invalid_tool_arguments", f"{name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("invalid_tool_arguments", f"{name}: expected an object")
    # A null means "not provided", and must not overwrite a known field.
    return {key: value for key, value in parsed.items() if value is not None}
