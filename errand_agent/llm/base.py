"""The model interface. Requests and responses use the Messages API's content-block shape."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMRequest:
    system: str
    messages: list[dict]
    tools: list[dict]
    max_tokens: int = 16000


@dataclass
class LLMResponse:
    content: list[dict]
    stop_reason: str | None = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "unknown"
    metadata: dict = field(default_factory=dict)

    @property
    def tool_uses(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    @property
    def text(self) -> str:
        return "\n".join(b["text"] for b in self.content if b.get("type") == "text").strip()


class LLMError(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = code
        self.detail = detail


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def tool_block(name: str, **arguments) -> dict:
    return {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:12]}", "name": name, "input": arguments}
