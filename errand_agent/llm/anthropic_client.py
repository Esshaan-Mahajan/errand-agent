"""The real model, called through the official Anthropic SDK."""
from __future__ import annotations

import anthropic

from .base import LLMError, LLMRequest, LLMResponse

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    def __init__(self, model: str = "claude-opus-5-5", *, timeout_s: float = 30.0, effort: str = "medium",
                 base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.effort = effort
        # One SDK retry at most: timeout x (retries + 1) has to fit inside the run's wall-clock limit.
        self.client = anthropic.Anthropic(timeout=timeout_s, max_retries=1, base_url=base_url, api_key=api_key)

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=request.messages,
                tools=request.tools,
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            raise LLMError("rate_limited", str(exc)) from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError("timeout", str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("connection", str(exc)) from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("auth", str(exc)) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"http_{exc.status_code}", str(exc)) from exc
        # Blocks go back unchanged on the next turn: thinking blocks must be replayed as received.
        return LLMResponse(
            content=[block.model_dump(exclude_none=True) for block in response.content],
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )
