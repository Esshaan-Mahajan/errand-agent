"""Hard limits and static configuration. Every bound the agent obeys lives here."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    # One user message (a "run").
    max_model_turns: int = 8
    max_tool_calls: int = 16
    max_wall_clock_s: float = 90.0
    # One tool call / one model call.
    tool_timeout_s: float = 10.0
    llm_timeout_s: float = 45.0
    # Transient read failures are retried this many extra times.
    read_retries: int = 2
    # Side effects.
    max_side_effect_attempts: int = 3
    status_poll_attempts: int = 4
    backoff_base_s: float = 0.5
    approval_ttl_s: float = 15 * 60
    # Ingress and context.
    max_user_input_chars: int = 4000
    context_token_budget: int = 6000
    memory_token_share: float = 0.10
    history_token_share: float = 0.20
    max_tool_result_chars: int = 3600
    max_holds_per_workflow: int = 5


DEFAULT_MODEL = "gpt-6.1-terra"

# USD per million tokens (input, output). Used only for the trace's cost estimate.
PRICING = {
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return the estimated cost, or None for models with no price (offline stand-ins)."""
    # A response can name a dated snapshot of the model that was requested.
    price = PRICING.get(model) or PRICING.get(re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model))
    if price is None:
        return None
    return round((input_tokens * price[0] + output_tokens * price[1]) / 1_000_000, 6)
