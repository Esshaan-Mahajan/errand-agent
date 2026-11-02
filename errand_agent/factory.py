"""Builds an agent from the command line's choices. Shared by the CLI, the web UI and the trace exporter."""
from __future__ import annotations

import os
from pathlib import Path

from .agent import Agent
from .clock import SystemClock
from .config import DEFAULT_MODEL
from .llm.offline import StandInModel
from .mocks import WorldConfig

LIVE_KEY_HELP = "--live needs OPENAI_API_KEY to be set (ANTHROPIC_API_KEY for claude- models)."

# Named failure scenarios for the demo. Each one scripts the mock world, nothing else.
DEMO_SCENARIOS = {
    "default": WorldConfig(),
    "happy": WorldConfig(book_script=["success"]),
    "timeout-after-success": WorldConfig(book_script=["timeout_booked"]),
    "timeout-not-booked": WorldConfig(book_script=["timeout_not_booked", "success"]),
    "slot-taken": WorldConfig(book_script=["slot_taken"]),
    "price-change": WorldConfig(quote_script=[150], book_script=["success"]),
    "price-change-after-approval": WorldConfig(quote_script=[0, 300], book_script=["success"]),
    # Offline, this also swaps in a stand-in model that obeys injected text, to show the defences hold.
    "prompt-injection": WorldConfig(book_script=["success"]),
}


def make_llm(live: bool, model: str | None = None, scenario: str = "default"):
    if not live:
        return StandInModel(gullible=scenario == "prompt-injection"), "standin-offline"
    name = model or DEFAULT_MODEL
    if name.startswith("claude-"):
        from .llm.anthropic_client import AnthropicLLM
        return AnthropicLLM(model=name), name
    from .llm.openai_client import OpenAILLM
    return OpenAILLM(model=name), name


def build_agent(data_dir: str | Path, *, live: bool = False, model: str | None = None, scenario: str = "default",
                trace_dir: str | Path = "traces", on_trace_event=None, clock=None) -> Agent:
    llm, name = make_llm(live, model, scenario)
    config = DEMO_SCENARIOS[scenario]
    return Agent(data_dir, llm, clock=clock or SystemClock(), world_config=config, trace_dir=trace_dir,
                 model_name=name, on_trace_event=on_trace_event)


def live_available(model: str | None = None) -> bool:
    if (model or DEFAULT_MODEL).startswith("claude-"):
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    return bool(os.environ.get("OPENAI_API_KEY"))
