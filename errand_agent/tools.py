"""Tool invocation: per-call timeouts, error normalisation, tracing, and run budgets.

Nothing above this layer sees a tool exception. A timed-out or crashed tool becomes a ToolResult,
so every failure takes an explicit path. Side-effect tools cannot be called without an
`Authorization`, which only the approval gate issues.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from .approval import Authorization
from .config import Limits
from .mocks import MockWorld, ServiceUnavailable, ToolTimeout

READ_TOOLS = {"search_providers", "get_provider", "get_booking_status", "list_bookings"}
HOLD_TOOL = "hold_slot"
SIDE_EFFECT_TOOLS = {"book", "cancel", "reschedule"}


class BudgetExceeded(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


class UnauthorizedSideEffect(Exception):
    pass


class RunBudget:
    """Hard limits for one run. The run's wall clock also bounds side-effect recovery."""

    def __init__(self, limits: Limits, clock):
        self.limits = limits
        self.clock = clock
        self.started = clock.monotonic()
        self.model_turns = 0
        self.tool_calls = 0

    def elapsed(self) -> float:
        return self.clock.monotonic() - self.started

    def check_time(self) -> None:
        if self.elapsed() > self.limits.max_wall_clock_s:
            raise BudgetExceeded("limit_wall_clock", f"run exceeded {self.limits.max_wall_clock_s}s")

    def take_model_turn(self) -> None:
        self.check_time()
        if self.model_turns >= self.limits.max_model_turns:
            raise BudgetExceeded("limit_model_turns", f"run exceeded {self.limits.max_model_turns} model turns")
        self.model_turns += 1

    def take_tool_call(self) -> None:
        self.check_time()
        if self.tool_calls >= self.limits.max_tool_calls:
            raise BudgetExceeded("limit_tool_calls", f"run exceeded {self.limits.max_tool_calls} tool calls")
        self.tool_calls += 1

    def out_of_time(self) -> bool:
        return self.elapsed() > self.limits.max_wall_clock_s


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None  # timeout | unavailable | unexpected
    detail: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1


class ToolRunner:
    def __init__(self, world: MockWorld, limits: Limits, clock):
        self.world = world
        self.limits = limits
        self.clock = clock

    def read(self, name: str, args: dict, budget: RunBudget, tracer, retries: int | None = None) -> ToolResult:
        """Call a read or hold tool, retrying transient failures. Holds retry too: an unanswered
        hold just expires, and the agent only ever acts on the hold ID it was given."""
        if name not in READ_TOOLS and name != HOLD_TOOL:
            raise UnauthorizedSideEffect(name)
        retries = self.limits.read_retries if retries is None else retries
        attempt = 0
        while True:
            attempt += 1
            budget.take_tool_call()
            result = self._invoke(name, args, tracer, attempt)
            result.attempts = attempt
            if result.ok or result.error == "unexpected" or attempt > retries:
                return result
            self.clock.sleep(self.limits.backoff_base_s * (2 ** (attempt - 1)))

    def side_effect(self, name: str, args: dict, authorization: Authorization, tracer) -> ToolResult:
        """Exactly one attempt; retry policy belongs to the executor, which verifies first."""
        if name not in SIDE_EFFECT_TOOLS:
            raise UnauthorizedSideEffect(name)
        if not isinstance(authorization, Authorization) or not authorization.permits(name, args):
            tracer.event("guard_blocked", guard="side_effect_authorization", tool=name)
            raise UnauthorizedSideEffect(f"{name} requires a matching authorization")
        return self._invoke(name, args, tracer, attempt=1)

    def _invoke(self, name: str, args: dict, tracer, attempt: int) -> ToolResult:
        tracer.tool_calls += 1
        started = time.perf_counter()
        outcome: dict[str, Any] = {}

        def target() -> None:
            try:
                outcome["data"] = getattr(self.world, name)(**args)
            except ToolTimeout as exc:
                outcome["error"], outcome["detail"] = "timeout", str(exc)
            except ServiceUnavailable as exc:
                outcome["error"], outcome["detail"] = "unavailable", str(exc)
            except Exception as exc:  # noqa: BLE001 - an unexpected tool error must not take down the run
                outcome["error"], outcome["detail"] = "unexpected", f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=target, daemon=True, name=f"tool-{name}")
        thread.start()
        thread.join(self.limits.tool_timeout_s)
        latency = (time.perf_counter() - started) * 1000
        if thread.is_alive():
            result = ToolResult(False, error="timeout", detail=f"no response in {self.limits.tool_timeout_s}s",
                                latency_ms=latency)
        elif "data" in outcome:
            result = ToolResult(True, data=outcome["data"], latency_ms=latency)
        else:
            result = ToolResult(False, error=outcome["error"], detail=outcome["detail"], latency_ms=latency)
        tracer.event("tool_call", tool=name, args=args, attempt=attempt, ok=result.ok, error=result.error,
                     detail=result.detail, latency_ms=round(latency, 1), result=_for_trace(name, result.data))
        return result


def _for_trace(name: str, data: Any) -> Any:
    """Trace-safe result: free-text reviews are recorded as data, clipped and marked."""
    if name == "get_provider" and isinstance(data, dict) and "reviews" in data:
        reviews = [{"rating": r.get("rating"), "untrusted_text": str(r.get("text", ""))[:160]} for r in data["reviews"]]
        return {k: v for k, v in data.items() if k != "reviews"} | {"reviews": reviews}
    return data
