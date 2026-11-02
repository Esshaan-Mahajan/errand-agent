"""The orchestrator: one message in, one user-facing outcome out, on every path."""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import state as st
from .approval import ApprovalGate, Proposal, Reply, is_bare_affirmative, parse_reply, render_prompt
from .clock import SystemClock, parse
from .config import Limits, estimate_cost_usd
from .context import ContextBudgetExceeded, ContextBuilder
from .db import AGENT_SCHEMA, Database
from .executor import FAILED, SUCCEEDED, ExecOutcome, Executor
from .llm.base import LLMError, LLMRequest, LLMResponse
from .memory import CONFIRMED, EXPLICIT, INFERRED, MemoryStore, render_inspection
from .mocks import MockWorld, WorldConfig, _normalise_service, fmt_seconds
from .sanitize import clean_user_input, project_provider, safe_label
from .tool_schemas import MODEL_TOOLS
from .tools import SIDE_EFFECT_TOOLS, BudgetExceeded, RunBudget, ToolRunner
from .trace import Tracer, scrub_traces

_CLAIM = re.compile(
    r"\b(has been|have been|is now|successfully|i've|i have|all set)\b.{0,40}\b"
    r"(booked|cancelled|canceled|rescheduled|moved|confirmed)\b"
    r"|\byour\b.{0,40}\b(booking|appointment|visit)\b.{0,25}\b(is|was)\b.{0,12}\b(confirmed|booked|set|done)\b",
    re.IGNORECASE | re.DOTALL,
)
_NEGATED = re.compile(r"\b(not|nothing|never|none|cannot|unable|failed)\b|n't\b", re.IGNORECASE)
_FILLER_NEGATIONS = re.compile(r"\bno (worries|problem|issues?)\b|\bnot to worry\b|\bnothing to worry\b", re.IGNORECASE)
_NO_CHANGE = "No booking, cancellation or reschedule was made in this request."


@dataclass
class AgentReply:
    text: str
    outcome: str
    run_id: str
    trace_path: str
    state: str | None = None
    proposal_code: str | None = None


@dataclass
class _Tool:
    payload: dict
    is_error: bool = False
    end: tuple[str, str] | None = None  # (outcome, reply): this tool ends the run
    proposal: Proposal | None = None


@dataclass
class _Run:
    user_id: str
    user_text: str
    message_id: int
    tracer: Tracer
    budget: RunBudget
    read_untrusted: bool = False
    read_bookings: bool = False
    extras: dict = field(default_factory=dict)


class Agent:
    def __init__(self, data_dir: str | Path, llm, *, clock=None, limits: Limits | None = None,
                 world_config: WorldConfig | None = None, user_id: str = "user-1", crash_hook=None,
                 trace_dir: str | Path | None = None, model_name: str | None = None, on_trace_event=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir = Path(trace_dir) if trace_dir else self.data_dir / "traces"
        self.clock = clock or SystemClock()
        self.limits = limits or Limits()
        self.llm = llm
        self.model_name = model_name
        self.on_trace_event = on_trace_event
        self.db = Database(self.data_dir / "agent.db", AGENT_SCHEMA)
        self.world = MockWorld(self.data_dir / "world.db", self.clock, world_config, user_id=user_id)
        self.tools = ToolRunner(self.world, self.limits, self.clock)
        self.workflows = st.WorkflowStore(self.db, self.clock)
        self.gate = ApprovalGate(self.db, self.clock, self.limits)
        self.memory = MemoryStore(self.db, self.clock)
        self.context = ContextBuilder(self.db, self.memory, self.limits, self.clock)
        self.executor = Executor(self.db, self.gate, self.tools, self.workflows, self.clock, self.limits, crash_hook)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def close(self) -> None:
        self.db.close()
        self.world.db.close()

    # ------------------------------------------------------------------ public entry point

    def handle_message(self, user_id: str, text: str) -> AgentReply:
        # One run per user at a time, so two quick confirmations can't interleave.
        with self._lock_for(user_id):
            tracer = self.new_tracer(user_id)
            try:
                return self._handle(user_id, text, tracer)
            except Exception as exc:  # noqa: BLE001 - every exit path must produce a user-facing outcome
                tracer.event("internal_error", error=f"{type(exc).__name__}: {exc}")
                reply = ("Something went wrong on my side, so I stopped. I haven't reported any change that I "
                         "couldn't verify, and anything in progress will be re-checked next time you message me.")
                return self._finish(user_id, tracer, "internal_error", reply)

    def new_tracer(self, user_id: str, kind: str = "message") -> Tracer:
        return Tracer(self.trace_dir, self.clock, user_id, kind=kind, on_event=self.on_trace_event)

    def add_message(self, user_id: str, role: str, text: str, run_id: str | None, private: bool = False) -> int:
        """`private` messages are shown to the user but never enter the model's context."""
        cursor = self.db.execute("INSERT INTO messages (user_id, role, text, run_id, private, created_at) VALUES (?,?,?,?,?,?)",
                                 (user_id, role, text, run_id, int(private), fmt_seconds(self.clock.now())))
        return cursor.lastrowid

    def _lock_for(self, user_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(user_id, threading.Lock())

    # ------------------------------------------------------------------ the deterministic front door

    def _handle(self, user_id: str, raw_text: str, tracer: Tracer) -> AgentReply:
        text, truncated = clean_user_input(raw_text, self.limits.max_user_input_chars)
        if not text:
            return self._finish(user_id, tracer, "awaiting_user", "I didn't catch that. What would you like to arrange?")
        message_id = self.add_message(user_id, "user", text, tracer.run_id)
        tracer.event("user_message", message_id=message_id, text=text, truncated=truncated)
        prefix = (f"(Your message was very long, so I only read the first {self.limits.max_user_input_chars} "
                  "characters.)\n") if truncated else ""

        # Recovery comes first: a confirmation must see the true state of anything left in flight.
        recovered = self.executor.recover(user_id, tracer)
        notes = []
        for outcome in recovered:
            if outcome.proposal:
                return self._show_proposal(user_id, tracer, outcome.proposal, intro=outcome.message)
            notes.append(outcome.message)
        if notes and is_bare_affirmative(text):
            outcome = recovered[-1]
            return self._finish(user_id, tracer, "execution_" + outcome.status, prefix + "\n".join(notes))
        if notes:
            prefix += "\n".join(notes) + "\n\n"

        pending = self.gate.pending(user_id)
        if pending is not None:
            verdict = parse_reply(text, pending.code)
            tracer.event("reply_classified", verdict=verdict.value, proposal_id=pending.id)
            if verdict is Reply.APPROVE:
                return self._on_approve(user_id, pending, message_id, tracer, prefix)
            if verdict is Reply.REJECT:
                return self._on_reject(user_id, pending, tracer, prefix)
        elif is_bare_affirmative(text):
            handled = self._repeat_confirmation(user_id, tracer, prefix)
            if handled is not None:
                return handled
        return self._run_model(user_id, text, message_id, tracer, prefix)

    def _on_approve(self, user_id, pending: Proposal, message_id, tracer, prefix) -> AgentReply:
        approval = self.gate.approve(pending, message_id, tracer)
        if approval is None:
            return self._finish(user_id, tracer, "awaiting_approval", prefix + render_prompt(pending),
                                proposal=pending)
        self.executor.crash_hook("after_approval")
        outcome = self.executor.execute(approval, pending, tracer)
        return self._report_execution(user_id, tracer, outcome, prefix)

    def _on_reject(self, user_id, pending: Proposal, tracer, prefix) -> AgentReply:
        self.gate.reject(pending, tracer)
        workflow = self.workflows.get(pending.workflow_id)
        if workflow and workflow.state == st.AWAITING_APPROVAL:
            self.workflows.transition(workflow, st.COMPARING, "user rejected proposal", tracer)
        reply = prefix + "Okay, I won't go ahead with that, and nothing was changed. Want me to look at other options?"
        return self._finish(user_id, tracer, "rejected", reply)

    def _repeat_confirmation(self, user_id, tracer, prefix) -> AgentReply | None:
        """A "yes" with nothing pending must never turn into a second side effect."""
        record = self.executor.latest_for_user(user_id)
        if record is None:
            return None
        tracer.event("repeat_confirmation", idempotency_key=record["idempotency_key"], status=record["status"])
        if record["status"] == SUCCEEDED:
            reply = f"That's already done, so I haven't repeated it. {self.executor.replay_message(record)}"
        elif record["status"] == FAILED:
            reply = f"That one didn't go through and I need a fresh request to retry. {self.executor.replay_message(record)}"
        else:
            reply = self.executor.replay_message(record)
        return self._finish(user_id, tracer, "repeat_confirmation", prefix + reply)

    def _report_execution(self, user_id, tracer, outcome: ExecOutcome, prefix: str) -> AgentReply:
        if outcome.proposal:
            return self._show_proposal(user_id, tracer, outcome.proposal, intro=prefix + outcome.message)
        return self._finish(user_id, tracer, f"execution_{outcome.status}", prefix + outcome.message)

    def _show_proposal(self, user_id, tracer, proposal: Proposal, intro: str | None = None) -> AgentReply:
        return self._finish(user_id, tracer, "awaiting_approval", render_prompt(proposal, intro), proposal=proposal)

    def _finish(self, user_id, tracer, outcome: str, reply: str, proposal: Proposal | None = None) -> AgentReply:
        # A memory listing can include unconfirmed inferences, which must not reach the model via history.
        message_id = self.add_message(user_id, "assistant", reply, tracer.run_id, private=outcome == "memory_shown")
        if proposal is not None:
            self.gate.mark_shown(proposal.id, message_id)
        workflow = self.workflows.active(user_id)
        state = workflow.state if workflow else None
        tracer.finish(outcome, reply, workflow_state=state, proposal=proposal.code if proposal else None)
        return AgentReply(reply, outcome, tracer.run_id, str(tracer.path), state, proposal.code if proposal else None)

    # ------------------------------------------------------------------ the bounded model loop

    def _run_model(self, user_id, text, message_id, tracer, prefix_text) -> AgentReply:
        run = _Run(user_id, text, message_id, tracer, RunBudget(self.limits, self.clock))
        workflow = self.workflows.active(user_id)
        context = self.context.build_prefix(user_id, workflow, text, message_id, MODEL_TOOLS)
        run_messages: list[dict] = [{"role": "user", "content": text}]
        try:
            while True:
                run.budget.take_model_turn()
                messages, total = self.context.assemble(context, run_messages)
                tracer.event("context_sent", turn=run.budget.model_turns, budget=self.limits.context_token_budget,
                             estimated_tokens=total, breakdown=context.tokens,
                             memories_included=context.memory_included, memories_excluded=context.memory_excluded,
                             history_message_ids=context.history_message_ids,
                             excerpt_message_ids=context.excerpt_message_ids,
                             payload={"system": context.system, "messages": messages})
                response = self._call_model(context.system, messages, run)
                if response.stop_reason == "refusal":
                    return self._finish(user_id, tracer, "model_refusal", prefix_text +
                                        f"The model declined to continue. {_NO_CHANGE}")
                run_messages.append({"role": "assistant", "content": response.content})
                uses = response.tool_uses
                if not uses:
                    if response.stop_reason == "max_tokens":
                        return self._finish(user_id, tracer, "model_truncated", prefix_text +
                                            f"My reply was cut off before I finished. {_NO_CHANGE} Please try again.")
                    reply = self._guard_final_text(response.text, run)
                    return self._finish(user_id, tracer, "completed", prefix_text + reply)
                ended = self._run_tools(uses, context, run_messages, run)
                if ended is not None:
                    outcome, reply, proposal = ended
                    return self._finish(user_id, tracer, outcome, prefix_text + reply, proposal=proposal)
        except BudgetExceeded as exc:
            tracer.event("limit_reached", code=exc.code, detail=exc.detail, scope="run",
                         model_turns=run.budget.model_turns, tool_calls=run.budget.tool_calls,
                         elapsed_s=round(run.budget.elapsed(), 2))
            what = {"limit_model_turns": "I hit my limit on reasoning steps",
                    "limit_tool_calls": "I hit my limit on tool calls",
                    "limit_wall_clock": "this was taking too long"}[exc.code]
            return self._finish(user_id, tracer, exc.code, prefix_text +
                                f"I stopped because {what}. {_NO_CHANGE} Could you try again, perhaps with a narrower request?")
        except ContextBudgetExceeded as exc:
            tracer.event("limit_reached", code="context_budget", detail=str(exc), scope="run")
            return self._finish(user_id, tracer, "context_budget", prefix_text +
                                f"This request grew too large for me to process safely. {_NO_CHANGE} "
                                "Could you break it into a smaller step?")
        except LLMError as exc:
            tracer.event("model_error", code=exc.code, detail=exc.detail)
            return self._finish(user_id, tracer, "model_error", prefix_text +
                                f"I can't reach my language model right now ({exc.code}). {_NO_CHANGE} "
                                "Please try again shortly.")

    def _call_model(self, system, messages, run: _Run) -> LLMResponse:
        """Bounded even if the client ignores its own timeout: the model runs in a daemon thread."""
        request = LLMRequest(system=system, messages=messages, tools=MODEL_TOOLS)
        box: dict = {}

        def target() -> None:
            try:
                box["response"] = self.llm.complete(request)
            except LLMError as exc:
                box["error"] = exc
            except Exception as exc:  # noqa: BLE001 - malformed clients surface as a model error
                box["error"] = LLMError("client_error", f"{type(exc).__name__}: {exc}")

        started = time.perf_counter()
        thread = threading.Thread(target=target, daemon=True, name="llm-call")
        thread.start()
        remaining = self.limits.max_wall_clock_s - run.budget.elapsed()
        thread.join(max(0.0, min(remaining, self.limits.llm_timeout_s * 2 + 5)))
        latency = (time.perf_counter() - started) * 1000
        if thread.is_alive():
            raise BudgetExceeded("limit_wall_clock", "model call did not return in time")
        if "error" in box:
            raise box["error"]
        response = box["response"]
        model = response.model or self.model_name or "unknown"
        run.tracer.model_call(model=model, turn=run.budget.model_turns, latency_ms=latency,
                              input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                              cost_usd=estimate_cost_usd(model, response.input_tokens, response.output_tokens),
                              stop_reason=response.stop_reason, response=response.content)
        return response

    def _run_tools(self, uses, context, run_messages, run: _Run):
        results = []
        ended = None
        for use in uses:
            block = {"type": "tool_result", "tool_use_id": use["id"]}
            if ended is not None:
                # The API requires a result for every tool_use, even once the turn is over.
                results.append(block | {"content": json.dumps({"skipped": "turn already ended"}), "is_error": True})
                continue
            outcome = self._dispatch(use["name"], use.get("input") or {}, run)
            content = json.dumps(outcome.payload, default=str)
            room = self.context.tool_result_room_chars(context, run_messages + [{"role": "user", "content": results}])
            if len(content) > room:
                run.tracer.event("tool_result_truncated", tool=use["name"], original_chars=len(content), kept_chars=room)
                content = content[:room] + "...[truncated]"
            results.append(block | {"content": content} | ({"is_error": True} if outcome.is_error else {}))
            if outcome.end is not None:
                ended = (*outcome.end, outcome.proposal)
        run_messages.append({"role": "user", "content": results})
        return ended

    def _guard_final_text(self, text: str, run: _Run) -> str:
        """The model never gets to report a side effect, however confident it sounds."""
        if not text:
            return f"I wasn't able to produce an answer. {_NO_CHANGE}"
        if _claims_success(text) and not run.read_bookings:
            run.tracer.event("guard_blocked", guard="unverified_success_claim", text=text[:300])
            return ("I can't confirm that anything has been booked or changed: nothing has gone through in this "
                    f"request. {_NO_CHANGE} Would you like me to continue?")
        return text

    # ------------------------------------------------------------------ guarded tool dispatch

    def _dispatch(self, name: str, args: dict, run: _Run) -> _Tool:
        if name in SIDE_EFFECT_TOOLS:
            run.tracer.event("guard_blocked", guard="model_called_side_effect", tool=name, args=args)
            return _Tool({"error": "approval_required",
                          "detail": "Side effects run only after the user confirms. Use request_approval."}, True)
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            run.tracer.event("guard_blocked", guard="unknown_tool", tool=name)
            return _Tool({"error": "unknown_tool", "tool": name}, True)
        if not isinstance(args, dict):
            return _Tool({"error": "invalid_arguments"}, True)
        try:
            return handler(args, run)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            run.tracer.event("tool_argument_error", tool=name, error=f"{type(exc).__name__}: {exc}")
            return _Tool({"error": "invalid_arguments", "detail": str(exc)}, True)

    def _workflow(self, run: _Run, kind: str | None = None, new: bool = False) -> st.Workflow:
        workflow = self.workflows.active(run.user_id)
        if workflow is not None and new:
            self.gate.invalidate_pending(run.user_id, "new errand started", run.tracer)
            if workflow.state in (st.INTAKE, st.COMPARING, st.AWAITING_APPROVAL):
                self.workflows.transition(workflow, st.ABANDONED, "new errand started", run.tracer)
            workflow = None
        if workflow is None:
            workflow = self.workflows.create(run.user_id, kind, run.tracer)
        return workflow

    def _tool_update_request(self, args, run: _Run) -> _Tool:
        workflow = self._workflow(run, args.get("kind", "book"), bool(args.get("new_errand")))
        if workflow.kind is None or (args.get("kind") and args["kind"] != workflow.kind):
            workflow.kind = args.get("kind", workflow.kind or "book")
        changed = {}
        for name in st.REQUEST_FIELDS:
            if name not in args or args[name] in (None, ""):
                continue
            value = args[name]
            if name in ("date_from", "date_to"):
                date.fromisoformat(value)
            elif name in ("earliest_time", "latest_time"):
                _minutes(value)
            elif name == "budget":
                value = int(value)
                if value <= 0:
                    raise ValueError("budget must be positive")
            elif name == "service":
                value = _normalise_service(value)
            elif name == "area":
                value = safe_label(value).lower()
            if workflow.request.get(name) != value:
                changed[name] = value
        if isinstance(args.get("preferences"), dict):
            workflow.preferences.update(args["preferences"])
        if "budget" in changed:
            workflow.request.pop("budget_waived", None)
        # Only the named fields change, so the rest of the request survives a correction.
        workflow.request.update(changed)
        invalidated = False
        if changed:
            invalidated = self.gate.invalidate_pending(run.user_id, f"request changed: {sorted(changed)}", run.tracer)
            if workflow.state in (st.AWAITING_APPROVAL, st.COMPARING):
                self.workflows.transition(workflow, st.INTAKE, f"request changed: {sorted(changed)}", run.tracer)
        self.workflows.save(workflow)
        run.tracer.event("request_updated", workflow_id=workflow.id, changed=changed, request=workflow.request)
        return _Tool({"request": workflow.request, "preferences": workflow.preferences,
                      "still_missing": workflow.still_missing(), "changed": sorted(changed),
                      "previous_proposal_invalidated": invalidated})

    def _tool_ask_user(self, args, run: _Run) -> _Tool:
        workflow = self._workflow(run)
        fields = [f for f in args.get("fields", []) if isinstance(f, str)]
        known = {f: workflow.known_value(f) for f in fields if workflow.known_value(f) not in (None, "", False)}
        if known and set(known) == set(fields):
            run.tracer.event("guard_blocked", guard="reask_known_field", fields=sorted(known))
            return _Tool({"error": "already_known", "known": known, "request": workflow.request,
                          "still_missing": workflow.still_missing()}, True)
        if known:
            # Some of the fields are new. The model's wording may still mention the known ones,
            # so the question is rebuilt from the fields actually missing.
            run.tracer.event("guard_blocked", guard="reask_known_field", fields=sorted(known))
            wanted = [f.replace("_", " ") for f in fields if f not in known]
            return _Tool({"asked": wanted}, end=("awaiting_user", "Could you tell me your " + " and ".join(wanted) + "?"))
        question = str(args["question"]).strip()
        if not question:
            raise ValueError("question is empty")
        return _Tool({"asked": fields}, end=("awaiting_user", self._guard_final_text(question, run)))

    def _tool_search_providers(self, args, run: _Run) -> _Tool:
        workflow = self._workflow(run, "book")
        service = _normalise_service(args["service"])
        date_range = {"start": str(args["date_range"]["start"])[:10], "end": str(args["date_range"]["end"])[:10]}
        date.fromisoformat(date_range["start"])
        date.fromisoformat(date_range["end"])
        result = self.tools.read("search_providers", {"service": service, "area": args["area"], "date_range": date_range},
                                 run.budget, run.tracer)
        if not result.ok:
            return _Tool({"error": result.error, "detail": "provider search is unavailable"}, True)
        earliest, latest = self._time_window(workflow, run.user_id)
        projected, candidates = [], {}
        for provider in result.data["providers"]:
            inside = [s for s in provider["available_slots"] if _in_window(s, earliest, latest)]
            candidates[provider["id"]] = {"name": safe_label(provider["name"]), "service": service,
                                          "price": provider["price"], "slots": provider["available_slots"]}
            projected.append({"id": provider["id"], "name": safe_label(provider["name"]), "price": provider["price"],
                              "rating": provider["rating"], "available_slots": inside[:6],
                              "attributes": provider["attributes"]})
        workflow.candidates = candidates
        workflow.request.setdefault("service", service)
        if workflow.kind is None:
            workflow.kind = "book"
        if workflow.state == st.INTAKE:
            self.workflows.transition(workflow, st.COMPARING, "providers found", run.tracer)
        self.workflows.save(workflow)
        return _Tool({"providers": projected, "time_window": {"earliest": earliest, "latest": latest}})

    def _tool_get_provider(self, args, run: _Run) -> _Tool:
        result = self.tools.read("get_provider", {"provider_id": str(args["provider_id"])}, run.budget, run.tracer)
        if not result.ok:
            return _Tool({"error": result.error}, True)
        if "error" in result.data:
            return _Tool(result.data, True)
        projection, suspected = project_provider(result.data)
        run.read_untrusted = True
        if suspected:
            run.tracer.event("untrusted_content_flagged", provider_id=projection["id"], enforcement="none_needed",
                             note="treated as data; approval and memory rules do not depend on this flag")
        return _Tool(projection)

    def _tool_hold_slot(self, args, run: _Run) -> _Tool:
        workflow = self._workflow(run, "book")
        provider_id, slot = str(args["provider_id"]), str(args["slot"])
        candidate = workflow.candidates.get(provider_id)
        if candidate is None or slot not in candidate["slots"]:
            run.tracer.event("guard_blocked", guard="hold_outside_search_results", provider_id=provider_id, slot=slot)
            return _Tool({"error": "not_in_search_results",
                          "detail": "Only providers and slots from search_providers can be held."}, True)
        if len(workflow.holds) >= self.limits.max_holds_per_workflow:
            return _Tool({"error": "too_many_holds"}, True)
        result = self.tools.read("hold_slot", {"provider_id": provider_id, "slot": slot}, run.budget, run.tracer)
        if not result.ok:
            return _Tool({"error": result.error}, True)
        if "error" in result.data:
            return _Tool(result.data, True)
        workflow.holds[result.data["hold_id"]] = result.data
        self.workflows.save(workflow)
        return _Tool(result.data | {"price_differs_from_search": result.data["quoted_price"] != result.data["search_price"]})

    def _tool_list_bookings(self, args, run: _Run) -> _Tool:
        # The user is fixed by the session. No model or injected text can read another account.
        result = self.tools.read("list_bookings", {"user_id": run.user_id}, run.budget, run.tracer)
        if not result.ok:
            return _Tool({"error": result.error}, True)
        run.read_bookings = True
        return _Tool(result.data)

    def _tool_get_booking_status(self, args, run: _Run) -> _Tool:
        result = self.tools.read("get_booking_status", {"hold_id": args.get("hold_id"),
                                                        "idempotency_key": args.get("idempotency_key")},
                                 run.budget, run.tracer)
        if not result.ok:
            return _Tool({"error": result.error}, True)
        return _Tool(result.data)

    def _tool_request_approval(self, args, run: _Run) -> _Tool:
        action = args["action"]
        if action == "book":
            return self._propose_book(args, run)
        if action in ("cancel", "reschedule"):
            return self._propose_change(action, args, run)
        raise ValueError(f"unknown action {action}")

    def _propose_book(self, args, run: _Run) -> _Tool:
        workflow = self._workflow(run, "book")
        hold = workflow.holds.get(str(args.get("hold_id")))
        if hold is None:
            run.tracer.event("guard_blocked", guard="proposal_without_hold", hold_id=args.get("hold_id"))
            return _Tool({"error": "unknown_hold", "detail": "Call hold_slot first."}, True)
        if self.clock.now() > parse(hold["expires_at"]):
            return _Tool({"error": "hold_expired", "detail": "Hold the slot again."}, True)
        problem = self._check_constraints(workflow, hold, run)
        if problem:
            run.tracer.event("guard_blocked", guard="constraint_violation", **problem)
            return _Tool(problem, True)
        duplicate = self._find_duplicate(run, hold["provider_id"], hold["slot"])
        if duplicate:
            run.tracer.event("guard_blocked", guard="duplicate_booking", booking_id=duplicate)
            return _Tool({"error": "duplicate_booking", "booking_id": duplicate,
                          "detail": "This slot is already booked for the user."}, True)
        candidate = workflow.candidates.get(hold["provider_id"], {})
        details = {
            "provider_id": hold["provider_id"], "provider_name": candidate.get("name", hold["provider_id"]),
            "service": candidate.get("service", workflow.request.get("service")), "slot": hold["slot"],
            "price": hold["quoted_price"], "search_price": hold.get("search_price"), "hold_id": hold["hold_id"],
            "hold_expires_at": hold["expires_at"], "booking_id": None, "old_slot": None,
        }
        return self._propose(workflow, "book", details, run)

    def _propose_change(self, action, args, run: _Run) -> _Tool:
        result = self.tools.read("list_bookings", {"user_id": run.user_id}, run.budget, run.tracer)
        if not result.ok:
            return _Tool({"error": result.error}, True)
        run.read_bookings = True
        booking = next((b for b in result.data["bookings"] if b["id"] == args.get("booking_id")), None)
        if booking is None:
            # Other people's bookings look the same as ones that don't exist.
            run.tracer.event("guard_blocked", guard="booking_not_owned", booking_id=args.get("booking_id"))
            return _Tool({"error": "booking_not_found"}, True)
        if booking["status"] != "confirmed" or booking["when"] != "upcoming":
            return _Tool({"error": "booking_not_active", "status": booking["status"]}, True)
        if action == "reschedule":
            target = str(args["slot"])
            if parse(target) <= self.clock.now():
                return _Tool({"error": "slot_in_past"}, True)
            details_slot, old_slot = target, booking["slot"]
        else:
            details_slot, old_slot = booking["slot"], None
        workflow = self._workflow(run, action)
        workflow.kind = action
        details = {
            "provider_id": booking["provider_id"], "provider_name": booking["provider"], "service": booking["service"],
            "slot": details_slot, "price": booking["price"], "search_price": None, "hold_id": None,
            "booking_id": booking["id"], "old_slot": old_slot,
        }
        return self._propose(workflow, action, details, run)

    def _propose(self, workflow, action, details, run: _Run) -> _Tool:
        proposal = self.gate.create_proposal(run.user_id, workflow.id, action, details, run.tracer)
        if workflow.state != st.AWAITING_APPROVAL:
            self.workflows.transition(workflow, st.AWAITING_APPROVAL, f"{action} proposed", run.tracer)
        return _Tool({"status": "shown_to_user"}, end=("awaiting_approval", render_prompt(proposal)), proposal=proposal)

    def _check_constraints(self, workflow, hold, run: _Run) -> dict | None:
        budget = workflow.request.get("budget")
        if budget is not None and hold["quoted_price"] > budget:
            return {"error": "over_budget", "quoted_price": hold["quoted_price"], "budget": budget,
                    "detail": "Tell the user, or find something else. They can raise the budget."}
        earliest, latest = self._time_window(workflow, run.user_id)
        if not _in_window(hold["slot"], earliest, latest):
            return {"error": "outside_time_window", "slot": hold["slot"], "earliest": earliest, "latest": latest,
                    "detail": "The slot breaks the user's time constraint."}
        date_from, date_to = workflow.request.get("date_from"), workflow.request.get("date_to")
        if date_from and date_to and not date_from <= hold["slot"][:10] <= date_to:
            return {"error": "outside_date_range", "slot": hold["slot"], "date_from": date_from, "date_to": date_to,
                    "detail": "The slot is outside the requested dates."}
        return None

    def _time_window(self, workflow, user_id) -> tuple[str | None, str | None]:
        """A value stated for this errand wins, then a stored explicit or confirmed preference."""
        earliest = workflow.request.get("earliest_time") or self.memory.preference(user_id, "earliest_time")
        latest = workflow.request.get("latest_time") or self.memory.preference(user_id, "latest_time")
        return earliest, latest

    def _find_duplicate(self, run: _Run, provider_id, slot) -> str | None:
        result = self.tools.read("list_bookings", {"user_id": run.user_id}, run.budget, run.tracer)
        if not result.ok:
            return None
        for booking in result.data["bookings"]:
            if booking["provider_id"] == provider_id and booking["slot"] == slot and booking["status"] == "confirmed":
                return booking["id"]
        return None

    # ------------------------------------------------------------------ memory, enforced by code

    def _tool_manage_memory(self, args, run: _Run) -> _Tool:
        action = args["action"]
        if action == "inspect":
            # Goes straight to the user, so unconfirmed inferences never pass through the model.
            run.tracer.event("memory_inspected", shown_directly_to_user=True)
            return _Tool({"shown_to_user": True},
                         end=("memory_shown", render_inspection(self.memory.inspect(run.user_id))))
        quote = str(args.get("source_quote") or "")
        quoted = bool(quote.strip()) and quote.strip().lower() in run.user_text.lower()
        if action == "remember":
            return self._memory_remember(args, run, quoted)
        if not quoted:
            run.tracer.event("guard_blocked", guard="memory_change_without_user_quote", action=action)
            return _Tool({"error": "needs_user_quote",
                          "detail": "source_quote must be copied verbatim from the user's current message."}, True)
        if action == "correct":
            key, value = str(args["key"]), str(args["value"])
            old = next((m.value for m in self.memory.inspect(run.user_id) if m.key == key), None)
            self.memory.correct(run.user_id, key, value, source=f"message:{run.message_id}")
            run.tracer.event("memory_corrected", key=key, old=old, new=value)
            before = f" (was {old})" if old else ""
            return _Tool({"status": "corrected"}, end=("memory_updated", f"Updated: {key} is now {value}{before}."))
        if action == "forget":
            return self._memory_forget(args, run)
        if action == "confirm":
            return self._memory_confirm(args, run)
        raise ValueError(f"unknown action {action}")

    def _memory_remember(self, args, run: _Run, quoted: bool) -> _Tool:
        key, value = str(args["key"]), str(args["value"])
        if not quoted and run.read_untrusted:
            # No user words back this, and third-party text was read in this run: likely poisoning.
            run.tracer.event("guard_blocked", guard="memory_write_after_untrusted", key=key)
            return _Tool({"error": "needs_user_quote", "detail": "Refused: not supported by the user's message."}, True)
        kind = EXPLICIT if quoted else INFERRED
        memory = self.memory.add(run.user_id, key, value, kind, str(args.get("category") or "general"),
                                 source=f"message:{run.message_id}")
        run.tracer.event("memory_written", key=key, kind=memory.kind, downgraded_to_inferred=not quoted)
        return _Tool({"status": "stored", "key": key, "value": value if quoted else "(withheld until confirmed)",
                      "kind": memory.kind})

    def _memory_forget(self, args, run: _Run) -> _Tool:
        matches = self.memory.find(run.user_id, str(args.get("selector") or args.get("key") or ""))
        if not matches:
            return _Tool({"status": "no_match"},
                         end=("memory_updated", "I couldn't find a stored fact matching that, so nothing was removed."))
        if len(matches) > 1:
            names = ", ".join(sorted(m.key for m in matches))
            return _Tool({"status": "ambiguous"}, end=("awaiting_user", f"That could mean several facts: {names}. Which one?"))
        memory = matches[0]
        value = self.memory.forget(memory)
        scrubbed = self._scrub(run, value)
        run.tracer.event("memory_forgotten", key=memory.key, scrubbed_messages=scrubbed)
        return _Tool({"status": "forgotten"}, end=("memory_updated", f"Forgotten: I no longer keep your {memory.key}."))

    def _memory_confirm(self, args, run: _Run) -> _Tool:
        selector = str(args.get("selector") or args.get("key") or "")
        matches = [m for m in self.memory.find(run.user_id, selector) if m.kind == INFERRED]
        if len(matches) != 1:
            return _Tool({"status": "no_match"},
                         end=("memory_updated", "I couldn't tell which inferred fact you mean, so nothing changed."))
        self.memory.confirm(matches[0].id)
        run.tracer.event("memory_confirmed", key=matches[0].key, kind=CONFIRMED)
        return _Tool({"status": "confirmed"},
                     end=("memory_updated", f"Thanks, I'll use {matches[0].key} = {matches[0].value} from now on."))

    def _scrub(self, run: _Run, value: str | None) -> int:
        """Forgetting means gone: remove the value from history, open workflows and traces."""
        if not value or len(value) < 3:
            return 0
        pattern = re.compile(rf"\b{re.escape(value)}\b", re.IGNORECASE)
        changed = 0
        for row in self.db.all("SELECT id, text FROM messages WHERE user_id=?", (run.user_id,)):
            scrubbed, count = pattern.subn("[forgotten]", row["text"])
            if count:
                self.db.execute("UPDATE messages SET text=? WHERE id=?", (scrubbed, row["id"]))
                changed += 1
        workflow = self.workflows.active(run.user_id)
        if workflow is not None:
            for name, current in list(workflow.request.items()):
                if isinstance(current, str) and pattern.search(current):
                    workflow.request.pop(name)
            self.workflows.save(workflow)
        scrub_traces(self.trace_dir, value)
        return changed


def _claims_success(text: str) -> bool:
    """True if any sentence asserts a completed action. "Nothing has been booked" asserts the opposite."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if _CLAIM.search(sentence) and not _NEGATED.search(_FILLER_NEGATIONS.sub("", sentence)):
            return True
    return False


def _minutes(value: str) -> int:
    hours, minutes = str(value).split(":")
    total = int(hours) * 60 + int(minutes)
    if not 0 <= total < 24 * 60:
        raise ValueError(f"bad time {value}")
    return total


def _in_window(slot: str, earliest: str | None, latest: str | None) -> bool:
    minute = _minutes(slot[11:16])
    if earliest and minute < _minutes(earliest):
        return False
    if latest and minute > _minutes(latest):
        return False
    return True
