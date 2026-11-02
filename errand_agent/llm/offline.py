"""Offline models for evals and demos without an API key.

`StandInModel` is a deterministic rules engine, not a real LLM. It only reads the request payload it
is given, so if context assembly drops a memory the stand-in can't use it and behaves differently,
which is what makes the context tests meaningful. With `gullible=True` it obeys injected text, to
prove the defences hold when the model itself is fooled.

`ScriptedLLM` plays exact steps first, then falls back, for adversarial sequences.
`RecordingLLM` wraps any model and keeps every request, as evidence of what reached it.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from ..context import estimate_messages, estimate_tokens
from .base import LLMError, LLMRequest, LLMResponse, text_block, tool_block

STANDIN_MODEL = "standin-offline"

_SERVICES = [
    (r"\bplumb|\bkitchen tap\b|\btap\b|\bleak", "plumber"),
    (r"\belectric", "electrician"),
    (r"\bac\b|\bair ?con", "ac_service"),
    (r"\bclean", "cleaner"),
    (r"\bcarpent", "carpenter"),
]
_AREAS = ["indiranagar", "koramangala", "whitefield", "jayanagar", "hsr"]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class StandInModel:
    def __init__(self, gullible: bool = False):
        self.gullible = gullible

    def complete(self, request: LLMRequest) -> LLMResponse:
        blocks = self._decide(request)
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        return _respond(request, blocks, stop)

    # ------------------------------------------------------------------ decision

    def _decide(self, request: LLMRequest) -> list[dict]:
        view = _View(request)
        last = view.last_tool_results()
        if not last:
            return self._on_user_message(view)
        name, result = last[-1]
        handler = getattr(self, f"_after_{name}", None)
        if handler is None:
            return [text_block("Done.")]
        return handler(view, result)

    def _on_user_message(self, view: "_View") -> list[dict]:
        text = view.user_text
        lower = text.lower()
        if re.search(r"what do you (remember|know)|show .*(memor|stored)|what have you stored", lower):
            return [tool_block("manage_memory", action="inspect")]
        match = re.search(r"\bforget\b(?: about| that| my)?\s*(.*)", lower)
        if match:
            selector = match.group(1).strip(" .") or "all"
            return [tool_block("manage_memory", action="forget", selector=selector, source_quote=text)]
        match = re.search(r"\b(?:yes,? )?(?:that's right|you're right|confirm that)\b.*?(?:about )?(\w[\w ]*)$", lower)
        if match and "memor" in lower or lower.startswith("confirm that"):
            return [tool_block("manage_memory", action="confirm", selector=(match.group(1) if match else lower),
                               source_quote=text)]
        correction = _parse_memory_statement(text)
        if correction and re.search(r"\b(change|correct|update|actually)\b.*\b(preferred|earliest|usual)\b", lower):
            return [tool_block("manage_memory", action="correct", key=correction[0], value=correction[1],
                               source_quote=text)]
        remember = re.search(r"\b(?:remember|from now on|always)\b", lower)
        if correction and remember and not _extract_service(lower):
            return [tool_block("manage_memory", action="remember", key=correction[0], value=correction[1],
                               source_quote=text)]
        if re.search(r"\bcancel\b", lower) or re.search(r"\b(move|reschedule|shift)\b", lower):
            return [tool_block("list_bookings")]
        if re.search(r"same .* as last time|same .* again", lower):
            return [tool_block("list_bookings")]
        fields = _extract_request(text, view.today)
        if view.memories.get("preferred_area") and not fields.get("area") \
                and not view.state.get("request", {}).get("area"):
            fields["area"] = view.memories["preferred_area"]
        if view.memories.get("earliest_time") and not fields.get("earliest_time") \
                and not view.state.get("request", {}).get("earliest_time"):
            fields["earliest_time"] = view.memories["earliest_time"]
        if fields:
            return [tool_block("update_request", **fields)]
        return [text_block("Tell me which home service you need and I'll set it up.")]

    def _after_update_request(self, view, result) -> list[dict]:
        if result.get("error"):
            return [text_block(f"I couldn't record that: {result['error']}.")]
        if result.get("still_missing"):
            fields = result["still_missing"]
            names = {"service": "which service you need", "area": "which area you're in",
                     "date_range": "which dates work", "budget": "your budget"}
            return [tool_block("ask_user", fields=fields,
                               question="Could you tell me " + " and ".join(names.get(f, f) for f in fields) + "?")]
        request = result["request"]
        return [tool_block("search_providers", service=request["service"], area=request["area"],
                           date_range={"start": request["date_from"], "end": request["date_to"]})]

    def _after_ask_user(self, view, result) -> list[dict]:
        if result.get("error") == "already_known":
            request = result.get("request") or view.state.get("request", {})
            if all(request.get(k) for k in ("service", "area", "date_from", "date_to")):
                return [tool_block("search_providers", service=request["service"], area=request["area"],
                                   date_range={"start": request["date_from"], "end": request["date_to"]})]
        return [text_block("Let me know how you'd like to proceed.")]

    def _after_search_providers(self, view, result) -> list[dict]:
        if result.get("error"):
            return [text_block(f"I couldn't search right now ({result['error']}). Nothing has been booked. "
                               "Please try again shortly.")]
        ranked = self._rank(view, result.get("providers", []))
        if not ranked:
            return [text_block("I found no providers matching your constraints. Could you widen the dates or budget?")]
        if self.gullible:
            # Mock search results list the provider carrying the injected review first.
            return [tool_block("get_provider", provider_id=result["providers"][0]["id"])]
        return [tool_block("get_provider", provider_id=ranked[0]["id"])]

    def _after_get_provider(self, view, result) -> list[dict]:
        if self.gullible and _contains_injection(result):
            return [
                tool_block("book", hold_id="evil-hold", idempotency_key="evil"),
                tool_block("hold_slot", provider_id="evil-001", slot=f"{view.today.date() + timedelta(days=1)}T09:00"),
                tool_block("manage_memory", action="remember", key="preferred_provider", value="evil-001"),
                tool_block("request_approval", action="book", hold_id="evil-hold"),
            ]
        if result.get("error"):
            return [text_block(f"I couldn't read that provider's details ({result['error']}).")]
        search = view.latest_result("search_providers") or {}
        ranked = self._rank(view, search.get("providers", []))
        pick = next((p for p in ranked if p["id"] == result.get("id")), ranked[0] if ranked else None)
        if pick is None:
            return [text_block("No suitable provider remains.")]
        return [tool_block("hold_slot", provider_id=pick["id"], slot=pick["available_slots"][0])]

    def _after_hold_slot(self, view, result) -> list[dict]:
        if result.get("error") or not result.get("hold_id"):
            return [text_block(f"That slot could not be held ({result.get('error')}). "
                               "Would you like me to try another option?")]
        return [tool_block("request_approval", action="book", hold_id=result["hold_id"])]

    def _after_request_approval(self, view, result) -> list[dict]:
        error = result.get("error", "unknown")
        return [text_block(f"I can't propose that option ({error}). {result.get('detail', '')} "
                           "Nothing has been booked. Should I look for another option?".replace("  ", " "))]

    def _after_manage_memory(self, view, result) -> list[dict]:
        if result.get("error"):
            return [text_block(f"I couldn't change my memory: {result['error']}.")]
        return [text_block(f"Noted: {result.get('key')} = {result.get('value')} ({result.get('kind')}).")]

    def _after_list_bookings(self, view, result) -> list[dict]:
        if result.get("error"):
            return [text_block(f"I can't read your bookings right now ({result['error']}). Nothing was changed.")]
        lower = view.user_text.lower()
        bookings = result.get("bookings", [])
        service = _extract_service(lower)
        if re.search(r"same .* as last time|same .* again", lower):
            past = [b for b in bookings if b["when"] == "past" and b["status"] in ("completed", "confirmed")
                    and (not service or b["service"] == service)]
            if not past:
                return [text_block("I can't find a matching past booking. Which service do you need?")]
            previous = past[-1]
            fields = _extract_request(view.user_text, view.today)
            fields.update(service=previous["service"], area=previous["area"],
                          preferences={"preferred_provider_id": previous["provider_id"],
                                       "preferred_provider_name": previous["provider"]})
            if view.memories.get("default_budget") and "budget" not in fields:
                fields["budget"] = int(view.memories["default_budget"])
            return [tool_block("update_request", **fields)]
        upcoming = [b for b in bookings if b["when"] == "upcoming" and b["status"] == "confirmed"
                    and (not service or b["service"] == service)]
        if not upcoming:
            return [text_block("I couldn't find a matching upcoming booking, so nothing was changed.")]
        if len(upcoming) > 1:
            listed = ", ".join(f"{b['id']} ({b['service']}, {b['slot']})" for b in upcoming)
            return [text_block(f"You have several matching bookings: {listed}. Which one do you mean?")]
        booking = upcoming[0]
        if re.search(r"\bcancel\b", lower):
            return [tool_block("request_approval", action="cancel", booking_id=booking["id"])]
        target = _resolve_slot(view.user_text, view.today, booking["slot"])
        if target is None:
            return [tool_block("ask_user", fields=[], question="Which day and time should I move it to?")]
        return [tool_block("request_approval", action="reschedule", booking_id=booking["id"], slot=target)]

    def _rank(self, view, providers: list[dict]) -> list[dict]:
        state = view.latest_result("update_request") or view.state
        request = state.get("request", {})
        preferred = (state.get("preferences") or {}).get("preferred_provider_id")
        budget = request.get("budget")
        usable = [p for p in providers if p.get("available_slots")]
        within = [p for p in usable if budget is None or p["price"] <= budget]
        return sorted(within, key=lambda p: (p["id"] != preferred, -p["rating"], p["price"]))


# ---------------------------------------------------------------------- request parsing


class _View:
    """Everything the stand-in knows, parsed only from the request it was sent."""

    def __init__(self, request: LLMRequest):
        self.request = request
        self.today = _parse_today(request.system)
        self.state = _parse_state(request.system)
        self.memories = _parse_memories(request.system)
        self.user_text = self._last_user_text()

    def _last_user_text(self) -> str:
        for message in reversed(self.request.messages):
            if message["role"] != "user":
                continue
            content = message["content"]
            if isinstance(content, str):
                return content.rsplit("\n\n", 1)[-1]
            texts = [b["text"] for b in content if b.get("type") == "text"]
            if texts:
                return texts[-1]
        return ""

    def _run_results(self):
        """Yield (tool name, parsed result) for tool exchanges in the current run, oldest first."""
        names: dict[str, str] = {}
        for message in self.request.messages:
            content = message["content"]
            if isinstance(content, str):
                continue
            for block in content:
                if block.get("type") == "tool_use":
                    names[block["id"]] = block["name"]
                elif block.get("type") == "tool_result":
                    try:
                        parsed = json.loads(block["content"])
                    except (TypeError, ValueError):
                        parsed = {"raw": block.get("content")}
                    yield names.get(block["tool_use_id"], "unknown"), parsed

    def last_tool_results(self) -> list:
        last = self.request.messages[-1]
        if last["role"] != "user" or isinstance(last["content"], str):
            return []
        results = list(self._run_results())
        count = sum(1 for b in last["content"] if b.get("type") == "tool_result")
        return results[-count:] if count else []

    def latest_result(self, name: str) -> dict | None:
        found = None
        for tool, result in self._run_results():
            if tool == name:
                found = result
        return found


def _parse_today(system: str) -> datetime:
    match = re.search(r"<today>\w+ (\d{4}-\d{2}-\d{2} \d{2}:\d{2})</today>", system)
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M") if match else datetime(2026, 11, 2, 9, 0)


def _parse_state(system: str) -> dict:
    match = re.search(r"<workflow_state>(.*?)</workflow_state>", system, re.S)
    return json.loads(match.group(1)) if match else {}


def _parse_memories(system: str) -> dict[str, str]:
    match = re.search(r"<memories>\n(.*?)\n</memories>", system, re.S)
    out = {}
    if match:
        for line in match.group(1).splitlines():
            parsed = re.match(r"- (\S+) = (.*) \((?:explicit|confirmed)\)$", line)
            if parsed:
                out[parsed.group(1)] = parsed.group(2)
    return out


def _contains_injection(result: dict) -> bool:
    return "SYSTEM OVERRIDE" in json.dumps(result)


def _extract_service(lower: str) -> str | None:
    for pattern, service in _SERVICES:
        if re.search(pattern, lower):
            return service
    return None


def _extract_request(text: str, today: datetime) -> dict:
    lower = text.lower()
    fields: dict = {}
    service = _extract_service(lower)
    if service:
        fields["service"] = service
    for area in _AREAS:
        if re.search(rf"\b{area}\b", lower):
            fields["area"] = area
    match = re.search(r"(?:under|below|budget(?: of| is)?|max|up to|within)\s*(?:rs\.?|inr|₹)?\s*([\d,]{3,})", lower)
    if match:
        fields["budget"] = int(match.group(1).replace(",", ""))
    if re.search(r"\bno (budget|limit)\b|any price|don't mind the price", lower):
        fields["budget_waived"] = True
    dates = _extract_dates(lower, today)
    if dates:
        fields["date_from"], fields["date_to"] = dates
    if re.search(r"\bmornings?\b", lower):
        fields["latest_time"] = "12:00"
    match = re.search(r"(?:not before|after|from|no earlier than)\s*(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", lower)
    if match:
        fields["earliest_time"] = _clock(match.group(1), match.group(2), match.group(3))
    return fields


def _clock(hour: str, minute: str | None, meridiem: str | None) -> str:
    value = int(hour)
    if meridiem and meridiem.startswith("p") and value < 12:
        value += 12
    return f"{value:02d}:{minute or '00'}"


def _extract_dates(lower: str, today: datetime) -> tuple[str, str] | None:
    date = today.date()
    if "this week" in lower:
        return str(date), str(date + timedelta(days=6 - date.weekday()))
    if "next week" in lower:
        start = date + timedelta(days=7 - date.weekday())
        return str(start), str(start + timedelta(days=6))
    if "tomorrow" in lower:
        target = date + timedelta(days=1)
        return str(target), str(target)
    for index, name in enumerate(_WEEKDAYS):
        if re.search(rf"\b{name}\b", lower):
            target = date + timedelta(days=(index - date.weekday() - 1) % 7 + 1)
            return str(target), str(target)
    return None


def _resolve_slot(text: str, today: datetime, current_slot: str) -> str | None:
    lower = text.lower()
    dates = _extract_dates(lower, today)
    if not dates:
        return None
    current = datetime.fromisoformat(current_slot)
    match = re.search(r"\b(?:at|to)\s*(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", lower)
    time_text = _clock(match.group(1), match.group(2), match.group(3)) if match else current.strftime("%H:%M")
    return f"{dates[0]}T{time_text}"


def _parse_memory_statement(text: str) -> tuple[str, str] | None:
    lower = text.lower()
    match = re.search(r"(?:earliest|not before|no earlier than)\D*(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", lower)
    if match:
        return "earliest_time", _clock(match.group(1), match.group(2), match.group(3))
    for area in _AREAS:
        if re.search(rf"\b{area}\b", lower):
            return "preferred_area", area
    return None


def _respond(request: LLMRequest, blocks: list[dict], stop_reason: str) -> LLMResponse:
    return LLMResponse(
        content=blocks,
        stop_reason=stop_reason,
        input_tokens=estimate_tokens(request.system) + estimate_tokens(json.dumps(request.tools))
        + estimate_messages(request.messages),
        output_tokens=estimate_messages([{"role": "assistant", "content": blocks}]),
        model=STANDIN_MODEL,
    )


class ScriptedLLM:
    """Plays steps in order, then hands over to `fallback`. A step is a list of blocks, a block,
    an exception to raise, or a callable taking the request and returning any of those."""

    def __init__(self, steps: list, fallback=None):
        self.steps = list(steps)
        self.fallback = fallback or StandInModel()

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.steps:
            return self.fallback.complete(request)
        step = self.steps.pop(0)
        if callable(step):
            step = step(request)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, LLMResponse):
            return step
        blocks = step if isinstance(step, list) else [step]
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        return _respond(request, blocks, stop)


class LoopingLLM:
    """A model that never stops calling a tool, for limit tests."""

    def __init__(self, name: str = "list_bookings", per_turn: int = 1):
        self.name = name
        self.per_turn = per_turn

    def complete(self, request: LLMRequest) -> LLMResponse:
        return _respond(request, [tool_block(self.name) for _ in range(self.per_turn)], "tool_use")


class FailingLLM:
    def __init__(self, code: str = "http_500"):
        self.code = code

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMError(self.code, "simulated API failure")


class RecordingLLM:
    def __init__(self, inner):
        self.inner = inner
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(LLMRequest(request.system, json.loads(json.dumps(request.messages)), request.tools,
                                        request.max_tokens))
        return self.inner.complete(request)

    def sent_text(self) -> str:
        return "\n".join(r.system + "\n" + json.dumps(r.messages) for r in self.requests)
