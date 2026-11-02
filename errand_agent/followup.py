"""Proactive follow-up. Detects provider cancellations by diffing list_bookings, with no user prompt.

The monitor decides whether to notify and how, looks for a replacement inside known preferences, and
only ever proposes one. Rebooking still requires a fresh approval through the normal gate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import timedelta

from . import state as st
from .agent import Agent, _in_window
from .approval import display_slot, render_prompt
from .clock import parse
from .mocks import fmt_seconds
from .sanitize import safe_label
from .tools import RunBudget

EVENT = "provider_cancelled"


@dataclass
class Decision:
    notify: bool
    channel: str | None
    seek_replacement: bool
    reason: str


class _Quiet:
    """Polling a healthy system should not write a trace file on every tick."""
    tool_calls = 0

    def event(self, *args, **kwargs) -> None:
        pass


def decide(booking: dict, bookings: list[dict], now, has_open_work: bool) -> Decision:
    """Whether and how to tell the user. Pure, so each branch is directly testable."""
    slot = parse(booking["slot"])
    if slot <= now:
        return Decision(False, None, False, "slot already in the past")
    covered = next((b for b in bookings if b["id"] != booking["id"] and b["service"] == booking["service"]
                    and b["status"] == "confirmed" and abs(parse(b["slot"]) - slot) <= timedelta(days=3)), None)
    channel = "urgent" if slot - now <= timedelta(hours=24) else "chat"
    if covered:
        return Decision(True, channel, False, f"another booking {covered['id']} already covers this service")
    if has_open_work:
        return Decision(True, channel, False, "user has an errand in progress; not replacing it unasked")
    return Decision(True, channel, True, "upcoming booking lost; replacement needed")


class FollowUpMonitor:
    def __init__(self, agent: Agent):
        self.agent = agent

    def tick(self, user_id: str) -> list[dict]:
        agent = self.agent
        with agent._lock_for(user_id):
            budget = RunBudget(agent.limits, agent.clock)
            result = agent.tools.read("list_bookings", {"user_id": user_id}, budget, _Quiet())
            if not result.ok:
                return []
            bookings = result.data["bookings"]
            events = []
            for booking in bookings:
                seen = agent.db.one("SELECT 1 FROM followups WHERE booking_id=? AND event=?", (booking["id"], EVENT))
                if booking["status"] == EVENT and not seen:
                    events.append(self._handle(user_id, booking, bookings))
                agent.db.execute("INSERT OR REPLACE INTO booking_snapshots VALUES (?,?,?,?)",
                                 (booking["id"], user_id, booking["status"], json.dumps(booking)))
            return events

    def _handle(self, user_id: str, booking: dict, bookings: list[dict]) -> dict:
        agent = self.agent
        tracer = agent.new_tracer(user_id, kind="proactive")
        tracer.event("provider_cancellation_detected", booking_id=booking["id"], provider=booking["provider"],
                     slot=booking["slot"], detected_by="list_bookings diff")
        workflow = agent.workflows.active(user_id)
        has_open_work = workflow is not None or agent.gate.pending(user_id) is not None
        decision = decide(booking, bookings, agent.clock.now(), has_open_work)
        tracer.event("followup_decision", notify=decision.notify, channel=decision.channel,
                     seek_replacement=decision.seek_replacement, reason=decision.reason)
        lost = (f"{safe_label(booking['provider'])} cancelled your {booking['service'].replace('_', ' ')} visit on "
                f"{display_slot(booking['slot'])}.")
        proposal = None
        if not decision.notify:
            text = None
        elif decision.seek_replacement:
            proposal, why = self._find_replacement(user_id, booking, tracer)
            if proposal is None:
                text = f"{lost} I couldn't find a replacement within your preferences ({why}). Nothing has been rebooked. Should I widen the search?"
            else:
                text = render_prompt(proposal, intro=f"{lost} I found a replacement within your preferences:")
        else:
            text = f"{lost} {decision.reason.capitalize()}, so I haven't looked for a replacement. Tell me if you'd like me to."

        now = fmt_seconds(agent.clock.now())
        with agent.db.transaction():
            # The notice and its dedupe record commit together: a restart can't repeat or lose it.
            message_id = None
            if text is not None:
                message_id = agent.add_message(user_id, "assistant", text, tracer.run_id)
                agent.db.execute("INSERT INTO outbox (user_id, message_id, channel, created_at) VALUES (?,?,?,?)",
                                 (user_id, message_id, decision.channel, now))
            agent.db.execute("INSERT INTO followups VALUES (?,?,?,?,?,?)",
                             (booking["id"], EVENT, user_id, "notified" if text else "suppressed", decision.reason, now))
        if proposal is not None and message_id is not None:
            agent.gate.mark_shown(proposal.id, message_id)
        tracer.finish("proactive_notified" if text else "proactive_suppressed", text or "", channel=decision.channel,
                      proposal=proposal.code if proposal else None)
        return {"booking_id": booking["id"], "decision": decision, "text": text,
                "proposal": proposal, "channel": decision.channel, "run_id": tracer.run_id}

    def _find_replacement(self, user_id: str, booking: dict, tracer):
        agent = self.agent
        budget = RunBudget(replace(agent.limits, max_tool_calls=12), agent.clock)
        slot = parse(booking["slot"])
        start = max(slot.date(), agent.clock.now().date())
        date_range = {"start": str(start), "end": str(start + timedelta(days=3))}
        result = agent.tools.read("search_providers", {"service": booking["service"], "area": booking["area"],
                                                       "date_range": date_range}, budget, tracer)
        if not result.ok:
            return None, "the provider search is unavailable"

        memory = agent.memory
        earliest = memory.preference(user_id, "earliest_time")
        latest = memory.preference(user_id, "latest_time")
        budget_pref = memory.preference(user_id, "default_budget")
        ceiling = int(budget_pref) if budget_pref else int(booking["price"] * 1.2)
        preferred = memory.preference(user_id, f"preferred_provider:{booking['service']}")
        technician = memory.preference(user_id, f"preferred_technician:{booking['service']}")
        avoid = memory.preference(user_id, "avoid_provider")
        tracer.event("replacement_constraints", earliest=earliest, latest=latest, price_ceiling=ceiling,
                     preferred_provider=preferred, preferred_technician=technician, avoid_provider=avoid,
                     excluded_provider=booking["provider_id"])

        options = []
        for provider in result.data["providers"]:
            if provider["id"] in (booking["provider_id"], avoid) or provider["price"] > ceiling:
                continue
            slots = [s for s in provider["available_slots"] if _in_window(s, earliest, latest)]
            if slots:
                options.append((provider, slots))
        if not options:
            return None, f"no slots at or under {ceiling} in your time window"
        options.sort(key=lambda item: (item[0]["id"] != preferred,
                                       item[0]["attributes"].get("technician") != technician,
                                       -item[0]["rating"], item[0]["price"]))

        for provider, slots in options[:3]:
            held = agent.tools.read("hold_slot", {"provider_id": provider["id"], "slot": slots[0]}, budget, tracer)
            if not held.ok or "error" in held.data:
                continue
            if held.data["quoted_price"] > ceiling:
                tracer.event("replacement_rejected", provider_id=provider["id"], reason="quote above ceiling",
                             quoted_price=held.data["quoted_price"], ceiling=ceiling)
                continue
            workflow = agent.workflows.create(user_id, "book", tracer, origin={"replaces": booking["id"]})
            workflow.request.update(service=booking["service"], area=booking["area"], date_from=date_range["start"],
                                    date_to=date_range["end"], budget=ceiling)
            workflow.candidates[provider["id"]] = {"name": safe_label(provider["name"]), "service": booking["service"],
                                                   "price": provider["price"], "slots": provider["available_slots"]}
            workflow.holds[held.data["hold_id"]] = held.data
            agent.workflows.transition(workflow, st.COMPARING, "replacement search", tracer)
            details = {
                "provider_id": provider["id"], "provider_name": safe_label(provider["name"]),
                "service": booking["service"], "slot": slots[0], "price": held.data["quoted_price"],
                "search_price": held.data["search_price"], "hold_id": held.data["hold_id"],
                "hold_expires_at": held.data["expires_at"], "booking_id": None, "old_slot": None,
            }
            proposal = agent.gate.create_proposal(user_id, workflow.id, "book", details, tracer)
            agent.workflows.transition(workflow, st.AWAITING_APPROVAL, "replacement proposed", tracer)
            return proposal, ""
        return None, "no suitable slot could be held"
