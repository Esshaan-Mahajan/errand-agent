"""Durable execution of approved side effects: no duplicates, no unverified success.

Intent is written to SQLite before each call, under an idempotency key derived from the approved
proposal. That lets a repeated confirmation, a network retry, or a restarted process each resolve
to the same effect. Any ambiguous result goes to the source of truth before a retry or a claim.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace

from . import state as st
from .approval import Approval, ApprovalError, ApprovalGate, Proposal, display_slot, money
from .mocks import fmt_seconds
from .sanitize import safe_label
from .tools import BudgetExceeded, RunBudget, ToolRunner

SUCCEEDED = "succeeded"
FAILED = "failed"
UNRESOLVED = "unresolved"
NEEDS_REAPPROVAL = "needs_reapproval"
OPEN_STATUSES = ("authorized", "in_flight", UNRESOLVED)


@dataclass
class ExecOutcome:
    status: str
    message: str
    booking_id: str | None = None
    proposal: Proposal | None = None
    verified: bool = False


class Executor:
    def __init__(self, db, gate: ApprovalGate, tools: ToolRunner, workflows: st.WorkflowStore, clock, limits,
                 crash_hook=None):
        self.db = db
        self.gate = gate
        self.tools = tools
        self.workflows = workflows
        self.clock = clock
        self.limits = limits
        self.crash_hook = crash_hook or (lambda point: None)

    # ------------------------------------------------------------------ entry points

    def execute(self, approval: Approval, proposal: Proposal, tracer) -> ExecOutcome:
        with self.db.transaction():
            self.db.execute(
                "INSERT OR IGNORE INTO executions (idempotency_key, approval_id, proposal_id, user_id, workflow_id, "
                "action, status, attempts, hold_id, result, updated_at) VALUES (?,?,?,?,?,?,?,0,?,NULL,?)",
                (approval.idempotency_key, approval.id, proposal.id, proposal.user_id, proposal.workflow_id,
                 proposal.action, "authorized", proposal.details.get("hold_id"), fmt_seconds(self.clock.now())),
            )
        record = self._record(approval.idempotency_key)
        if record["status"] in (SUCCEEDED, FAILED, NEEDS_REAPPROVAL):
            tracer.event("idempotent_replay", idempotency_key=approval.idempotency_key, status=record["status"])
            return self._replay(record)
        workflow = self.workflows.get(proposal.workflow_id)
        # The execution has its own bound: the run's model budget must not strand a side effect.
        budget = RunBudget(replace(self.limits, max_tool_calls=60), self.clock)
        try:
            if proposal.action == "book":
                return self._run_book(record, approval, proposal, workflow, budget, tracer)
            return self._run_change(record, approval, proposal, workflow, budget, tracer)
        except BudgetExceeded as exc:
            tracer.event("limit_reached", code=exc.code, detail=exc.detail, scope="execution")
            return self._unresolved(record, proposal, workflow, tracer)

    def recover(self, user_id: str, tracer) -> list[ExecOutcome]:
        """Finish anything a crash or an unresolved timeout left open. Safe to run repeatedly."""
        rows = self.db.all(
            "SELECT a.id AS approval_id FROM approvals a "
            "LEFT JOIN executions e ON e.idempotency_key = a.idempotency_key "
            "WHERE a.user_id=? AND a.status='valid' AND (e.status IS NULL OR e.status IN (?,?,?))",
            (user_id, *OPEN_STATUSES),
        )
        outcomes = []
        for row in rows:
            approval = self.gate.get_approval(row["approval_id"])
            proposal = self.gate.get(approval.proposal_id)
            tracer.event("recovery_started", approval_id=approval.id, idempotency_key=approval.idempotency_key,
                         action=proposal.action)
            outcomes.append(self.execute(approval, proposal, tracer))
        return outcomes

    def latest_for_user(self, user_id: str):
        return self.db.one("SELECT * FROM executions WHERE user_id=? ORDER BY updated_at DESC, rowid DESC LIMIT 1",
                           (user_id,))

    def replay_message(self, record) -> str:
        return self._replay(record).message

    # ------------------------------------------------------------------ booking

    def _run_book(self, record, approval, proposal, workflow, budget, tracer) -> ExecOutcome:
        key = approval.idempotency_key
        details = dict(proposal.details)
        details["hold_id"] = record["hold_id"] or details.get("hold_id")
        attempts = record["attempts"]

        # A previous attempt may have landed. Never send another before asking the source of truth.
        if attempts > 0:
            self._move(workflow, st.VERIFYING, "resume: checking previous attempt", tracer)
            verdict, data = self._poll_booking(key, budget, tracer)
            if verdict == "booked":
                return self._booked(record, approval, proposal, workflow, data, tracer)
            if verdict == "unknown":
                return self._unresolved(record, proposal, workflow, tracer)

        while attempts < self.limits.max_side_effect_attempts:
            if budget.out_of_time():
                return self._unresolved(record, proposal, workflow, tracer)
            self._move(workflow, st.EXECUTING, f"booking attempt {attempts + 1}", tracer)
            try:
                authorization = self.gate.authorize(approval, proposal, details, tracer)
            except ApprovalError as exc:
                return self._approval_lost(record, approval, proposal, workflow, exc.code, tracer)
            attempts = self._mark_in_flight(key, details["hold_id"])
            self.crash_hook("before_book")
            result = self.tools.side_effect("book", {"hold_id": details["hold_id"], "idempotency_key": key},
                                            authorization, tracer)
            self.crash_hook("after_book")
            self._move(workflow, st.VERIFYING, "checking book outcome", tracer)
            status = (result.data or {}).get("status") if result.ok else None

            if status == "slot_taken":
                return self._closed(record, approval, workflow, st.COMPARING, FAILED, tracer,
                                    f"{safe_label(details.get('provider_name'))} lost the "
                                    f"{display_slot(details['slot'])} slot before the booking went through. "
                                    "Nothing was booked. Would you like me to look at other slots?")
            if status in ("hold_expired", "hold_not_found"):
                # An expired hold can still hide an earlier success, so check before re-holding.
                verdict, data = self._poll_booking(key, budget, tracer)
                if verdict == "booked":
                    return self._booked(record, approval, proposal, workflow, data, tracer)
                if verdict == "unknown":
                    return self._unresolved(record, proposal, workflow, tracer)
                renewed = self._rehold(record, approval, proposal, workflow, details, budget, tracer)
                if isinstance(renewed, ExecOutcome):
                    return renewed
                details = renewed
                continue

            # A success response is checked too. Timeouts and 503s are ambiguous until checked.
            verdict, data = self._poll_booking(key, budget, tracer)
            if verdict == "booked":
                return self._booked(record, approval, proposal, workflow, data, tracer)
            if verdict == "unknown":
                return self._unresolved(record, proposal, workflow, tracer)
            tracer.event("book_attempt_not_committed", attempt=attempts, tool_error=result.error)
            self.clock.sleep(self.limits.backoff_base_s * (2 ** (attempts - 1)))

        return self._closed(record, approval, workflow, st.FAILED, FAILED, tracer,
                            f"I tried {attempts} times and could not complete the booking. I checked with the "
                            "provider's system and nothing was booked. Would you like to try again or pick "
                            "another option?")

    def _rehold(self, record, approval, proposal, workflow, details, budget, tracer):
        """Replace a lapsed hold. Returns new details, or an ExecOutcome if the approval no longer applies."""
        result = self.tools.read("hold_slot", {"provider_id": details["provider_id"], "slot": details["slot"]},
                                 budget, tracer)
        data = result.data if result.ok else {}
        if not result.ok or "error" in data:
            self.gate.invalidate_approval(approval.id, "slot_no_longer_available", tracer)
            return self._closed(record, approval, workflow, st.COMPARING, NEEDS_REAPPROVAL, tracer,
                                f"The {display_slot(details['slot'])} slot is no longer available, so I have not "
                                "booked anything. Would you like me to look at other slots?", consume=False)
        updated = dict(details, hold_id=data["hold_id"], price=data["quoted_price"],
                       search_price=data.get("search_price"), hold_expires_at=data["expires_at"])
        workflow.holds[data["hold_id"]] = data
        self.workflows.save(workflow)
        try:
            self.gate.authorize(approval, proposal, updated, tracer)
        except ApprovalError as exc:
            if exc.code != "material_change":
                return self._approval_lost(record, approval, proposal, workflow, exc.code, tracer)
            self.gate.invalidate_approval(approval.id, "price_changed_on_rehold", tracer)
            fresh = self.gate.create_proposal(proposal.user_id, proposal.workflow_id, "book", updated, tracer)
            self._move(workflow, st.AWAITING_APPROVAL, "material change needs fresh approval", tracer)
            self._set_execution(record["idempotency_key"], NEEDS_REAPPROVAL, {"replaced_by": fresh.id})
            return ExecOutcome(NEEDS_REAPPROVAL,
                               f"The hold lapsed and the new quote is {money(updated['price'])} instead of "
                               f"{money(details['price'])}, so your earlier confirmation no longer applies. "
                               "Nothing has been booked.",
                               proposal=fresh)
        self.db.execute("UPDATE executions SET hold_id=? WHERE idempotency_key=?",
                        (data["hold_id"], record["idempotency_key"]))
        tracer.event("hold_renewed", hold_id=data["hold_id"], price_unchanged=True)
        return updated

    def _poll_booking(self, key, budget, tracer) -> tuple[str, dict | None]:
        """Ask the authoritative tool. Returns booked, not_booked, or unknown if it can't be reached."""
        for attempt in range(self.limits.status_poll_attempts):
            if budget.out_of_time():
                break
            result = self.tools.read("get_booking_status", {"idempotency_key": key}, budget, tracer, retries=0)
            if result.ok:
                status = result.data.get("status")
                if status == "booked":
                    return "booked", result.data
                if status == "not_booked":
                    return "not_booked", None
            self.clock.sleep(self.limits.backoff_base_s * (2 ** attempt))
        tracer.event("verification_unavailable", idempotency_key=key)
        return "unknown", None

    def _booked(self, record, approval, proposal, workflow, data, tracer) -> ExecOutcome:
        details = proposal.details
        drifted = [name for name in ("provider_id", "slot", "price") if data.get(name) != details.get(name)]
        if drifted:
            tracer.event("verification_mismatch", fields=drifted, status=data)
            return self._unresolved(record, proposal, workflow, tracer)
        self._move(workflow, st.COMPLETED, "source of truth confirms booking", tracer)
        message = (f"Booked and verified: {safe_label(details.get('provider_name'))}, "
                   f"{display_slot(details['slot'])}, price {money(details['price'])}. "
                   f"Booking ID {data['booking_id']}.")
        self._finish(record, approval, SUCCEEDED, {"message": message, "booking_id": data["booking_id"]})
        tracer.event("side_effect_verified", action="book", booking_id=data["booking_id"])
        return ExecOutcome(SUCCEEDED, message, booking_id=data["booking_id"], verified=True)

    # ------------------------------------------------------------------ cancel and reschedule

    def _run_change(self, record, approval, proposal, workflow, budget, tracer) -> ExecOutcome:
        action = proposal.action
        key = approval.idempotency_key
        details = proposal.details
        attempts = record["attempts"]

        if attempts > 0:
            self._move(workflow, st.VERIFYING, "resume: checking previous attempt", tracer)
            verdict, _ = self._verify_change(proposal, budget, tracer)
            if verdict == "done":
                return self._changed(record, approval, proposal, workflow, tracer)
            if verdict == "unknown":
                return self._unresolved(record, proposal, workflow, tracer)

        last_error = None
        while attempts < self.limits.max_side_effect_attempts:
            if budget.out_of_time():
                return self._unresolved(record, proposal, workflow, tracer)
            self._move(workflow, st.EXECUTING, f"{action} attempt {attempts + 1}", tracer)
            verdict, booking = self._verify_change(proposal, budget, tracer)
            if verdict == "done":
                return self._changed(record, approval, proposal, workflow, tracer)
            if verdict == "unknown":
                return self._unresolved(record, proposal, workflow, tracer)
            live = dict(details, provider_id=booking["provider_id"], price=booking["price"])
            if action == "cancel":
                live["slot"] = booking["slot"]
            else:
                live["old_slot"] = booking["slot"]
            problem = "booking_not_active" if booking["status"] != "confirmed" else None
            try:
                if problem:
                    raise ApprovalError(problem)
                authorization = self.gate.authorize(approval, proposal, live, tracer)
            except ApprovalError as exc:
                return self._approval_lost(record, approval, proposal, workflow, exc.code, tracer)
            attempts = self._mark_in_flight(key, None)
            self.crash_hook(f"before_{action}")
            args = {"booking_id": details["booking_id"], "idempotency_key": key}
            if action == "reschedule":
                args["slot"] = details["slot"]
            result = self.tools.side_effect(action, args, authorization, tracer)
            self.crash_hook(f"after_{action}")
            self._move(workflow, st.VERIFYING, f"checking {action} outcome", tracer)
            status = (result.data or {}).get("status") if result.ok else None

            if status == "slot_conflict":
                return self._closed(record, approval, workflow, st.COMPARING, FAILED, tracer,
                                    f"{display_slot(details['slot'])} conflicts with the provider's schedule, so "
                                    f"your booking stays at {display_slot(details['old_slot'])}. "
                                    "Which other time would suit you?")
            if result.ok and "error" in result.data:
                return self._closed(record, approval, workflow, st.FAILED, FAILED, tracer,
                                    f"I couldn't {action} booking {details['booking_id']}: "
                                    f"{result.data['error']}. Nothing was changed.")
            verdict, _ = self._verify_change(proposal, budget, tracer)
            if verdict == "done":
                return self._changed(record, approval, proposal, workflow, tracer)
            if verdict == "unknown":
                return self._unresolved(record, proposal, workflow, tracer)
            last_error = result.error
            tracer.event("change_attempt_not_applied", action=action, attempt=attempts, tool_error=last_error)
            self.clock.sleep(self.limits.backoff_base_s * (2 ** (attempts - 1)))

        return self._closed(record, approval, workflow, st.FAILED, FAILED, tracer,
                            f"I tried {attempts} times to {action} booking {details['booking_id']} and the provider's "
                            f"system kept failing ({last_error}). I checked: the booking is unchanged. "
                            "Please try again later.")

    def _verify_change(self, proposal, budget, tracer) -> tuple[str, dict | None]:
        """Read list_bookings. Returns done, not_done, or unknown, with the current booking."""
        details = proposal.details
        for attempt in range(self.limits.status_poll_attempts):
            if budget.out_of_time():
                break
            result = self.tools.read("list_bookings", {"user_id": proposal.user_id}, budget, tracer, retries=0)
            if result.ok:
                booking = next((b for b in result.data["bookings"] if b["id"] == details["booking_id"]), None)
                if booking is None:
                    return "unknown", None
                if proposal.action == "cancel" and booking["status"] == "cancelled":
                    return "done", booking
                if proposal.action == "reschedule" and booking["slot"] == details["slot"] \
                        and booking["status"] == "confirmed":
                    return "done", booking
                return "not_done", booking
            self.clock.sleep(self.limits.backoff_base_s * (2 ** attempt))
        tracer.event("verification_unavailable", booking_id=details.get("booking_id"))
        return "unknown", None

    def _changed(self, record, approval, proposal, workflow, tracer) -> ExecOutcome:
        details = proposal.details
        self._move(workflow, st.COMPLETED, f"source of truth confirms {proposal.action}", tracer)
        if proposal.action == "cancel":
            message = (f"Cancelled and verified: booking {details['booking_id']} with "
                       f"{safe_label(details.get('provider_name'))} on {display_slot(details['slot'])}.")
        else:
            message = (f"Rescheduled and verified: booking {details['booking_id']} with "
                       f"{safe_label(details.get('provider_name'))} is now on {display_slot(details['slot'])}.")
        self._finish(record, approval, SUCCEEDED, {"message": message, "booking_id": details["booking_id"]})
        tracer.event("side_effect_verified", action=proposal.action, booking_id=details["booking_id"])
        return ExecOutcome(SUCCEEDED, message, booking_id=details["booking_id"], verified=True)

    # ------------------------------------------------------------------ shared outcomes

    def _approval_lost(self, record, approval, proposal, workflow, code, tracer) -> ExecOutcome:
        self.gate.invalidate_approval(approval.id, code, tracer)
        reasons = {
            "approval_expired": "your confirmation is too old to act on",
            "material_change": "the details changed after you confirmed",
            "booking_not_active": "that booking is no longer active",
            "approval_not_valid": "that confirmation is no longer valid",
        }
        return self._closed(record, approval, workflow, st.COMPARING, NEEDS_REAPPROVAL, tracer,
                            f"I stopped because {reasons.get(code, code)}. Nothing was changed. "
                            "Would you like me to check the options again?", consume=False)

    def _unresolved(self, record, proposal, workflow, tracer) -> ExecOutcome:
        self._move(workflow, st.UNRESOLVED, "could not verify outcome", tracer)
        message = ("I can't yet confirm whether this went through: the provider's system isn't answering, so I "
                   f"won't claim success or try again blindly. Please don't re-request this {proposal.action}. "
                   "I'll check again the next time we talk.")
        self._set_execution(record["idempotency_key"], UNRESOLVED, {"message": message})
        return ExecOutcome(UNRESOLVED, message)

    def _closed(self, record, approval, workflow, target, status, tracer, message, consume=True) -> ExecOutcome:
        self._move(workflow, target, f"execution closed: {status}", tracer)
        self._finish(record, approval, status, {"message": message}, consume=consume)
        return ExecOutcome(status, message)

    def _finish(self, record, approval, status, result, consume=True) -> None:
        self._set_execution(record["idempotency_key"], status, result)
        if consume:
            self.db.execute("UPDATE approvals SET status='consumed' WHERE id=?", (approval.id,))

    def _replay(self, record) -> ExecOutcome:
        result = json.loads(record["result"] or "{}")
        return ExecOutcome(record["status"], result.get("message", "That request has already been handled."),
                           booking_id=result.get("booking_id"), verified=record["status"] == SUCCEEDED)

    def _move(self, workflow, target, reason, tracer) -> None:
        if workflow.state == target:
            return
        # Resuming from a crash can land anywhere in the execution path; step through VERIFYING
        # so that every recorded path is one the transition table permits.
        if target not in st.TRANSITIONS[workflow.state] and st.VERIFYING in st.TRANSITIONS[workflow.state]:
            self.workflows.transition(workflow, st.VERIFYING, "intermediate", tracer)
        self.workflows.transition(workflow, target, reason, tracer)

    def _record(self, key):
        return self.db.one("SELECT * FROM executions WHERE idempotency_key=?", (key,))

    def _mark_in_flight(self, key, hold_id) -> int:
        with self.db.transaction():
            self.db.execute("UPDATE executions SET status='in_flight', attempts=attempts+1, hold_id=COALESCE(?, hold_id), "
                            "updated_at=? WHERE idempotency_key=?", (hold_id, fmt_seconds(self.clock.now()), key))
        return self._record(key)["attempts"]

    def _set_execution(self, key, status, result) -> None:
        self.db.execute("UPDATE executions SET status=?, result=?, updated_at=? WHERE idempotency_key=?",
                        (status, json.dumps(result), fmt_seconds(self.clock.now()), key))
