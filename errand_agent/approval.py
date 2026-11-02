"""Approval enforcement. All decisions here are made by code from the user's raw text.

The model can *request* a proposal but cannot approve one. Details come from tool results rather
than model arguments, and the prompt the user sees is rendered here rather than by the model.
A proposal is bound to a fingerprint of its material details, so changing the provider, slot or
price yields a different fingerprint, which no existing approval matches.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

from .clock import parse
from .mocks import fmt_seconds
from .sanitize import safe_label

MATERIAL_FIELDS = ("action", "user_id", "provider_id", "slot", "price", "booking_id", "old_slot")


class Reply(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    OTHER = "other"


_APPROVE = {
    "yes", "y", "yep", "yeah", "yes please", "ok", "okay", "ok go ahead", "okay go ahead", "sure", "confirm",
    "confirmed", "i confirm", "yes confirm", "yes i confirm", "go ahead", "yes go ahead", "please go ahead",
    "do it", "yes do it", "please do", "proceed", "please proceed", "yes proceed", "approve", "approved",
    "i approve", "book it", "yes book it", "please book it", "reschedule it", "yes reschedule it",
    "move it", "yes move it", "cancel it", "yes cancel it", "please cancel it",
}
_REJECT = {
    "no", "n", "nope", "nah", "no thanks", "no thank you", "not now", "dont", "do not", "stop", "never mind",
    "nevermind", "reject", "decline", "skip it", "hold off",
}


def parse_reply(text: str, proposal_code: str) -> Reply:
    """Strict grammar over the raw message. Anything with extra content is not an approval."""
    normalised = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    normalised = re.sub(r"\s+", " ", normalised)
    # "confirm P-AB12" names the proposal; naming a different one is not consent to this one.
    match = re.fullmatch(r"(?:yes |i )?(?:confirm|approve) (p-[a-z0-9]{4})", normalised)
    if match:
        return Reply.APPROVE if match.group(1) == proposal_code.lower() else Reply.OTHER
    if normalised in _APPROVE:
        return Reply.APPROVE
    if normalised in _REJECT:
        return Reply.REJECT
    return Reply.OTHER


def is_bare_affirmative(text: str) -> bool:
    return parse_reply(text, "p-0000") is Reply.APPROVE


def fingerprint(action: str, user_id: str, details: dict) -> str:
    material = {"action": action, "user_id": user_id}
    for name in MATERIAL_FIELDS[2:]:
        material[name] = details.get(name)
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


@dataclass
class Proposal:
    id: str
    code: str
    user_id: str
    workflow_id: str
    action: str
    details: dict
    fingerprint: str
    status: str
    shown_msg_id: int | None
    created_at: str


@dataclass
class Approval:
    id: str
    proposal_id: str
    user_id: str
    fingerprint: str
    idempotency_key: str
    user_message_id: int
    status: str
    created_at: str


@dataclass(frozen=True)
class Authorization:
    """Permission for one side-effect call under one approval and idempotency key."""
    approval_id: str
    tool: str
    idempotency_key: str

    def permits(self, tool: str, args: dict) -> bool:
        return tool == self.tool and args.get("idempotency_key") == self.idempotency_key


class ApprovalError(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = code


TOOL_FOR_ACTION = {"book": "book", "cancel": "cancel", "reschedule": "reschedule"}


class ApprovalGate:
    def __init__(self, db, clock, limits):
        self.db = db
        self.clock = clock
        self.limits = limits

    def create_proposal(self, user_id: str, workflow_id: str, action: str, details: dict, tracer) -> Proposal:
        """Record a proposal, superseding any pending one: only the latest can be approved."""
        now = fmt_seconds(self.clock.now())
        proposal = Proposal(
            id=f"prop-{uuid.uuid4().hex[:10]}",
            code=f"P-{uuid.uuid4().hex[:4].upper()}",
            user_id=user_id,
            workflow_id=workflow_id,
            action=action,
            details=details,
            fingerprint=fingerprint(action, user_id, details),
            status="pending",
            shown_msg_id=None,
            created_at=now,
        )
        with self.db.transaction():
            self.db.execute("UPDATE proposals SET status='superseded' WHERE user_id=? AND status='pending'", (user_id,))
            self.db.execute(
                "INSERT INTO proposals VALUES (?,?,?,?,?,?,?,?,?,?)",
                (proposal.id, proposal.code, user_id, workflow_id, action, json.dumps(details),
                 proposal.fingerprint, "pending", None, now),
            )
        tracer.event("proposal_created", proposal_id=proposal.id, code=proposal.code, action=action,
                     details=details, fingerprint=proposal.fingerprint[:16])
        return proposal

    def mark_shown(self, proposal_id: str, message_id: int) -> None:
        self.db.execute("UPDATE proposals SET shown_msg_id=? WHERE id=?", (message_id, proposal_id))

    def pending(self, user_id: str) -> Proposal | None:
        row = self.db.one("SELECT * FROM proposals WHERE user_id=? AND status='pending' "
                          "ORDER BY created_at DESC, rowid DESC LIMIT 1", (user_id,))
        return _proposal(row) if row else None

    def get(self, proposal_id: str) -> Proposal | None:
        row = self.db.one("SELECT * FROM proposals WHERE id=?", (proposal_id,))
        return _proposal(row) if row else None

    def invalidate_pending(self, user_id: str, reason: str, tracer) -> bool:
        cursor = self.db.execute("UPDATE proposals SET status='invalidated' WHERE user_id=? AND status='pending'",
                                 (user_id,))
        if cursor.rowcount:
            tracer.event("approval_invalidated", scope="pending_proposal", reason=reason)
        return bool(cursor.rowcount)

    def reject(self, proposal: Proposal, tracer) -> None:
        self.db.execute("UPDATE proposals SET status='rejected' WHERE id=? AND status='pending'", (proposal.id,))
        tracer.event("proposal_rejected", proposal_id=proposal.id)

    def approve(self, proposal: Proposal, user_message_id: int, tracer) -> Approval | None:
        """Atomically turn a pending, already-shown proposal into an approval.

        Returns None if another confirmation won the race or the proposal wasn't shown before
        this message: a proposal can only be approved by a message sent after the user saw it.
        """
        if proposal.shown_msg_id is None or user_message_id <= proposal.shown_msg_id:
            tracer.event("guard_blocked", guard="approval_before_shown", proposal_id=proposal.id)
            return None
        now = fmt_seconds(self.clock.now())
        key = f"{proposal.action}-{proposal.workflow_id[-8:]}-{proposal.fingerprint[:16]}"
        approval = Approval(f"appr-{uuid.uuid4().hex[:10]}", proposal.id, proposal.user_id, proposal.fingerprint,
                            key, user_message_id, "valid", now)
        with self.db.transaction():
            cursor = self.db.execute("UPDATE proposals SET status='approved' WHERE id=? AND status='pending'",
                                     (proposal.id,))
            if cursor.rowcount != 1:
                return None
            self.db.execute(
                "INSERT INTO approvals VALUES (?,?,?,?,?,?,?,?)",
                (approval.id, approval.proposal_id, approval.user_id, approval.fingerprint, key,
                 user_message_id, "valid", now),
            )
        tracer.event("approval_recorded", approval_id=approval.id, proposal_id=proposal.id,
                     user_message_id=user_message_id, fingerprint=proposal.fingerprint[:16], idempotency_key=key)
        return approval

    def get_approval(self, approval_id: str) -> Approval | None:
        row = self.db.one("SELECT * FROM approvals WHERE id=?", (approval_id,))
        return Approval(**dict(row)) if row else None

    def invalidate_approval(self, approval_id: str, reason: str, tracer) -> None:
        self.db.execute("UPDATE approvals SET status='invalidated' WHERE id=?", (approval_id,))
        tracer.event("approval_invalidated", scope="approval", approval_id=approval_id, reason=reason)

    def authorize(self, approval: Approval, proposal: Proposal, live_details: dict, tracer) -> Authorization:
        """The last check before any side effect. Raises ApprovalError if anything drifted."""
        current = self.get_approval(approval.id)
        if current is None or current.status != "valid":
            raise ApprovalError("approval_not_valid")
        if self.approval_expired(approval):
            raise ApprovalError("approval_expired")
        live = fingerprint(proposal.action, proposal.user_id, live_details)
        if live != approval.fingerprint:
            tracer.event("guard_blocked", guard="material_change", approved=approval.fingerprint[:16], live=live[:16])
            raise ApprovalError("material_change")
        return Authorization(approval.id, TOOL_FOR_ACTION[proposal.action], approval.idempotency_key)

    def approval_expired(self, approval: Approval) -> bool:
        return self.clock.now() > parse(approval.created_at) + timedelta(seconds=self.limits.approval_ttl_s)


def _proposal(row) -> Proposal:
    data = dict(row)
    data["details"] = json.loads(data["details"])
    return Proposal(**data)


def display_slot(slot: str) -> str:
    return parse(slot).strftime("%a %d %b, %H:%M")


def money(amount) -> str:
    return f"{int(amount):,}"


def render_prompt(proposal: Proposal, intro: str | None = None) -> str:
    """The exact text the user approves. Rendered from stored data, never from model output."""
    d = proposal.details
    name = safe_label(d.get("provider_name"))
    if proposal.action == "book":
        lines = [f"Please confirm this booking ({proposal.code}):",
                 f"  Provider: {name}",
                 f"  Service:  {d.get('service', '').replace('_', ' ')}",
                 f"  Slot:     {display_slot(d['slot'])}",
                 f"  Price:    {money(d['price'])}"]
        if d.get("search_price") is not None and d["search_price"] != d["price"]:
            lines.append(f"  Note:     the quoted price differs from the search price of {money(d['search_price'])}.")
        lines.append(f"  Slot held until {parse(d['hold_expires_at']).strftime('%H:%M')}.")
    elif proposal.action == "cancel":
        lines = [f"Please confirm this cancellation ({proposal.code}):",
                 f"  Booking:  {d['booking_id']}",
                 f"  Provider: {name}",
                 f"  Slot:     {display_slot(d['slot'])}",
                 f"  Price:    {money(d['price'])}"]
    else:
        lines = [f"Please confirm this reschedule ({proposal.code}):",
                 f"  Booking:  {d['booking_id']}",
                 f"  Provider: {name}",
                 f"  From:     {display_slot(d['old_slot'])}",
                 f"  To:       {display_slot(d['slot'])}",
                 f"  Price:    {money(d['price'])}"]
    lines.append("Reply \"yes\" to go ahead or \"no\" to stop.")
    text = "\n".join(lines)
    return f"{intro}\n\n{text}" if intro else text
