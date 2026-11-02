"""The explicit workflow model: a persisted state machine with an enforced transition table."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

from .mocks import fmt_seconds

INTAKE = "INTAKE"
COMPARING = "COMPARING"
AWAITING_APPROVAL = "AWAITING_APPROVAL"
EXECUTING = "EXECUTING"
VERIFYING = "VERIFYING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
UNRESOLVED = "UNRESOLVED"
ABANDONED = "ABANDONED"

TRANSITIONS = {
    INTAKE: {COMPARING, AWAITING_APPROVAL, ABANDONED, FAILED},
    COMPARING: {INTAKE, AWAITING_APPROVAL, ABANDONED, FAILED},
    AWAITING_APPROVAL: {EXECUTING, COMPARING, INTAKE, ABANDONED},
    EXECUTING: {VERIFYING, AWAITING_APPROVAL, COMPARING, FAILED, UNRESOLVED},
    VERIFYING: {COMPLETED, EXECUTING, AWAITING_APPROVAL, COMPARING, FAILED, UNRESOLVED},
    UNRESOLVED: {VERIFYING, EXECUTING, COMPLETED, FAILED, COMPARING, AWAITING_APPROVAL},
    COMPLETED: set(),
    FAILED: set(),
    ABANDONED: set(),
}
TERMINAL = {COMPLETED, FAILED, ABANDONED}

REQUEST_FIELDS = ("service", "area", "date_from", "date_to", "earliest_time", "latest_time", "budget", "budget_waived")
REQUIRED_FOR_BOOKING = ("service", "area", "date_range", "budget")


class InvalidTransition(Exception):
    pass


@dataclass
class Workflow:
    id: str
    user_id: str
    kind: str | None
    state: str
    request: dict = field(default_factory=dict)
    preferences: dict = field(default_factory=dict)
    candidates: dict = field(default_factory=dict)
    holds: dict = field(default_factory=dict)
    origin: dict = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    def still_missing(self) -> list[str]:
        if self.kind in ("cancel", "reschedule"):
            return []
        missing = []
        if not self.request.get("service"):
            missing.append("service")
        if not self.request.get("area"):
            missing.append("area")
        if not (self.request.get("date_from") and self.request.get("date_to")):
            missing.append("date_range")
        if self.request.get("budget") is None and not self.request.get("budget_waived"):
            missing.append("budget")
        return missing

    def known_value(self, name: str):
        if name == "date_range":
            if self.request.get("date_from") and self.request.get("date_to"):
                return {"start": self.request["date_from"], "end": self.request["date_to"]}
            return None
        if name == "budget" and self.request.get("budget_waived"):
            return "no limit"
        return self.request.get(name) or self.preferences.get(name)

    def summary(self) -> dict:
        """What the model is told about the workflow at the start of a run."""
        return {
            "workflow_id": self.id,
            "kind": self.kind,
            "state": self.state,
            "request": self.request,
            "preferences": self.preferences,
            "still_missing": self.still_missing(),
            "candidate_provider_ids": sorted(self.candidates)[:8],
        }


class WorkflowStore:
    def __init__(self, db, clock):
        self.db = db
        self.clock = clock

    def create(self, user_id: str, kind: str | None, tracer, origin: dict | None = None) -> Workflow:
        workflow = Workflow(id=f"wf-{uuid.uuid4().hex[:10]}", user_id=user_id, kind=kind, state=INTAKE,
                            origin=origin or {})
        now = fmt_seconds(self.clock.now())
        self.db.execute("INSERT INTO workflows VALUES (?,?,?,?,?,?,?)",
                        (workflow.id, user_id, kind, workflow.state, self._data(workflow), now, now))
        tracer.event("workflow_created", workflow_id=workflow.id, kind=kind, state=INTAKE, origin=workflow.origin)
        return workflow

    def active(self, user_id: str) -> Workflow | None:
        rows = self.db.all("SELECT * FROM workflows WHERE user_id=? AND state NOT IN (?,?,?) "
                           "ORDER BY updated_at DESC, rowid DESC", (user_id, COMPLETED, FAILED, ABANDONED))
        return self._load(rows[0]) if rows else None

    def get(self, workflow_id: str) -> Workflow | None:
        row = self.db.one("SELECT * FROM workflows WHERE id=?", (workflow_id,))
        return self._load(row) if row else None

    def save(self, workflow: Workflow) -> None:
        self.db.execute("UPDATE workflows SET kind=?, state=?, data=?, updated_at=? WHERE id=?",
                        (workflow.kind, workflow.state, self._data(workflow), fmt_seconds(self.clock.now()), workflow.id))

    def transition(self, workflow: Workflow, new_state: str, reason: str, tracer) -> None:
        if new_state == workflow.state:
            return
        if new_state not in TRANSITIONS[workflow.state]:
            tracer.event("invalid_transition", workflow_id=workflow.id, from_state=workflow.state, to_state=new_state)
            raise InvalidTransition(f"{workflow.state} -> {new_state}")
        previous, workflow.state = workflow.state, new_state
        self.save(workflow)
        tracer.event("state_transition", workflow_id=workflow.id, from_state=previous, to_state=new_state, reason=reason)

    @staticmethod
    def _data(workflow: Workflow) -> str:
        return json.dumps({"request": workflow.request, "preferences": workflow.preferences,
                           "candidates": workflow.candidates, "holds": workflow.holds, "origin": workflow.origin})

    @staticmethod
    def _load(row) -> Workflow:
        data = json.loads(row["data"])
        return Workflow(id=row["id"], user_id=row["user_id"], kind=row["kind"], state=row["state"], **data)
