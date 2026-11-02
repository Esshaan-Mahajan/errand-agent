"""Persistent memory with provenance. Explicit and inferred facts stay distinct in every path.

- explicit:  the user stated it (a verbatim quote from their own message is required)
- inferred:  derived or unverified; stored and inspectable, but never sent to the model
- confirmed: was inferred, and the user then confirmed it
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from .clock import parse
from .mocks import fmt_seconds
from .sanitize import neutralise

EXPLICIT = "explicit"
INFERRED = "inferred"
CONFIRMED = "confirmed"

ACTIVE = "active"
SUPERSEDED = "superseded"
FORGOTTEN = "forgotten"

# Standing preferences that are relevant to any errand.
CORE_KEYS = {"preferred_area", "earliest_time", "latest_time", "default_budget"}

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "my", "for", "to", "of", "in", "at", "on", "and", "or", "is", "it", "i", "me", "this",
         "that", "with", "please", "find", "book", "get", "need", "want", "last", "time", "same", "as", "but", "not"}


@dataclass
class Memory:
    id: str
    user_id: str
    key: str
    value: str | None
    kind: str
    status: str
    category: str
    source: str
    valid_until: str | None
    created_at: str
    updated_at: str


@dataclass
class Selection:
    included: list[tuple[Memory, int]]
    excluded: dict[str, list[str]]


def tokens_of(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class MemoryStore:
    def __init__(self, db, clock):
        self.db = db
        self.clock = clock

    def add(self, user_id: str, key: str, value: str, kind: str = EXPLICIT, category: str = "general",
            source: str = "conversation", valid_until: str | None = None) -> Memory:
        """Store a fact. A new explicit fact replaces the current one; an inferred fact never
        overrides anything the user actually said."""
        now = fmt_seconds(self.clock.now())
        with self.db.transaction():
            current = self.db.one("SELECT * FROM memories WHERE user_id=? AND key=? AND status=?", (user_id, key, ACTIVE))
            if current is not None:
                if kind == INFERRED and current["kind"] != INFERRED:
                    return _memory(current)
                self.db.execute("UPDATE memories SET status=?, updated_at=? WHERE id=?", (SUPERSEDED, now, current["id"]))
            memory = Memory(f"mem-{uuid.uuid4().hex[:10]}", user_id, key, value, kind, ACTIVE, category, source,
                            valid_until, now, now)
            self.db.execute("INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (memory.id, user_id, key, value, kind, ACTIVE, category, source, valid_until, now, now))
        return memory

    def inspect(self, user_id: str) -> list[Memory]:
        rows = self.db.all("SELECT * FROM memories WHERE user_id=? AND status=? ORDER BY category, key", (user_id, ACTIVE))
        return [_memory(r) for r in rows]

    def all_rows(self, user_id: str) -> list[Memory]:
        return [_memory(r) for r in self.db.all("SELECT * FROM memories WHERE user_id=?", (user_id,))]

    def find(self, user_id: str, selector: str) -> list[Memory]:
        """Match active memories by exact key first, then by words in the key or value."""
        active = self.inspect(user_id)
        exact = [m for m in active if m.key == selector or m.id == selector]
        if exact:
            return exact
        wanted = tokens_of(selector.replace("_", " "))
        if not wanted:
            return []
        scored = []
        for memory in active:
            have = tokens_of(f"{memory.key.replace('_', ' ')} {memory.value or ''}")
            overlap = len(wanted & have)
            if overlap:
                scored.append((overlap, memory))
        if not scored:
            return []
        best = max(score for score, _ in scored)
        return [memory for score, memory in scored if score == best]

    def correct(self, user_id: str, key: str, value: str, source: str) -> Memory:
        existing = self.db.one("SELECT category FROM memories WHERE user_id=? AND key=? AND status=?", (user_id, key, ACTIVE))
        return self.add(user_id, key, value, EXPLICIT, existing["category"] if existing else "general", source)

    def confirm(self, memory_id: str) -> None:
        self.db.execute("UPDATE memories SET kind=?, updated_at=? WHERE id=? AND kind=?",
                        (CONFIRMED, fmt_seconds(self.clock.now()), memory_id, INFERRED))

    def forget(self, memory: Memory) -> str | None:
        """Tombstone a fact, wiping its value and any superseded versions. Returns the wiped value
        so the caller can scrub it from stored conversation, workflow state and traces."""
        now = fmt_seconds(self.clock.now())
        self.db.execute("UPDATE memories SET status=?, value=NULL, updated_at=? WHERE user_id=? AND key=?",
                        (FORGOTTEN, now, memory.user_id, memory.key))
        return memory.value

    def preference(self, user_id: str, key: str) -> str | None:
        """Read one fact for use in enforcement. Unconfirmed inferences never constrain anything."""
        row = self.db.one("SELECT value FROM memories WHERE user_id=? AND key=? AND status=? AND kind IN (?,?)",
                          (user_id, key, ACTIVE, EXPLICIT, CONFIRMED))
        return row["value"] if row else None

    def select_for_context(self, user_id: str, query_terms: set[str], service: str | None) -> Selection:
        """Rank what may reach the model. Exclusion reasons are logged in the trace as proof."""
        now = self.clock.now()
        excluded: dict[str, list[str]] = {"forgotten": [], "superseded": [], "inferred_unconfirmed": [],
                                          "expired": [], "irrelevant": []}
        scored: list[tuple[Memory, int]] = []
        for memory in self.all_rows(user_id):
            if memory.status == FORGOTTEN:
                excluded["forgotten"].append(memory.id)
            elif memory.status == SUPERSEDED:
                excluded["superseded"].append(memory.id)
            elif memory.kind == INFERRED:
                excluded["inferred_unconfirmed"].append(memory.id)
            elif memory.valid_until and parse(memory.valid_until) < now:
                excluded["expired"].append(memory.id)
            else:
                score = 0
                if memory.key in CORE_KEYS:
                    score += 3
                if service and memory.category == service:
                    score += 3
                score += len(query_terms & tokens_of(f"{memory.key.replace('_', ' ')} {memory.value or ''} {memory.category}"))
                if score > 0:
                    scored.append((memory, score))
                else:
                    excluded["irrelevant"].append(memory.id)
        scored.sort(key=lambda item: (-item[1], item[0].key))
        return Selection(scored, excluded)


def render_memory_line(memory: Memory) -> str:
    return f"- {memory.key} = {neutralise(memory.value or '', 120)} ({memory.kind})"


def render_inspection(memories: list[Memory]) -> str:
    """Shown directly to the user, so inferred facts can be reviewed without reaching the model."""
    if not memories:
        return "I don't have anything stored about you."
    stated = [m for m in memories if m.kind in (EXPLICIT, CONFIRMED)]
    guessed = [m for m in memories if m.kind == INFERRED]
    lines = []
    if stated:
        lines.append("Things you told me:")
        lines += [f"  - {m.key}: {m.value}" + (" (confirmed)" if m.kind == CONFIRMED else "") for m in stated]
    if guessed:
        lines.append("Things I inferred (not used until you confirm them):")
        lines += [f"  - {m.key}: {m.value}" for m in guessed]
    lines.append("You can ask me to correct or forget any of these.")
    return "\n".join(lines)


def _memory(row) -> Memory:
    return Memory(**dict(row))
