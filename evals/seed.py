"""Seed data for the context-management scenarios: 60 stored memories and a long prior conversation."""
from __future__ import annotations

from errand_agent.memory import EXPLICIT, INFERRED

USER = "user-1"

# Values are distinctive strings, so a leak is a simple substring check.
RELEVANT = [
    ("preferred_area", "indiranagar", EXPLICIT, "general"),
    ("earliest_time", "09:00", EXPLICIT, "general"),
    ("preferred_technician:plumber", "Meera", EXPLICIT, "plumber"),
    ("plumber_note", "kitchen-tap-is-brass-marker", EXPLICIT, "plumber"),
]
INFERRED_FACTS = [
    ("default_budget", "inferred-budget-marker-7777", "general"),
    ("preferred_provider:plumber", "inferred-provider-marker", "plumber"),
    ("evening_person", "inferred-evening-marker", "general"),
]
IRRELEVANT_FACTS = [(f"misc_fact_{i}", f"irrelevant-marker-{i}", "misc") for i in range(44)]
SERVICE_NOISE = [
    ("preferred_technician:cleaner", "cleaner-tech-marker", "cleaner"),
    ("preferred_technician:carpenter", "carpenter-tech-marker", "carpenter"),
    ("ac_note", "ac-unit-marker-1-ton", "ac_service"),
]


def seed_memories(agent) -> dict:
    store = agent.memory
    for key, value, kind, category in RELEVANT:
        store.add(USER, key, value, kind, category, source="seed")
    for key, value, category in INFERRED_FACTS:
        store.add(USER, key, value, INFERRED, category, source="seed")
    for key, value, category in IRRELEVANT_FACTS + SERVICE_NOISE:
        store.add(USER, key, value, EXPLICIT, category, source="seed")

    # A preference that changed: only the newer value is current.
    store.add(USER, "usual_plumber_day", "superseded-day-marker", EXPLICIT, "plumber", source="seed")
    store.add(USER, "usual_plumber_day", "wednesday-current-marker", EXPLICIT, "plumber", source="seed")
    # A preference that has lapsed.
    store.add(USER, "plumber_promo", "expired-promo-marker", EXPLICIT, "plumber", source="seed",
              valid_until="2026-01-01T00:00")
    # A fact the user asked to forget.
    forgotten = store.add(USER, "old_plumber_address", "forgotten-address-marker", EXPLICIT, "plumber", source="seed")
    store.forget(forgotten)

    rows = store.all_rows(USER)
    return {"total": len(rows), "active": sum(1 for r in rows if r.status == "active")}


def seed_conversation(agent, turns: int = 150) -> int:
    """A long prior conversation (two messages per turn). The filler is long enough to overflow any budget."""
    count = 0
    agent.add_message(USER, "user", "Long ago: my plumber fixed the bathroom tap in Indiranagar, ancient-turn-marker.", None)
    agent.add_message(USER, "assistant", "Noted.", None)
    count += 2
    for i in range(turns):
        agent.add_message(USER, "user", f"Old question {i} about nothing in particular, filler text to take up room, history-marker-{i}.", None)
        agent.add_message(USER, "assistant", f"Old answer {i}, also filler that the context window should not need, reply-marker-{i}.", None)
        count += 2
    return count
