"""Regression tests for real bugs hit while building. docs/BUGS.md has the full stories."""
from __future__ import annotations

import pytest

from errand_agent.agent import Agent, _Run
from errand_agent.clock import FakeClock
from errand_agent.config import Limits
from errand_agent.db import AGENT_SCHEMA, Database
from errand_agent.llm.offline import RecordingLLM, StandInModel
from errand_agent.memory import EXPLICIT, INFERRED, MemoryStore, tokens_of
from errand_agent.mocks import MockWorld, ToolTimeout, WorldConfig
from errand_agent.sanitize import redact_pii
from errand_agent.tools import RunBudget

USER = "user-1"


def make_agent(tmp_path, **kwargs):
    llm = RecordingLLM(kwargs.pop("llm", StandInModel()))
    return Agent(tmp_path, llm, clock=FakeClock(), user_id=USER, trace_dir=tmp_path / "traces", **kwargs), llm


# Bug 3: book() opened a nested BEGIN through its counter helper, so every scripted book raised
# sqlite3.OperationalError, which the agent reported as a verified failure. The mock never committed.

def test_transaction_blocks_can_nest(tmp_path):
    db = Database(tmp_path / "t.db", "CREATE TABLE t (v INTEGER);")
    with db.transaction():
        db.execute("INSERT INTO t VALUES (1)")
        with db.transaction():
            db.execute("INSERT INTO t VALUES (2)")
    assert db.one("SELECT COUNT(*) AS n FROM t")["n"] == 2


def test_nested_transaction_rolls_back_as_one_unit(tmp_path):
    db = Database(tmp_path / "t.db", "CREATE TABLE t (v INTEGER);")
    with pytest.raises(ValueError):
        with db.transaction():
            db.execute("INSERT INTO t VALUES (1)")
            with db.transaction():
                db.execute("INSERT INTO t VALUES (2)")
            raise ValueError("boom")
    assert db.one("SELECT COUNT(*) AS n FROM t")["n"] == 0


def test_scripted_timeout_after_success_really_commits(tmp_path):
    world = MockWorld(tmp_path / "w.db", FakeClock(), WorldConfig(book_script=["timeout_booked"]))
    provider = world.search_providers("plumber", "indiranagar", {"start": "2026-11-03", "end": "2026-11-03"})["providers"][0]
    hold = world.hold_slot(provider["id"], provider["available_slots"][0])
    with pytest.raises(ToolTimeout):
        world.book(hold["hold_id"], "key-1")
    assert world.get_booking_status(idempotency_key="key-1")["status"] == "booked"
    assert len(world.bookings(USER)) == 1


# Also found: the unverified-success guard matched "Nothing has been booked" and replaced honest failure
# messages, including over-budget and search-outage explanations, with a generic line.

@pytest.mark.parametrize("honest", [
    "I couldn't search right now. Nothing has been booked. Please try again shortly.",
    "I can't propose that option (over_budget). Nothing has been booked. Should I look for another option?",
    "Your booking was not confirmed.",
    "It hasn't been rescheduled yet.",
])
def test_claim_guard_keeps_honest_negative_statements(tmp_path, honest):
    agent, _ = make_agent(tmp_path)
    tracer = agent.new_tracer(USER)
    run = _Run(USER, "hi", 1, tracer, RunBudget(Limits(), agent.clock))
    assert agent._guard_final_text(honest, run) == honest


@pytest.mark.parametrize("claim", [
    "Great news, your plumber has been booked for tomorrow!",
    "Your booking is confirmed.",
    "I've cancelled it for you.",
    "Nothing to worry about. Your visit has been rescheduled to Friday.",
    "No worries, your plumber has been booked.",
])
def test_claim_guard_still_blocks_false_claims(tmp_path, claim):
    agent, _ = make_agent(tmp_path)
    tracer = agent.new_tracer(USER)
    run = _Run(USER, "hi", 1, tracer, RunBudget(Limits(), agent.clock))
    assert agent._guard_final_text(claim, run) != claim


# Bug 2: a memory inspection showed inferred facts to the user, then was stored as an ordinary
# assistant message, so the next turn's history carried unconfirmed facts to the model.

def test_inspection_reply_never_enters_model_context(tmp_path):
    agent, llm = make_agent(tmp_path)
    agent.memory.add(USER, "preferred_area", "whitefield", INFERRED, source="history")
    shown = agent.handle_message(USER, "What do you remember about me?")
    assert "whitefield" in shown.text
    before = len(llm.requests)
    agent.handle_message(USER, "Find a plumber this week under 1,500")
    later = llm.requests[before:]
    assert later, "the second message should reach the model"
    assert all("whitefield" not in r.system + str(r.messages) for r in later)


def test_inspection_reply_stays_private_after_the_fact_is_confirmed(tmp_path):
    """A message's visibility is fixed when stored: confirming a fact doesn't republish old listings."""
    agent, llm = make_agent(tmp_path)
    agent.memory.add(USER, "evening_person", "late-riser", INFERRED, source="history")
    agent.handle_message(USER, "What do you remember about me?")
    assert agent.db.one("SELECT private FROM messages WHERE role='assistant'")["private"] == 1


# Also found: relevance scoring tokenised the budget "1,500" into "1" and "500", so it matched memories
# such as misc_fact_1 and sent irrelevant facts to the model.

def test_numbers_do_not_make_a_memory_relevant(tmp_path):
    assert not {"1", "500"} & tokens_of("under 1,500")
    db = Database(tmp_path / "a.db", AGENT_SCHEMA)
    store = MemoryStore(db, FakeClock())
    store.add(USER, "misc_fact_1", "unrelated-value-1", EXPLICIT, "misc", source="seed")
    store.add(USER, "preferred_area", "indiranagar", EXPLICIT, "general", source="seed")
    chosen = store.select_for_context(USER, tokens_of("Find a plumber this week, under 1,500"), "plumber")
    assert [m.key for m, _ in chosen.included] == ["preferred_area"]
    assert len(chosen.excluded["irrelevant"]) == 1


# Bug 1: approval phrases ignored the proposal's action, so "cancel it" (meaning stop) on a booking
# prompt counted as consent and booked the plumber.

def test_cancel_it_on_a_booking_prompt_does_not_book(tmp_path):
    agent, _ = make_agent(tmp_path, world_config=WorldConfig(book_script=["success"]))
    prompt = agent.handle_message(USER, "Find a plumber in Indiranagar this week, mornings only, under 1,500.")
    assert prompt.outcome == "awaiting_approval"
    reply = agent.handle_message(USER, "cancel it")
    assert reply.outcome == "rejected"
    assert agent.world.bookings(USER) == []
    assert agent.world.call_count("book") == 0


def test_forgotten_preference_is_not_inferred_back(tmp_path):
    agent, _ = make_agent(tmp_path)
    agent.world.seed_booking(USER, "plumber", "whitefield", "2026-11-03T10:00")
    agent.world.seed_booking(USER, "cleaner", "whitefield", "2026-11-04T10:00")
    memory = agent.memory.add(USER, "preferred_area", "whitefield", EXPLICIT, source="seed")
    agent.memory.forget(memory)
    agent._infer_from_history(USER, agent.new_tracer(USER))
    assert agent.memory.inspect(USER) == []


# Also found: the phone pattern counted characters, not digits, so every ISO date in every trace became
# "[phone]". Spotted by eye in the live trace pane; no test looked at trace contents until then.

@pytest.mark.parametrize("kept", ["2026-11-03", "2026-11-03T09:00", "2026-11-02T09:10:00", "1500", "bk-12345678", "hold-97d3986c6f"])
def test_redaction_leaves_dates_and_ids_alone(kept):
    assert redact_pii(kept) == kept


@pytest.mark.parametrize("phone", ["+91 98123 45678", "9812345678", "98123-45678", "+1 415 555 0132"])
def test_redaction_still_masks_phone_numbers(phone):
    assert redact_pii(f"call {phone} today") == "call [phone] today"


def test_stored_trace_keeps_dates_readable(tmp_path):
    agent, _ = make_agent(tmp_path)
    reply = agent.handle_message(USER, "Find a plumber in Indiranagar this week, mornings only, under 1,500.")
    with open(reply.trace_path, encoding="utf-8") as handle:
        text = handle.read()
    assert '"start": "2026-11-02"' in text
    assert "Monday 2026-11-02 09:00" in text
    assert '"contact": "[phone]"' in text
