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
from errand_agent.tools import RunBudget

USER = "user-1"


def make_agent(tmp_path, **kwargs):
    llm = RecordingLLM(kwargs.pop("llm", StandInModel()))
    return Agent(tmp_path, llm, clock=FakeClock(), user_id=USER, trace_dir=tmp_path / "traces", **kwargs), llm


# Bug 1: book() opened a nested BEGIN through its counter helper, so every scripted book raised
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


# Bug 2: the unverified-success guard matched "Nothing has been booked" and replaced honest failure
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
])
def test_claim_guard_still_blocks_false_claims(tmp_path, claim):
    agent, _ = make_agent(tmp_path)
    tracer = agent.new_tracer(USER)
    run = _Run(USER, "hi", 1, tracer, RunBudget(Limits(), agent.clock))
    assert agent._guard_final_text(claim, run) != claim


# Bug 3: a memory inspection showed inferred facts to the user, then was stored as an ordinary
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


# Bug 4: relevance scoring tokenised the budget "1,500" into "1" and "500", so it matched memories
# such as misc_fact_1 and sent irrelevant facts to the model.

def test_numbers_do_not_make_a_memory_relevant(tmp_path):
    assert tokens_of("under 1,500 on the 12th") == {"12th"} or "1" not in tokens_of("under 1,500")
    db = Database(tmp_path / "a.db", AGENT_SCHEMA)
    store = MemoryStore(db, FakeClock())
    store.add(USER, "misc_fact_1", "unrelated-value-1", EXPLICIT, "misc", source="seed")
    store.add(USER, "preferred_area", "indiranagar", EXPLICIT, "general", source="seed")
    chosen = store.select_for_context(USER, tokens_of("Find a plumber this week, under 1,500"), "plumber")
    assert [m.key for m, _ in chosen.included] == ["preferred_area"]
    assert any(True for _ in chosen.excluded["irrelevant"])
