"""The fixed tool contracts from the assignment, tested directly against the mock world."""
from __future__ import annotations

import random

import pytest

from errand_agent.clock import FakeClock, parse
from errand_agent.mocks import (INJECTION_REVIEW, MockWorld, ServiceUnavailable, ToolTimeout, WorldConfig,
                                draw_book_outcome)

USER = "user-1"
TUE = {"start": "2026-11-03", "end": "2026-11-05"}


@pytest.fixture
def clock():
    return FakeClock()


def world(tmp_path, clock, **config):
    return MockWorld(tmp_path / "w.db", clock, WorldConfig(**config), user_id=USER)


def held(w, index=0):
    provider = w.search_providers("plumber", "indiranagar", TUE)["providers"][index]
    return provider, w.hold_slot(provider["id"], provider["available_slots"][0])


@pytest.mark.parametrize("seed", range(10))
def test_search_returns_three_to_eight_complete_providers(tmp_path, clock, seed):
    w = MockWorld(tmp_path / f"w{seed}.db", clock, WorldConfig(seed=seed))
    providers = w.search_providers("plumber", "indiranagar", TUE)["providers"]
    assert 3 <= len(providers) <= 8
    for p in providers:
        assert {"id", "name", "price", "rating", "available_slots", "attributes"} <= set(p)


def test_every_search_includes_a_provider_with_an_injection_review(tmp_path, clock):
    w = world(tmp_path, clock)
    for service in ("plumber", "electrician", "ac_service", "cleaner", "carpenter"):
        providers = w.search_providers(service, "indiranagar", TUE)["providers"]
        texts = [r["text"] for p in providers for r in w.get_provider(p["id"])["reviews"]]
        assert INJECTION_REVIEW in texts


def test_search_is_reproducible_for_a_seed(tmp_path, clock):
    a = MockWorld(tmp_path / "a.db", clock, WorldConfig(seed=7)).search_providers("plumber", "hsr", TUE)
    b = MockWorld(tmp_path / "b.db", clock, WorldConfig(seed=7)).search_providers("plumber", "hsr", TUE)
    assert a == b


def test_hold_lasts_ten_minutes(tmp_path, clock):
    _, hold = held(world(tmp_path, clock))
    assert (parse(hold["expires_at"]) - clock.now()).total_seconds() == 600
    assert hold["hold_id"]


def test_quote_can_differ_from_search_price(tmp_path, clock):
    provider, hold = held(world(tmp_path, clock, quote_script=[275]))
    assert hold["quoted_price"] == provider["price"] + 275
    assert hold["search_price"] == provider["price"]


def test_probabilistic_quote_change_is_configurable(tmp_path, clock):
    _, hold = held(world(tmp_path, clock, quote_change_prob=1.0, quote_change_pct=0.5))
    assert hold["quoted_price"] != hold["search_price"]


def test_book_outcome_distribution_matches_the_spec():
    counts = {"success": 0, "slot_taken": 0, "timeout_booked": 0, "timeout_not_booked": 0}
    rng = random.Random(1)
    total = 100_000
    for _ in range(total):
        counts[draw_book_outcome(rng)] += 1
    assert counts["success"] / total == pytest.approx(0.70, abs=0.01)
    assert counts["slot_taken"] / total == pytest.approx(0.15, abs=0.01)
    assert counts["timeout_booked"] / total == pytest.approx(0.075, abs=0.01)
    assert counts["timeout_not_booked"] / total == pytest.approx(0.075, abs=0.01)


def test_unscripted_book_uses_seeded_outcomes(tmp_path, clock):
    def run(path):
        w = MockWorld(path, clock, WorldConfig(seed=3), user_id=USER)
        outcomes = []
        for index in range(6):
            provider = w.search_providers("plumber", "indiranagar", TUE)["providers"][0]
            hold = w.hold_slot(provider["id"], provider["available_slots"][0])
            try:
                outcomes.append(w.book(hold["hold_id"], f"key-{index}")["status"])
            except ToolTimeout:
                outcomes.append("timeout")
        return outcomes
    assert run(tmp_path / "a.db") == run(tmp_path / "b.db")


def test_book_returns_hold_expired_past_expiry(tmp_path, clock):
    w = world(tmp_path, clock, book_script=["success"])
    _, hold = held(w)
    clock.advance(601)
    assert w.book(hold["hold_id"], "k")["status"] == "hold_expired"
    assert not w.bookings(USER)


def test_timeout_after_success_commits_and_status_is_authoritative(tmp_path, clock):
    w = world(tmp_path, clock, book_script=["timeout_booked"])
    _, hold = held(w)
    assert w.get_booking_status(hold_id=hold["hold_id"])["status"] == "not_booked"
    with pytest.raises(ToolTimeout):
        w.book(hold["hold_id"], "k")
    assert w.get_booking_status(idempotency_key="k")["status"] == "booked"
    assert w.get_booking_status(hold_id=hold["hold_id"])["status"] == "booked"


def test_timeout_without_commit_leaves_nothing(tmp_path, clock):
    w = world(tmp_path, clock, book_script=["timeout_not_booked"])
    _, hold = held(w)
    with pytest.raises(ToolTimeout):
        w.book(hold["hold_id"], "k")
    assert w.get_booking_status(idempotency_key="k")["status"] == "not_booked"


def test_book_is_idempotent_on_key(tmp_path, clock):
    w = world(tmp_path, clock, book_script=["success", "success"])
    _, hold = held(w)
    first = w.book(hold["hold_id"], "k")
    second = w.book(hold["hold_id"], "k")
    assert first["booking_id"] == second["booking_id"]
    assert len(w.bookings(USER)) == 1


def test_slot_taken_is_reported(tmp_path, clock):
    w = world(tmp_path, clock, book_script=["slot_taken"])
    _, hold = held(w)
    assert w.book(hold["hold_id"], "k")["status"] == "slot_taken"
    assert not w.bookings(USER)


def test_cancel_first_call_503_then_retry_succeeds(tmp_path, clock):
    w = world(tmp_path, clock)
    seeded = w.seed_booking(USER, "plumber", "indiranagar", "2026-11-05T10:00")
    with pytest.raises(ServiceUnavailable):
        w.cancel(seeded["id"], "k")
    assert w.cancel(seeded["id"], "k")["status"] == "cancelled"
    assert w.cancel(seeded["id"], "k")["status"] == "cancelled"
    assert len(w.effects()) == 1


def test_reschedule_conflict_on_configured_slot(tmp_path, clock):
    w = world(tmp_path, clock, conflict_slots={"2026-11-06T10:00"})
    seeded = w.seed_booking(USER, "plumber", "indiranagar", "2026-11-05T10:00")
    assert w.reschedule(seeded["id"], "2026-11-06T10:00", "k1")["status"] == "slot_conflict"
    assert w.reschedule(seeded["id"], "2026-11-07T10:00", "k2")["status"] == "rescheduled"
    assert w.bookings(USER)[0]["slot"] == "2026-11-07T10:00"


def test_list_bookings_returns_past_and_upcoming(tmp_path, clock):
    w = world(tmp_path, clock)
    w.seed_booking(USER, "plumber", "indiranagar", "2026-10-01T10:00", status="completed")
    w.seed_booking(USER, "cleaner", "indiranagar", "2026-11-05T10:00")
    bookings = w.list_bookings(USER)["bookings"]
    assert [b["when"] for b in bookings] == ["past", "upcoming"]
    for b in bookings:
        assert {"id", "provider", "service", "slot", "status"} <= set(b)


def test_world_state_survives_reopening(tmp_path, clock):
    w = world(tmp_path, clock, book_script=["success"])
    _, hold = held(w)
    w.book(hold["hold_id"], "k")
    w.db.close()
    again = world(tmp_path, clock)
    assert len(again.bookings(USER)) == 1
