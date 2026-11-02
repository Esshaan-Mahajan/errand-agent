"""Runs every eval scenario under pytest and regenerates the eval table at the end of the session."""
from __future__ import annotations

import pytest

from evals.runner import run_scenario, write_table
from evals.scenarios import SCENARIOS

_results = {}


@pytest.fixture(scope="module", autouse=True)
def write_results():
    yield
    if len(_results) == len(SCENARIOS):
        write_table([_results[s.id] for s in SCENARIOS])


def test_at_least_twenty_scenarios():
    assert len(SCENARIOS) >= 20
    assert len({s.id for s in SCENARIOS}) == len(SCENARIOS)


def test_required_categories_are_covered():
    required = {"adversarial", "injection", "price_change", "timeout", "slot_conflict", "duplicate", "memory",
                "restart", "proactive", "context", "limits", "tool_failure"}
    assert required <= {s.category for s in SCENARIOS}


@pytest.mark.parametrize("item", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario(item):
    result = run_scenario(item)
    _results[item.id] = result
    assert result.passed, f"{item.id}: {result.failed_checks}; actual: {result.actual}"
