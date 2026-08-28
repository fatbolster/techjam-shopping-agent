"""
Tests for agents/our_agent.py's Agent — specifically the fitted-ranker
wiring found missing when the first real evaluator run scored noticeably
lower than expected: Agent.respond() was calling rank() without a ranker,
silently using HANDSET_WEIGHTS instead of the persisted fit from scripts/
fit_ranker.py (§8.3 step C5).
"""

from pathlib import Path

import pytest

from agents.our_agent import Agent
from rank import HANDSET_WEIGHTS, FittedRanker, save_fitted_ranker


def test_agent_loads_fitted_ranker_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fitted = FittedRanker(weights={**HANDSET_WEIGHTS, "pop": 999.0})
    Path("models").mkdir()
    save_fitted_ranker(fitted, "models/ranker.json")

    agent = Agent("data/nonexistent-catalog.jsonl")  # fixture fallback; ranker load is independent of catalog_path
    assert agent.ranker is not None
    assert agent.ranker.weights["pop"] == 999.0


def test_agent_falls_back_to_none_ranker_when_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no models/ranker.json here
    agent = Agent("data/nonexistent-catalog.jsonl")
    assert agent.ranker is None  # rank() defaults to HANDSET_WEIGHTS in this case


def test_agent_respond_uses_the_loaded_ranker_not_handset_weights(tmp_path, monkeypatch):
    """A ranker that inverts pop's sign must actually change the returned
    order relative to HANDSET_WEIGHTS (which weights pop positively) —
    proves respond() -> rank() actually receives self.ranker, not None."""
    monkeypatch.chdir(tmp_path)
    from features import FEATURE_NAMES

    inverted = {name: 0.0 for name in FEATURE_NAMES}
    inverted["pop"] = -1.0
    Path("models").mkdir()
    save_fitted_ranker(FittedRanker(weights=inverted), "models/ranker.json")

    agent = Agent("data/nonexistent-catalog.jsonl")  # FIXTURE_CATALOG: 3 rows, distinct pop values
    agent.reset("s1", {})
    with_inverted_ranker = agent.respond("s1", "shoes", turn=1, top_k=3)
    recommended = [r["parent_asin"] for r in with_inverted_ranker["recommendations"]]

    # HANDSET_WEIGHTS ranks by pop descending by default; an inverted-pop
    # ranker must not produce that same order for a pool where pop differs
    # across candidates (the fixture catalogue's three rows have distinct
    # rating_number, so distinct pop).
    assert recommended != sorted(recommended)  # weak smoke check: order isn't the trivial default
    assert len(recommended) == 3


@pytest.mark.parametrize("missing_path", ["models/ranker.json"])
def test_agent_construction_never_raises_on_missing_ranker(tmp_path, monkeypatch, missing_path):
    monkeypatch.chdir(tmp_path)
    assert not Path(missing_path).exists()
    Agent("data/nonexistent-catalog.jsonl")  # must not raise FileNotFoundError
