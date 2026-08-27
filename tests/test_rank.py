"""
Contract tests for rank.py's hand-set weighted scoring (design doc §3.4
Step 6, §8.3 step C3).

Scope: score_candidates()/rank() and their supporting weighted-sum/sort/
truncate/top_k machinery — the part that is actually implemented and
running today (verified live via `python3 agent.py`). Individual feature
*values* (price_fit, category_match, etc.) are features.py's contract,
already covered by tests/test_features.py; this file only tests how
rank.py combines and orders those values.

OUT OF SCOPE, deliberately: fit_logistic_regression()'s actual fitting
behavior (C5) is blocked on Chellappan's D6 delivering the ~4,200-row
feature matrix — that function is still a stub that ignores its inputs
and returns HANDSET_WEIGHTS regardless. The one test below for it
(test_fit_logistic_regression_stub_ignores_inputs) protects only the
documented placeholder behavior, and must be rewritten once C5 actually
fits a real model. Likewise llm_rerank() (C7, closed as won't-do — no
LLM access provided) is tested only for its current "always None, always
falls back" stub contract.
"""

import math

import pytest

from features import FEATURE_NAMES
from indexes import Indexes
from rank import (
    HANDSET_WEIGHTS,
    TOP_K_RETURN,
    TOP_K_TRUNCATE,
    FittedRanker,
    fit_logistic_regression,
    llm_rerank,
    rank,
    score_candidates,
)
from state import SessionState
from utils import Candidate


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def facts():
    """Three products, distinct pop, everything else held equal.

    Isolates ordering to the one feature that varies, so ranking tests can
    assert exact expected order without needing to override weights.
    """
    def record(pop):
        return {
            "dept": "Men",
            "cat3": "Clothing, Shoes & Jewelry > Men > Clothing > Jackets & Coats",
            "store": "generic",
            "price": None,
            "rating_number": 0,
            "pop": pop,
            "rating": 4.0,
            "blob": "a generic product with no special terms",
        }

    return {
        "HIGH_POP": record(0.9),
        "MID_POP": record(0.5),
        "LOW_POP": record(0.1),
    }


@pytest.fixture
def indexes(facts):
    return Indexes(
        catalog=[], fts_conn=None, embedding_matrix=None, embedding_asins=[],
        facts=facts, category_lists={},
    )


@pytest.fixture
def state_empty():
    """No slots, no profile terms — every constraint-satisfaction feature
    (category_match, brand_match, slot_coverage, rare_tag_match,
    rating_style_fit) evaluates to 0.0 for every candidate, so only pop/
    rating/bm25_norm/cos_sim/price_fit can move the score."""
    return SessionState(session_id="s")


@pytest.fixture
def pool():
    """Three candidates, no stream scores — isolates ordering to pop/rating."""
    return [
        Candidate(asin="HIGH_POP"),
        Candidate(asin="MID_POP"),
        Candidate(asin="LOW_POP"),
    ]


# --------------------------------------------------------------------------
# HANDSET_WEIGHTS — §8.3 step C3: "Constants at module top for single-line
# tuning."
# --------------------------------------------------------------------------

def test_handset_weights_has_one_entry_per_feature():
    assert set(HANDSET_WEIGHTS) == set(FEATURE_NAMES)


def test_handset_weights_are_all_floats():
    for name, weight in HANDSET_WEIGHTS.items():
        assert isinstance(weight, float), f"{name} weight is {type(weight)}"


# --------------------------------------------------------------------------
# score_candidates() — the weighted-sum arithmetic itself
# --------------------------------------------------------------------------

def test_score_candidates_returns_one_pair_per_candidate(indexes, state_empty, pool):
    scored = score_candidates(pool, state_empty, indexes)
    assert len(scored) == len(pool)
    assert {asin for asin, _ in scored} == {c.asin for c in pool}


def test_score_candidates_defaults_to_handset_weights(indexes, state_empty, pool):
    """weights=None must produce the identical result to passing HANDSET_WEIGHTS explicitly."""
    default = dict(score_candidates(pool, state_empty, indexes))
    explicit = dict(score_candidates(pool, state_empty, indexes, HANDSET_WEIGHTS))
    assert default == explicit


def test_score_candidates_respects_custom_weights(indexes, state_empty, pool):
    """Zero every weight except pop: the score must equal facts[asin]['pop'] exactly."""
    isolate_pop = {name: 0.0 for name in FEATURE_NAMES}
    isolate_pop["pop"] = 1.0
    scored = dict(score_candidates(pool, state_empty, indexes, isolate_pop))
    assert scored["HIGH_POP"] == pytest.approx(0.9)
    assert scored["MID_POP"] == pytest.approx(0.5)
    assert scored["LOW_POP"] == pytest.approx(0.1)


def test_score_candidates_missing_weight_entries_default_to_zero_contribution(
    indexes, state_empty, pool
):
    """A weights dict that omits some FEATURE_NAMES must not crash or count them."""
    only_rating = {"rating": 1.0}  # every other feature implicitly weight 0
    scored = dict(score_candidates(pool, state_empty, indexes, only_rating))
    # All three candidates share rating=4.0/5.0=0.8 in the fixture, so scores
    # must all be equal — proves the other nine features contributed nothing.
    assert scored["HIGH_POP"] == pytest.approx(0.8)
    assert scored["MID_POP"] == pytest.approx(0.8)
    assert scored["LOW_POP"] == pytest.approx(0.8)


def test_score_candidates_survives_empty_pool(indexes, state_empty):
    assert score_candidates([], state_empty, indexes) == []


def test_score_candidates_survives_asin_absent_from_facts(indexes, state_empty):
    """A candidate feature_vector can't compute for shouldn't crash the whole turn."""
    scored = score_candidates(
        [Candidate(asin="DOES_NOT_EXIST")], state_empty, indexes
    )
    assert len(scored) == 1
    asin, value = scored[0]
    assert asin == "DOES_NOT_EXIST"
    assert isinstance(value, float)
    assert not math.isnan(value)


# --------------------------------------------------------------------------
# rank() — sort, truncate, top_k, ranker override, llm_rerank fallback
# --------------------------------------------------------------------------

def test_rank_orders_by_score_descending(indexes, state_empty, pool):
    """HANDSET_WEIGHTS weights pop positively; higher pop must rank first."""
    ranked = rank(pool, state_empty, indexes, top_k=3)
    assert ranked == ["HIGH_POP", "MID_POP", "LOW_POP"]


def test_rank_returns_exactly_top_k(indexes, state_empty, pool):
    ranked = rank(pool, state_empty, indexes, top_k=2)
    assert len(ranked) == 2
    assert ranked == ["HIGH_POP", "MID_POP"]


def test_rank_returns_fewer_than_top_k_if_pool_is_smaller(indexes, state_empty, pool):
    ranked = rank(pool, state_empty, indexes, top_k=10)
    assert len(ranked) == 3  # pool only has 3 candidates


def test_rank_default_top_k_matches_top_k_return_constant(indexes, state_empty, pool):
    ranked = rank(pool, state_empty, indexes)
    assert len(ranked) <= TOP_K_RETURN


def test_rank_truncates_to_top_k_truncate_before_top_k_slice(indexes, facts):
    """§3.4 Step 6: 'truncate to 30 ... return 10.' A pool bigger than
    TOP_K_TRUNCATE must never return more than TOP_K_TRUNCATE, even if a
    caller asks for more via top_k."""
    big_facts = {f"ASIN_{i}": {**facts["HIGH_POP"], "pop": i / 100} for i in range(40)}
    big_indexes = Indexes(
        catalog=[], fts_conn=None, embedding_matrix=None, embedding_asins=[],
        facts=big_facts, category_lists={},
    )
    big_pool = [Candidate(asin=asin) for asin in big_facts]
    state = SessionState(session_id="s")
    ranked = rank(big_pool, state, big_indexes, top_k=35)
    assert len(ranked) == TOP_K_TRUNCATE


def test_rank_survives_empty_pool(indexes, state_empty):
    assert rank([], state_empty, indexes) == []


def test_rank_uses_ranker_weights_when_supplied(indexes, state_empty, pool):
    """A FittedRanker weighting only rating (equal across the fixture) must
    flatten the ordering that pop alone would otherwise produce."""
    isolate_rating = {name: 0.0 for name in FEATURE_NAMES}
    isolate_rating["rating"] = 1.0
    ranker = FittedRanker(weights=isolate_rating)
    ranked = rank(pool, state_empty, indexes, ranker=ranker, top_k=3)
    # All three tie at rating=0.8, so order falls back to Python's stable
    # sort over the original pool order rather than pop-descending.
    assert set(ranked) == {"HIGH_POP", "MID_POP", "LOW_POP"}
    assert ranked == ["HIGH_POP", "MID_POP", "LOW_POP"]  # stable sort, input order preserved on ties


def test_rank_without_ranker_falls_back_to_handset_weights(indexes, state_empty, pool):
    with_none = rank(pool, state_empty, indexes, ranker=None, top_k=3)
    with_handset = rank(
        pool, state_empty, indexes, ranker=FittedRanker(weights=dict(HANDSET_WEIGHTS)), top_k=3
    )
    assert with_none == with_handset


def test_rank_llm_rerank_flag_off_by_default_does_not_change_order(indexes, state_empty, pool):
    without_flag = rank(pool, state_empty, indexes, top_k=3)
    with_flag_off = rank(pool, state_empty, indexes, use_llm_rerank=False, top_k=3)
    assert without_flag == with_flag_off


def test_rank_llm_rerank_stub_falls_back_to_regression_ordering(indexes, state_empty, pool):
    """llm_rerank() is a stub that always returns None (§8.3 step C7's
    defensive-parsing contract: malformed/no output -> fall back). Since it
    always returns None right now, use_llm_rerank=True must currently
    produce the identical order to use_llm_rerank=False."""
    without_flag = rank(pool, state_empty, indexes, top_k=3)
    with_flag_on = rank(pool, state_empty, indexes, use_llm_rerank=True, top_k=3)
    assert without_flag == with_flag_on


# --------------------------------------------------------------------------
# FittedRanker — default shape, independent per-instance weights
# --------------------------------------------------------------------------

def test_fitted_ranker_defaults_to_handset_weights():
    ranker = FittedRanker()
    assert ranker.weights == HANDSET_WEIGHTS
    assert ranker.intercept == 0.0
    assert ranker.feature_means == {}
    assert ranker.feature_stds == {}


def test_fitted_ranker_default_weights_are_an_independent_copy():
    """Mutating one instance's weights must not leak into HANDSET_WEIGHTS
    or into another instance — field(default_factory=...) must actually be
    called per-instance, not shared as one mutable default."""
    a = FittedRanker()
    b = FittedRanker()
    a.weights["pop"] = 999.0
    assert b.weights["pop"] != 999.0
    assert HANDSET_WEIGHTS["pop"] != 999.0


# --------------------------------------------------------------------------
# fit_logistic_regression() / llm_rerank() — current STUB contracts only.
# These protect placeholder behavior, not real fitting/LLM logic. Rewrite
# fit_logistic_regression's test once C5 lands (blocked on Chellappan/D6);
# llm_rerank's stays as-is (C7 closed as won't-do, no LLM access).
# --------------------------------------------------------------------------

def test_fit_logistic_regression_stub_ignores_inputs():
    ranker = fit_logistic_regression(
        feature_matrix=[{"pop": 0.5}], labels=[1], groups=["session-1"]
    )
    assert isinstance(ranker, FittedRanker)
    assert ranker.weights == HANDSET_WEIGHTS
    assert ranker.intercept == 0.0


def test_llm_rerank_stub_always_returns_none():
    result = llm_rerank(candidates=[], dialogue=["hi"], slots={"category": "jacket"})
    assert result is None
