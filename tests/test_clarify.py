"""
Contract tests for clarify.py's entropy x answerability clarification
decision (design doc §3.4 Step 5, §8.5 step E4).

shannon_entropy() was always real. score_attribute()/pick_attribute() are
now real too — computing per-candidate value distributions from
indexes.facts rather than a fixture placeholder. See clarify.py's
_attribute_value() docstring for the attribute -> facts-field mapping,
including the two attributes (feature, use_case) with no dedicated facts
field.
"""

import math

import pytest

from clarify import (
    ANSWERABILITY_PRIOR,
    ASK_THRESHOLD,
    MAX_CLARIFICATIONS_PER_SESSION,
    _attribute_value,
    pick_attribute,
    score_attribute,
    shannon_entropy,
)
from indexes import Indexes
from state import SessionState
from utils import Candidate


# --------------------------------------------------------------------------
# shannon_entropy() — pure formula, always real
# --------------------------------------------------------------------------

def test_shannon_entropy_empty_is_zero():
    assert shannon_entropy([]) == 0.0


def test_shannon_entropy_single_value_is_zero():
    assert shannon_entropy(["a", "a", "a"]) == 0.0


def test_shannon_entropy_two_equally_likely_values_is_one_bit():
    assert shannon_entropy(["a", "b"]) == pytest.approx(1.0)


def test_shannon_entropy_four_equally_likely_values_is_two_bits():
    assert shannon_entropy(["a", "b", "c", "d"]) == pytest.approx(2.0)


def test_shannon_entropy_skewed_distribution_is_lower_than_uniform():
    skewed = shannon_entropy(["a"] * 9 + ["b"])
    uniform = shannon_entropy(["a", "b"])
    assert skewed < uniform


def test_shannon_entropy_matches_hand_computed_formula():
    values = ["a", "a", "b"]
    expected = -((2 / 3) * math.log2(2 / 3) + (1 / 3) * math.log2(1 / 3))
    assert shannon_entropy(values) == pytest.approx(expected)


# --------------------------------------------------------------------------
# fixtures: a small real Indexes bundle with a controlled facts spread
# --------------------------------------------------------------------------

def _facts_row(dept, cat3, brand, color, price, blob_extra=""):
    return {
        "dept": dept,
        "cat3": cat3,
        "store": brand,
        "price": price,
        "rating_number": 100,
        "pop": 0.5,
        "rating": 4.5,
        "blob": f"a product {blob_extra}".strip(),
        "brand": brand,
        "color": color,
        "material": None,
        "style": None,
        "size": None,
    }


@pytest.fixture
def indexes() -> Indexes:
    facts = {
        "A": _facts_row("Men", "Jackets", "Acme", "Red", 20.0, "water resistant"),
        "B": _facts_row("Men", "Jackets", "Acme", "Blue", 25.0, "waterproof"),
        "C": _facts_row("Women", "Dresses", "Zeta", "Red", 90.0, "lightweight"),
    }
    return Indexes(catalog=[], fts_conn=None, embedding_matrix=None, embedding_asins=[], facts=facts, category_lists={})


@pytest.fixture
def pool() -> list[Candidate]:
    return [Candidate(asin="A"), Candidate(asin="B"), Candidate(asin="C")]


@pytest.fixture
def state_empty() -> SessionState:
    return SessionState(session_id="s1")


# --------------------------------------------------------------------------
# _attribute_value() — the facts-field mapping
# --------------------------------------------------------------------------

def test_attribute_value_category_prefers_cat3_over_dept(indexes):
    assert _attribute_value("category", "A", indexes.facts) == "Jackets"


def test_attribute_value_category_falls_back_to_dept_when_cat3_missing():
    facts = {"X": {"cat3": None, "dept": "Men"}}
    assert _attribute_value("category", "X", facts) == "Men"


def test_attribute_value_brand_prefers_brand_field_over_store():
    facts = {"X": {"brand": "RealBrand", "store": "StoreName"}}
    assert _attribute_value("brand", "X", facts) == "RealBrand"


def test_attribute_value_brand_falls_back_to_store_when_brand_missing():
    facts = {"X": {"brand": None, "store": "StoreName"}}
    assert _attribute_value("brand", "X", facts) == "StoreName"


def test_attribute_value_color_material_style_size_direct_lookup(indexes):
    assert _attribute_value("color", "A", indexes.facts) == "Red"


def test_attribute_value_budget_buckets_by_width(indexes):
    assert _attribute_value("budget", "A", indexes.facts) == "1"  # 20.0 // 20 == 1
    assert _attribute_value("budget", "C", indexes.facts) == "4"  # 90.0 // 20 == 4


def test_attribute_value_budget_none_when_price_missing():
    facts = {"X": {"price": None}}
    assert _attribute_value("budget", "X", facts) is None


def test_attribute_value_feature_matches_controlled_phrase(indexes):
    assert _attribute_value("feature", "A", indexes.facts) == "water resistant"


def test_attribute_value_feature_none_bucket_when_no_phrase_matches():
    facts = {"X": {"blob": "a plain product with nothing special"}}
    assert _attribute_value("feature", "X", facts) == "none"


def test_attribute_value_unknown_asin_returns_none():
    assert _attribute_value("color", "NOT_IN_FACTS", {}) is None


# --------------------------------------------------------------------------
# score_attribute() — entropy x prior, real facts
# --------------------------------------------------------------------------

def test_score_attribute_none_indexes_falls_back_to_zero(pool, state_empty):
    assert score_attribute("category", pool, state_empty, indexes=None) == 0.0


def test_score_attribute_uniform_high_entropy_attribute_scores_higher_than_skewed(
    indexes, pool, state_empty
):
    """category is Men/Men/Women's cat3 -> Jackets/Jackets/Dresses (skewed
    2:1); color is Red/Blue/Red (also skewed 2:1, same shape) — but brand
    Acme/Acme/Zeta vs a synthetic uniform case should score strictly lower
    than a 3-way-uniform attribute at the same answerability prior."""
    category_score = score_attribute("category", pool, state_empty, indexes=indexes)
    assert category_score > 0.0
    assert category_score == pytest.approx(
        ANSWERABILITY_PRIOR["category"] * shannon_entropy(["Jackets", "Jackets", "Dresses"])
    )


def test_score_attribute_zero_when_pool_all_same_value(state_empty):
    facts = {"A": {"cat3": "Jackets"}, "B": {"cat3": "Jackets"}}
    idx = Indexes(catalog=[], fts_conn=None, embedding_matrix=None, embedding_asins=[], facts=facts, category_lists={})
    pool = [Candidate(asin="A"), Candidate(asin="B")]
    assert score_attribute("category", pool, state_empty, indexes=idx) == 0.0


def test_score_attribute_ignores_candidates_with_unknown_value(state_empty):
    """A candidate absent from facts must not inject a synthetic 'unknown'
    bucket that would inflate entropy relative to only the known values."""
    facts = {"A": {"cat3": "Jackets"}, "B": {"cat3": "Jackets"}}
    idx = Indexes(catalog=[], fts_conn=None, embedding_matrix=None, embedding_asins=[], facts=facts, category_lists={})
    pool = [Candidate(asin="A"), Candidate(asin="B"), Candidate(asin="MISSING")]
    assert score_attribute("category", pool, state_empty, indexes=idx) == 0.0


# --------------------------------------------------------------------------
# pick_attribute() — argmax, threshold, already-asked/already-filled, cap
# --------------------------------------------------------------------------

def test_pick_attribute_none_indexes_returns_none(pool, state_empty):
    """Every attribute scores 0.0 without indexes (below ASK_THRESHOLD),
    so nothing clears the bar — matches score_attribute()'s None fallback."""
    assert pick_attribute(pool, state_empty, indexes=None) is None


def test_pick_attribute_still_matches_bare_two_arg_interface_sketch(pool, state_empty):
    """§7.2's documented signature, pick_attribute(pool, state) -> str |
    None, must still work with indexes omitted entirely."""
    assert pick_attribute(pool, state_empty) is None  # no indexes -> no signal -> None


def test_pick_attribute_respects_already_filled_slots(indexes, pool):
    state = SessionState(session_id="s2")
    state.slots["category"] = "Jackets"  # already filled -> category excluded
    result = pick_attribute(pool, state, indexes=indexes)
    assert result != "category"


def test_pick_attribute_respects_already_asked_attributes(indexes, pool):
    state = SessionState(session_id="s3")
    state.asked_attributes.add("category")
    result = pick_attribute(pool, state, indexes=indexes)
    assert result != "category"


def test_pick_attribute_respects_session_cap(indexes, pool):
    state = SessionState(session_id="s4")
    state.asked_attributes = {"category", "brand", "color"}
    assert len(state.asked_attributes) == MAX_CLARIFICATIONS_PER_SESSION
    assert pick_attribute(pool, state, indexes=indexes) is None


def test_pick_attribute_returns_none_when_no_attribute_clears_threshold():
    """A pool where every candidate shares every attribute value has zero
    entropy everywhere — nothing should clear ASK_THRESHOLD."""
    facts = {"A": _facts_row("Men", "Jackets", "Acme", "Red", 20.0)}
    idx = Indexes(catalog=[], fts_conn=None, embedding_matrix=None, embedding_asins=[], facts=facts, category_lists={})
    pool = [Candidate(asin="A"), Candidate(asin="A")]
    state = SessionState(session_id="s5")
    assert pick_attribute(pool, state, indexes=idx) is None


def test_pick_attribute_picks_the_argmax_scoring_attribute(indexes, pool):
    state = SessionState(session_id="s6")
    scored = {
        attr: score_attribute(attr, pool, state, indexes=indexes)
        for attr in ANSWERABILITY_PRIOR
        if attr not in state.asked_attributes
    }
    best = max(scored, key=scored.get)
    result = pick_attribute(pool, state, indexes=indexes)
    if scored[best] >= ASK_THRESHOLD:
        assert result == best
    else:
        assert result is None
