"""
Contract tests for retrieval.py's three streams, union/dedupe, and the
floor check (design doc §3.4 Step 4, §8.1 steps A7-A9).

union_dedupe() (A8) was already real before this file existed. The rest —
keyword_stream's buy-track filter, semantic_stream's browse-track diversity
cap, popularity_stream's modal-department targeting, and floor_check's
relax-and-retry — are what's actually under test here.
"""

import numpy as np
import pytest

from indexes import build_indexes, Indexes
from retrieval import (
    CATEGORY_DIVERSITY_MAX_SHARE,
    POOL_FLOOR,
    STREAM_QUOTAS,
    floor_check,
    keyword_stream,
    popularity_stream,
    retrieve,
    semantic_stream,
    union_dedupe,
)
from state import SessionState
from utils import Candidate

sentence_transformers = pytest.importorskip("sentence_transformers")


# --------------------------------------------------------------------------
# A larger synthetic catalogue (fixture-sized real data is too small to
# meaningfully test quotas/floor/diversity — 3 rows can't fill a 50-row
# floor). Two departments, enough per-department volume for the diversity
# cap and floor-check top-up to have real headroom to work with.
# --------------------------------------------------------------------------

def _synthetic_catalog(n_per_dept: int = 40) -> list[dict]:
    catalog = []
    for dept in ("Men", "Women"):
        for i in range(n_per_dept):
            catalog.append(
                {
                    "parent_asin": f"{dept[0]}{i:03d}",
                    "title": f"{dept} running shoe model {i}",
                    "features": ["breathable mesh"],
                    "description": [f"A {dept.lower()} running shoe, style {i}."],
                    "categories": ["Clothing, Shoes & Jewelry", dept, "Shoes", "Running"],
                    "store": "Acme",
                    "details": {"Department": dept},
                    "price": 20.0 + i,
                    "rating_number": (n_per_dept - i) * 100,  # descending popularity within dept
                    "average_rating": 4.0,
                }
            )
    return catalog


@pytest.fixture(scope="module")
def indexes() -> Indexes:
    catalog = _synthetic_catalog()
    return build_indexes(catalog, embedding_cache_path=None)


@pytest.fixture
def state_buy(indexes) -> SessionState:
    s = SessionState(session_id="s1", track="buy")
    s.canonical_intent = "running shoe"
    s.canonical_vector = np.mean(indexes.embedding_matrix[:2], axis=0)
    s.canonical_vector = s.canonical_vector / np.linalg.norm(s.canonical_vector)
    return s


@pytest.fixture
def state_browse(indexes) -> SessionState:
    s = SessionState(session_id="s2", track="browse")
    s.canonical_intent = "running shoe"
    s.canonical_vector = np.mean(indexes.embedding_matrix, axis=0)
    s.canonical_vector = s.canonical_vector / np.linalg.norm(s.canonical_vector)
    return s


# --------------------------------------------------------------------------
# keyword_stream() — A7 buy-track department/category filter
# --------------------------------------------------------------------------

def test_keyword_stream_unfiltered_on_browse_track(indexes, state_browse):
    results = keyword_stream(state_browse, indexes, quota=20)
    assert len(results) > 0
    assert all(c.sources == {"keyword"} for c in results)


def test_keyword_stream_filters_by_department_on_buy_track(indexes, state_buy):
    state_buy.slots["department"] = "Men"
    results = keyword_stream(state_buy, indexes, quota=50)
    depts = {indexes.facts[c.asin]["dept"] for c in results}
    assert depts <= {"Men"}
    assert len(results) > 0  # the filter didn't zero out a department that does exist


def test_keyword_stream_department_filter_is_case_insensitive(indexes, state_buy):
    state_buy.slots["department"] = "men"  # lowercase, catalogue has "Men"
    results = keyword_stream(state_buy, indexes, quota=50)
    assert len(results) > 0
    assert all(indexes.facts[c.asin]["dept"] == "Men" for c in results)


def test_keyword_stream_category_filter_is_substring_on_blob(indexes, state_buy):
    state_buy.slots["category"] = "running"
    results = keyword_stream(state_buy, indexes, quota=50)
    assert len(results) > 0
    assert all("running" in indexes.facts[c.asin]["blob"] for c in results)


def test_keyword_stream_no_slots_set_behaves_like_browse_on_buy_track(indexes, state_buy):
    with_no_slots = keyword_stream(state_buy, indexes, quota=20)
    assert len(with_no_slots) > 0


def test_keyword_stream_respects_quota(indexes, state_browse):
    results = keyword_stream(state_browse, indexes, quota=5)
    assert len(results) <= 5


# --------------------------------------------------------------------------
# semantic_stream() — A7 (never filters) + A9 (browse-track diversity cap)
# --------------------------------------------------------------------------

def test_semantic_stream_none_vector_returns_empty(indexes):
    s = SessionState(session_id="s3", track="browse")
    assert semantic_stream(s, indexes, quota=10) == []


def test_semantic_stream_buy_track_no_cap_applied(indexes, state_buy):
    """Buy track: plain top-`quota` by cosine score, no diversity logic."""
    results = semantic_stream(state_buy, indexes, quota=10)
    assert len(results) == 10
    scores = [c.cos_raw for c in results]
    assert scores == sorted(scores, reverse=True)


def test_semantic_stream_browse_track_diversifies_vs_buy_track(indexes, state_browse, state_buy):
    """Smoke test against real embeddings: browse-track diversity must
    produce a *less* department-skewed result than buy-track's uncapped
    top-quota (exact cap-boundary behavior is verified deterministically
    below, since real MiniLM scores can't be hand-controlled precisely
    enough to test the cap/top-up boundary itself)."""
    quota = 20
    state_buy.canonical_vector = state_browse.canonical_vector  # same query, compare tracks only
    browse_results = semantic_stream(state_browse, indexes, quota=quota)
    buy_results = semantic_stream(state_buy, indexes, quota=quota)
    assert len(browse_results) == quota

    def max_dept_share(results):
        counts: dict[str, int] = {}
        for c in results:
            dept = indexes.facts[c.asin]["dept"]
            counts[dept] = counts.get(dept, 0) + 1
        return max(counts.values())

    assert max_dept_share(browse_results) <= max_dept_share(buy_results)


def _controlled_semantic_indexes(dept_scores: dict[str, list[float]]) -> Indexes:
    """A minimal Indexes with 2-d embeddings engineered so cosine
    similarity to query_vec=[1,0] equals the given score exactly (each row
    is `[score, sqrt(1-score**2)]`, already unit-norm) — lets the diversity
    cap be tested against an exact, controlled score ranking rather than
    real (unpredictable) MiniLM output.
    """
    asins, rows, facts = [], [], {}
    for dept, scores in dept_scores.items():
        for i, score in enumerate(scores):
            asin = f"{dept}_{i}"
            asins.append(asin)
            rows.append([score, (1 - score**2) ** 0.5])
            facts[asin] = {"dept": dept}
    matrix = np.array(rows, dtype=np.float32)
    return Indexes(catalog=[], fts_conn=None, embedding_matrix=matrix, embedding_asins=asins, facts=facts, category_lists={})


def test_semantic_stream_browse_track_diversity_cap_holds_with_ample_supply():
    """Four departments, interleaved scores, ample supply above the cap in
    each — the cap alone (no top-up fallback needed) must keep every
    department at or under floor(quota * CATEGORY_DIVERSITY_MAX_SHARE)."""
    quota = 20
    cap = max(1, int(quota * CATEGORY_DIVERSITY_MAX_SHARE))  # 6
    dept_scores = {
        dept: [0.90 - 0.01 * offset - 0.01 * i for i in range(20)]
        for offset, dept in enumerate(["A", "B", "C", "D"])
    }
    idx = _controlled_semantic_indexes(dept_scores)
    s = SessionState(session_id="s-cap", track="browse")
    s.canonical_vector = np.array([1.0, 0.0], dtype=np.float32)
    results = semantic_stream(s, idx, quota=quota)
    assert len(results) == quota
    counts: dict[str, int] = {}
    for c in results:
        dept = idx.facts[c.asin]["dept"]
        counts[dept] = counts.get(dept, 0) + 1
    assert len(counts) == 4  # every department represented
    assert max(counts.values()) <= cap


def test_semantic_stream_browse_track_uncapped_topup_when_one_dept_dominates():
    """With only one department having any real supply, the cap's top-up
    fallback must still fill the full quota from that one department
    rather than leaving the result short."""
    dept_scores = {"A": [0.9 - 0.01 * i for i in range(30)], "B": [0.5, 0.4]}
    idx = _controlled_semantic_indexes(dept_scores)
    s = SessionState(session_id="s-topup", track="browse")
    s.canonical_vector = np.array([1.0, 0.0], dtype=np.float32)
    results = semantic_stream(s, idx, quota=20)
    assert len(results) == 20  # cap alone (6+2=8) can't reach 20; top-up must cover the rest


def test_semantic_stream_browse_track_still_fills_quota_when_cap_would_undersize():
    """If one department has far more supply than the other, the uncapped
    top-up path must still reach `quota` rather than leaving it short."""
    catalog = _synthetic_catalog(n_per_dept=2)  # tiny "Women" supply relative to a large quota
    catalog += [
        {
            "parent_asin": f"M{i:03d}",
            "title": f"Men running shoe model {i}",
            "features": [],
            "description": [],
            "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Running"],
            "store": "Acme",
            "details": {"Department": "Men"},
            "price": 20.0,
            "rating_number": 100,
            "average_rating": 4.0,
        }
        for i in range(2, 30)
    ]
    idx = build_indexes(catalog, embedding_cache_path=None)
    s = SessionState(session_id="s4", track="browse")
    s.canonical_intent = "running shoe"
    s.canonical_vector = idx.embedding_matrix[0]
    results = semantic_stream(s, idx, quota=20)
    assert len(results) == 20  # quota still met even though the cap alone can't supply it


# --------------------------------------------------------------------------
# popularity_stream() — A7 modal-department targeting
# --------------------------------------------------------------------------

def test_popularity_stream_targets_pools_modal_department(indexes):
    pool = [Candidate(asin=f"M{i:03d}") for i in range(5)]  # all "Men"
    results = popularity_stream(pool, indexes, quota=10)
    assert len(results) == 10
    assert all(indexes.facts[c.asin]["dept"] == "Men" for c in results)


def test_popularity_stream_sorted_by_rating_number_descending(indexes):
    pool = [Candidate(asin="M000")]
    results = popularity_stream(pool, indexes, quota=5)
    rating_numbers = [indexes.facts[c.asin]["rating_number"] for c in results]
    assert rating_numbers == sorted(rating_numbers, reverse=True)


def test_popularity_stream_excludes_asins_already_in_pool(indexes):
    top_men = sorted(
        (a for a in indexes.facts if indexes.facts[a]["dept"] == "Men"),
        key=lambda a: indexes.facts[a]["rating_number"],
        reverse=True,
    )
    pool = [Candidate(asin=top_men[0])]
    results = popularity_stream(pool, indexes, quota=5)
    assert top_men[0] not in {c.asin for c in results}


def test_popularity_stream_empty_pool_still_returns_quota(indexes):
    results = popularity_stream([], indexes, quota=10)
    assert len(results) == 10


def test_popularity_stream_tops_up_from_other_departments_when_modal_runs_short():
    """Men has only 3 candidates total (2 left after excluding the pool's
    one); Women has ample supply — the modal-department (Men) slice alone
    can't reach quota=10, so the top-up path must draw the rest from Women."""
    catalog = [
        {
            "parent_asin": f"M{i:03d}",
            "title": f"Men item {i}",
            "categories": ["Root", "Men", "Cat"],
            "details": {"Department": "Men"},
            "store": "Acme",
            "price": 10.0,
            "rating_number": 100 - i,
            "average_rating": 4.0,
            "features": [],
            "description": [],
        }
        for i in range(3)
    ] + [
        {
            "parent_asin": f"W{i:03d}",
            "title": f"Women item {i}",
            "categories": ["Root", "Women", "Cat"],
            "details": {"Department": "Women"},
            "store": "Acme",
            "price": 10.0,
            "rating_number": 100 - i,
            "average_rating": 4.0,
            "features": [],
            "description": [],
        }
        for i in range(20)
    ]
    idx = build_indexes(catalog, embedding_cache_path=None)
    pool = [Candidate(asin="M000")]
    results = popularity_stream(pool, idx, quota=10)
    assert len(results) == 10  # Men (2 left) + top-up from Women
    depts = {idx.facts[c.asin]["dept"] for c in results}
    assert depts == {"Men", "Women"}


# --------------------------------------------------------------------------
# union_dedupe() — already real; light regression coverage
# --------------------------------------------------------------------------

def test_union_dedupe_merges_sources_across_streams():
    a = [Candidate(asin="X", bm25_raw=1.0, sources={"keyword"})]
    b = [Candidate(asin="X", cos_raw=2.0, sources={"semantic"})]
    merged = union_dedupe([a, b])
    assert len(merged) == 1
    assert merged[0].sources == {"keyword", "semantic"}
    assert merged[0].bm25_raw == 1.0
    assert merged[0].cos_raw == 2.0


# --------------------------------------------------------------------------
# floor_check() — A8 relax-and-retry
# --------------------------------------------------------------------------

def test_floor_check_returns_pool_unchanged_when_already_at_floor(indexes):
    pool = [Candidate(asin=f"M{i:03d}") for i in range(POOL_FLOOR)]
    s = SessionState(session_id="s5")
    assert floor_check(pool, s, indexes) == pool


def test_floor_check_tops_up_undersized_pool_to_floor(indexes):
    pool = [Candidate(asin="M000", sources={"keyword"})]
    s = SessionState(session_id="s6")
    result = floor_check(pool, s, indexes)
    assert len(result) >= POOL_FLOOR


def test_floor_check_topped_up_candidates_are_new_not_duplicates(indexes):
    pool = [Candidate(asin="M000", sources={"keyword"})]
    s = SessionState(session_id="s7")
    result = floor_check(pool, s, indexes)
    asins = [c.asin for c in result]
    assert len(asins) == len(set(asins))  # no duplicates introduced


def test_floor_check_empty_pool_and_empty_catalog_returns_placeholder():
    empty_indexes = build_indexes([], embedding_cache_path=None)
    s = SessionState(session_id="s8")
    result = floor_check([], s, empty_indexes)
    assert len(result) == 1
    assert result[0].sources == {"floor_check"}


# --------------------------------------------------------------------------
# retrieve() — full composition, still the stable A2 signature
# --------------------------------------------------------------------------

def test_retrieve_buy_track_never_returns_empty(indexes, state_buy):
    result = retrieve(state_buy, "buy", indexes)
    assert len(result) > 0


def test_retrieve_browse_track_never_returns_empty(indexes, state_browse):
    result = retrieve(state_browse, "browse", indexes)
    assert len(result) > 0


def test_retrieve_unknown_track_falls_back_to_browse_quotas(indexes, state_browse):
    state_browse.track = "not-a-real-track"
    result = retrieve(state_browse, "not-a-real-track", indexes)
    assert len(result) > 0


def test_retrieve_meets_pool_floor_given_enough_catalogue(indexes, state_buy):
    result = retrieve(state_buy, "buy", indexes)
    assert len(result) >= POOL_FLOOR


def test_stream_quotas_cover_buy_and_browse():
    assert set(STREAM_QUOTAS) == {"buy", "browse"}
    for quotas in STREAM_QUOTAS.values():
        assert set(quotas) == {"keyword", "semantic", "popularity"}
