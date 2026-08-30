"""Focused tests for the supplemental labels-free keyword source."""

import retrieval
from retrieval import clean_keyword_query, clean_keyword_stream, union_dedupe
from state import SessionState
from utils import Candidate, FIXTURE_CATALOG
from indexes import build_facts_dict, build_fts5_index, Indexes


def test_clean_keyword_experiment_defaults_off() -> None:
    assert retrieval.CLEAN_KEYWORD_ENABLED is False


def test_clean_keyword_query_contains_scalar_values_without_labels() -> None:
    state = SessionState(
        session_id="clean-query",
        slots={"category": "Running", "color": ("blue",)},
    )

    query = clean_keyword_query(state)

    assert "Running" in query
    assert "blue" in query
    assert "category" not in query.casefold()
    assert "color" not in query.casefold()


def test_clean_keyword_query_contains_feature_without_label() -> None:
    state = SessionState(
        session_id="clean-query",
        slots={"feature": ("waterproof",)},
    )

    query = clean_keyword_query(state)

    assert query == "waterproof"
    assert "features" not in query.casefold()


def test_clean_keyword_query_retains_safe_scenario_text() -> None:
    state = SessionState(
        session_id="clean-query",
        scenario_buffer="for a beach holiday, but I'm still exploring",
    )

    query = clean_keyword_query(state)

    assert "beach holiday" in query
    assert "still exploring" not in query.casefold()


def test_clean_keyword_query_preserves_all_active_multi_values() -> None:
    state = SessionState(
        session_id="clean-query",
        slots={
            "material": ("cotton", "linen"),
            "feature": ("waterproof", "zipper closure"),
        },
    )

    assert clean_keyword_query(state) == "cotton linen waterproof zipper closure"


def test_clean_keyword_query_uses_only_current_overwritten_values() -> None:
    state = SessionState(
        session_id="clean-query",
        slots={"color": ("blue",)},
        slot_override_flags={"color": True},
        override_reference_values={"color": ("red",)},
    )

    query = clean_keyword_query(state)

    assert "blue" in query
    assert "red" not in query


def test_clean_keyword_query_ignores_session_metadata() -> None:
    state = SessionState(
        session_id="Key-NOT-SOLE",
        slots={"category": "Running"},
        profile_terms=["Style"],
        rating_style="NOT",
        asked_attributes={"brand"},
        pending_clarification="style",
    )

    assert clean_keyword_query(state) == "Running"


def test_clean_keyword_query_empty_state_is_empty() -> None:
    assert clean_keyword_query(SessionState(session_id="empty")) == ""


def test_clean_keyword_query_removes_dialogue_boilerplate_from_scenario() -> None:
    state = SessionState(
        session_id="clean-query",
        scenario_buffer=(
            "A key requirement is beach wear. For that, what matters is comfort. "
            "Those options are not quite right."
        ),
    )

    query = clean_keyword_query(state)

    lowered = query.casefold()
    assert "key requirement" not in lowered
    assert "for that, what matters is" not in lowered
    assert "not quite right" not in lowered
    assert "beach wear" in lowered
    assert "comfort" in lowered


def test_clean_keyword_query_does_not_restore_historical_false_values() -> None:
    state = SessionState(
        session_id="clean-query",
        slots={"category": "Running"},
        canonical_intent="brand: Style; brand: Key; brand: NOT; brand: SOLE",
        override_reference_values={"brand": ("Style", "Key", "NOT", "SOLE")},
    )

    query = clean_keyword_query(state)

    assert query == "Running"


def test_clean_keyword_query_is_deterministic() -> None:
    state = SessionState(
        session_id="clean-query",
        slots={
            "feature": ("waterproof",),
            "category": "Running",
            "color": ("blue",),
        },
        scenario_buffer="for a beach holiday",
    )

    assert clean_keyword_query(state) == clean_keyword_query(state)
    assert clean_keyword_query(state) == "Running blue waterproof for a beach holiday"


def test_clean_keyword_query_renders_prices_without_price_labels() -> None:
    state = SessionState(
        session_id="clean-query",
        slots={"price_min": "50", "price_max": "100", "price_target": "80"},
    )

    query = clean_keyword_query(state)

    assert query == "$50 $100 $80"
    assert "price" not in query.casefold()
    assert "minimum" not in query.casefold()
    assert "maximum" not in query.casefold()


def _keyword_only_indexes() -> Indexes:
    return Indexes(
        catalog=FIXTURE_CATALOG,
        fts_conn=build_fts5_index(FIXTURE_CATALOG),
        embedding_matrix=None,
        embedding_asins=[],
        facts=build_facts_dict(FIXTURE_CATALOG),
        category_lists={},
    )


def test_clean_keyword_stream_marks_its_source() -> None:
    state = SessionState(session_id="clean-query", slots={"category": "Running"})

    results = clean_keyword_stream(state, _keyword_only_indexes(), quota=10)

    assert results
    assert all(candidate.sources == {"keyword_clean"} for candidate in results)


def test_union_preserves_canonical_candidate_and_merges_clean_source() -> None:
    canonical = [Candidate(asin="shared", bm25_raw=4.0, sources={"keyword"})]
    clean = [
        Candidate(asin="shared", bm25_raw=8.0, sources={"keyword_clean"}),
        Candidate(asin="added", bm25_raw=3.0, sources={"keyword_clean"}),
    ]

    result = union_dedupe([canonical, clean])

    assert [candidate.asin for candidate in result] == ["shared", "added"]
    assert result[0].sources == {"keyword", "keyword_clean"}
    assert result[0].bm25_raw == 4.0
