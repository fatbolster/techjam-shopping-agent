"""Tests for Owner D's telemetry + training-corpus assembly (§8.4 D2, D6, D7)."""

from __future__ import annotations

import json

import pytest

import telemetry
from features import FEATURE_NAMES
from indexes import build_indexes
from state import init_state
from utils import Candidate, FIXTURE_CATALOG

FIXTURE_ASINS = [row["parent_asin"] for row in FIXTURE_CATALOG]


# --------------------------------------------------------------------------
# n_hard_slots
# --------------------------------------------------------------------------
def test_n_hard_slots_counts_only_filter_authorising_slots() -> None:
    state = init_state("s")
    assert telemetry.n_hard_slots(state) == 0
    state.slots = {"category": "Running", "color": ("black",), "brand": "Nike"}
    assert telemetry.n_hard_slots(state) == 2  # category + brand, not color
    assert telemetry.n_hard_slots(None) == 0


# --------------------------------------------------------------------------
# D2 — log_turn
# --------------------------------------------------------------------------
def test_log_turn_writes_one_row_with_features_and_sources(tmp_path) -> None:
    path = str(tmp_path / "telem.jsonl")
    indexes = build_indexes(FIXTURE_CATALOG, embedding_cache_path=None)
    state = init_state("sess-1")
    state.slots = {"category": "Running"}
    state.track = "buy"
    state.canonical_intent = "category: running"
    pool = [
        Candidate(asin=FIXTURE_ASINS[0], bm25_raw=2.0, sources={"keyword"}),
        Candidate(asin=FIXTURE_ASINS[1], cos_raw=0.4, sources={"semantic", "popularity"}),
    ]

    telemetry.log_turn("sess-1", 3, pool, state=state, indexes=indexes, ask_attribute="color", path=path)

    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == 1
    row = rows[0]
    assert row["session_id"] == "sess-1" and row["turn"] == 3
    assert row["track"] == "buy" and row["n_hard_slots"] == 1
    assert row["ask_attribute"] == "color" and row["pool_size"] == 2
    assert row["feature_names"] == list(FEATURE_NAMES)
    assert row["candidates"][0]["sources"] == ["keyword"]
    assert row["candidates"][1]["sources"] == ["popularity", "semantic"]
    assert len(row["candidates"][0]["features"]) == len(FEATURE_NAMES)


def test_log_turn_appends(tmp_path) -> None:
    path = str(tmp_path / "t.jsonl")
    for turn in (1, 2):
        telemetry.log_turn("s", turn, [Candidate(asin="A")], path=path)
    assert len(open(path).read().splitlines()) == 2


def test_telemetry_path_ctx_redirects_default_output(tmp_path) -> None:
    redirected = str(tmp_path / "redirected.jsonl")
    with telemetry.telemetry_path_ctx(redirected):
        telemetry.log_turn("s", 1, [Candidate(asin="A")])  # no explicit path
    assert json.loads(open(redirected).read())["session_id"] == "s"


# --------------------------------------------------------------------------
# D6 — training rows
# --------------------------------------------------------------------------
def _telemetry_row(session_id: str, turn: int, asins: list[str], n_hard: int = 1) -> dict:
    return {
        "session_id": session_id,
        "turn": turn,
        "n_hard_slots": n_hard,
        "candidates": [
            {"asin": a, "sources": ["keyword"], "features": [float(i)] * len(FEATURE_NAMES)}
            for i, a in enumerate(asins)
        ],
    }


def test_build_training_rows_schema_labels_and_pool_negative_cap() -> None:
    pool = [f"A{i}" for i in range(40)]
    rows = telemetry.build_training_rows(
        [_telemetry_row("s1", 1, pool)],
        ground_truth={"s1": "A5"},
        negatives_per_turn=20,
        session_meta={"s1": {"scenario_type": "buying"}},
    )
    positives = [r for r in rows if r["label"] == 1]
    negatives = [r for r in rows if r["label"] == 0]
    assert len(positives) == 1 and positives[0]["parent_asin"] == "A5"
    assert len(negatives) == 20
    assert all(r["parent_asin"] != "A5" for r in negatives)
    assert set(rows[0]) == {
        "session_id", "turn", "n_hard_slots", "parent_asin", "features", "label", "scenario_type"
    }
    assert len(rows[0]["features"]) == len(FEATURE_NAMES)


def test_build_training_rows_no_positive_when_target_absent_from_pool() -> None:
    rows = telemetry.build_training_rows(
        [_telemetry_row("s1", 1, ["A0", "A1", "A2"])], ground_truth={"s1": "NOT_IN_POOL"}
    )
    assert all(r["label"] == 0 for r in rows)
    assert len(rows) == 3


def test_build_training_rows_is_one_row_set_per_turn() -> None:
    telem = [
        _telemetry_row("s1", 1, ["A0", "A1", "A2"], n_hard=1),
        _telemetry_row("s1", 2, ["A0", "A1", "A2"], n_hard=3),
    ]
    rows = telemetry.build_training_rows(telem, ground_truth={"s1": "A0"})
    assert {r["turn"] for r in rows} == {1, 2}
    assert {r["n_hard_slots"] for r in rows if r["turn"] == 2} == {3}


def test_sample_pool_negatives_excludes_target_and_is_reproducible() -> None:
    cands = [{"asin": f"A{i}"} for i in range(30)]
    a = telemetry.sample_pool_negatives(cands, "A0", telemetry._seeded_rng("s", 1), 10)
    b = telemetry.sample_pool_negatives(cands, "A0", telemetry._seeded_rng("s", 1), 10)
    assert [c["asin"] for c in a] == [c["asin"] for c in b]
    assert all(c["asin"] != "A0" for c in a) and len(a) == 10


# --------------------------------------------------------------------------
# D7 — per-stream recall
# --------------------------------------------------------------------------
def test_per_stream_recall_report_counts_presence_unique_and_ceiling() -> None:
    telem = [
        {"session_id": "s1", "turn": 1, "candidates": [
            {"asin": "T", "sources": ["keyword", "semantic"]},
            {"asin": "X", "sources": ["popularity"]},
        ]},
        {"session_id": "s2", "turn": 1, "candidates": [
            {"asin": "Y", "sources": ["semantic"]},  # target absent
        ]},
        {"session_id": "s3", "turn": 1, "candidates": [
            {"asin": "T3", "sources": ["popularity"]},
        ]},
    ]
    gt = {"s1": "T", "s2": "T2", "s3": "T3"}
    meta = {"s1": {"scenario_type": "buying"}, "s2": {"scenario_type": "browsing"},
            "s3": {"scenario_type": "buying"}}
    report = telemetry.per_stream_recall_report(telem, gt, meta)

    overall = report["overall"]
    assert overall["turns"] == 3 and overall["turns_with_target"] == 2
    assert overall["pool_recall"] == pytest.approx(2 / 3)
    assert overall["present"]["keyword"] == 1
    assert overall["present"]["semantic"] == 1
    assert overall["present"]["popularity"] == 1
    assert overall["unique"]["popularity"] == 1  # only s3's target was popularity-only
    assert overall["unique"]["keyword"] == 0
    assert report["by_scenario"]["buying"]["turns_with_target"] == 2


# --------------------------------------------------------------------------
# write_feature_matrix + end-to-end
# --------------------------------------------------------------------------
def test_write_feature_matrix_overwrites(tmp_path) -> None:
    path = str(tmp_path / "features.jsonl")
    telemetry.write_feature_matrix([{"a": 1}, {"a": 2}], path)
    telemetry.write_feature_matrix([{"a": 3}], path)
    rows = [json.loads(line) for line in open(path)]
    assert rows == [{"a": 3}]


def test_run_instrumented_corpus_end_to_end(tmp_path) -> None:
    sessions = []
    for i, row in enumerate(FIXTURE_CATALOG):
        for scenario in ("buying", "browsing", "intent_override", "boundary"):
            sessions.append({
                "sample_id": f"pub_{i}_{scenario}",
                "scenario_type": scenario,
                "ground_truth": {"parent_asin": row["parent_asin"]},
                "user_profile": {"preference_tags": ["warmth"]},
            })
    public_set = tmp_path / "public_set.jsonl"
    public_set.write_text("\n".join(json.dumps(s) for s in sessions), encoding="utf-8")
    features_path = tmp_path / "features.jsonl"

    summary = telemetry.run_instrumented_corpus(
        catalog_path=str(tmp_path / "no-catalog.jsonl"),
        public_set_path=str(public_set),
        telemetry_path=str(tmp_path / "telem.jsonl"),
        feature_matrix_path=str(features_path),
        # transcript_path defaults to "data/transcripts.txt" — without this
        # override every run of this test silently overwrites the real D8
        # export with these 6 fixture-catalog sessions (found by diffing a
        # real run_instrumented_corpus() run's transcripts.txt against a
        # suite run that happened to interleave with it).
        transcript_path=str(tmp_path / "transcripts.txt"),
        max_turns=6,
    )

    assert summary["sessions"] == len(sessions)
    assert summary["turns"] > 0 and summary["feature_rows"] > 0
    assert set(summary["recall"]["by_scenario"]) == {"buying", "browsing", "intent_override", "boundary"}

    rows = [json.loads(line) for line in open(features_path)]
    assert rows and all(len(r["features"]) == len(FEATURE_NAMES) for r in rows)
    assert any(r["label"] == 1 for r in rows)
    # every intent_override session contributed at least one post-contradiction turn
    override_turns = {
        r["session_id"] for r in rows if r["session_id"].endswith("intent_override")
    }
    assert len(override_turns) == 3


# --------------------------------------------------------------------------
# D7 formatter / D8 transcript export
# --------------------------------------------------------------------------
def test_format_recall_report_mentions_ceiling_and_streams() -> None:
    report = telemetry.per_stream_recall_report(
        [{"session_id": "s1", "turn": 1, "candidates": [
            {"asin": "T", "sources": ["keyword"]}]}],
        {"s1": "T"},
        {"s1": {"scenario_type": "buying"}},
    )
    text = telemetry.format_recall_report(report)
    assert "pool recall" in text
    assert "keyword" in text and "semantic" in text and "popularity" in text
    assert "buying" in text


def test_export_transcripts_only_writes_named_scenarios(tmp_path) -> None:
    results = [
        {"session_id": "a", "scenario_type": "intent_override", "target_asin": "T",
         "turns": 2, "converged_turn": None, "transcript": [
             {"turn": 1, "user": "I want a blue jacket", "ask_attribute": None,
              "recommended": ["T", "X"], "target_rank": 1},
             {"turn": 2, "user": "not blue, red instead", "ask_attribute": "color",
              "recommended": ["X", "T"], "target_rank": 2},
         ]},
        {"session_id": "b", "scenario_type": "boundary", "target_asin": "Q",
         "turns": 1, "converged_turn": 1, "transcript": [
             {"turn": 1, "user": "just browsing", "ask_attribute": None,
              "recommended": ["Q"], "target_rank": 1}]},
        {"session_id": "c", "scenario_type": "buying", "target_asin": "Z",
         "turns": 1, "converged_turn": 1, "transcript": []},
    ]
    path = tmp_path / "transcripts.txt"
    written = telemetry.export_transcripts(results, str(path))
    body = path.read_text()

    assert written == 2
    assert "session_id" not in body  # it's a plain transcript, not JSON
    assert "not blue, red instead" in body
    assert "agent asked -> color" in body
    assert "[buying]" not in body  # buying session excluded
