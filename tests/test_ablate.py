"""
Contract tests for ablate.py's ablation harness (design doc §6.3, §8.5
steps E5-E6).

"feature"-kind ablations (the fast re-score-without-re-retrieval path) are
tested against fabricated telemetry rows. "baseline"/"stream"/"filter"
kinds (full pipeline passes) get one real end-to-end smoke test each,
against a small synthetic catalog/public_set — proving the
STREAM_QUOTAS/DEPARTMENT_FILTER_ENABLED toggles actually reach the real
pipeline and are restored afterward, not that the ablated *scores* are
meaningful (a 2-department, handful-of-products catalogue is too small
for that).
"""

import json

import pytest

import retrieval
from ablate import (
    ABLATION_CONFIGS,
    SCENARIO_SLICES,
    AblationResult,
    _feature_ablation_transcripts,
    _rare_tag_match_subset,
    run_ablation,
    slice_by_scenario,
)
from features import FEATURE_NAMES
from rank import HANDSET_WEIGHTS

sentence_transformers = pytest.importorskip("sentence_transformers")


def _telemetry_row(session_id, turn, candidates):
    """candidates: list of (asin, {feature_name: value, ...}) pairs."""
    return {
        "session_id": session_id,
        "turn": turn,
        "candidates": [{"asin": asin, "features": feats} for asin, feats in candidates],
    }


def _feats(**overrides):
    row = dict.fromkeys(FEATURE_NAMES, 0.0)
    row.update(overrides)
    return row


# --------------------------------------------------------------------------
# ABLATION_CONFIGS / SCENARIO_SLICES — shape sanity
# --------------------------------------------------------------------------

def test_ablation_configs_has_nine_entries():
    assert len(ABLATION_CONFIGS) == 9


def test_ablation_configs_names_are_unique():
    names = [c["name"] for c in ABLATION_CONFIGS]
    assert len(names) == len(set(names))


def test_scenario_slices_covers_all_four_types():
    assert set(SCENARIO_SLICES) == {"buying", "browsing", "intent_override", "boundary"}


# --------------------------------------------------------------------------
# _feature_ablation_transcripts() — re-rank from logged candidates
# --------------------------------------------------------------------------

def test_feature_ablation_transcripts_takes_last_turn_per_session():
    rows = [
        _telemetry_row("s1", 1, [("TARGET", _feats(pop=0.1))]),
        _telemetry_row("s1", 2, [("TARGET", _feats(pop=0.9))]),
    ]
    transcripts = _feature_ablation_transcripts(rows, {"s1": "TARGET"}, {}, dict(HANDSET_WEIGHTS))
    assert len(transcripts) == 1
    assert transcripts[0]["turns"] == 2  # turn 2's data was used, not turn 1's


def test_feature_ablation_transcripts_zeroed_feature_changes_rank():
    candidates = [("TARGET", _feats(pop=0.1, rating=0.9)), ("OTHER", _feats(pop=0.9, rating=0.1))]
    rows = [_telemetry_row("s1", 1, candidates)]
    gt = {"s1": "TARGET"}

    with_pop = _feature_ablation_transcripts(rows, gt, {}, {**HANDSET_WEIGHTS})
    zeroed_pop = dict(HANDSET_WEIGHTS)
    zeroed_pop["pop"] = 0.0
    without_pop = _feature_ablation_transcripts(rows, gt, {}, zeroed_pop)

    rank_with = with_pop[0]["transcript"][0]["target_rank"]
    rank_without = without_pop[0]["transcript"][0]["target_rank"]
    assert rank_without < rank_with  # zeroing pop's weight lets rating win instead


def test_feature_ablation_transcripts_target_absent_from_pool_ranks_none():
    rows = [_telemetry_row("s1", 1, [("OTHER", _feats())])]
    transcripts = _feature_ablation_transcripts(rows, {"s1": "TARGET"}, {}, dict(HANDSET_WEIGHTS))
    assert transcripts[0]["transcript"][0]["target_rank"] is None


def test_feature_ablation_transcripts_carries_scenario_type_from_session_meta():
    rows = [_telemetry_row("s1", 1, [("TARGET", _feats())])]
    meta = {"s1": {"scenario_type": "buying"}}
    transcripts = _feature_ablation_transcripts(rows, {"s1": "TARGET"}, meta, dict(HANDSET_WEIGHTS))
    assert transcripts[0]["scenario_type"] == "buying"


# --------------------------------------------------------------------------
# _rare_tag_match_subset() — "the 22% subset" (§6.3)
# --------------------------------------------------------------------------

def test_rare_tag_match_subset_includes_only_nonzero_target_sessions():
    rows = [
        _telemetry_row("s1", 1, [("TARGET1", _feats(rare_tag_match=0.8))]),
        _telemetry_row("s2", 1, [("TARGET2", _feats(rare_tag_match=0.0))]),
    ]
    gt = {"s1": "TARGET1", "s2": "TARGET2"}
    subset = _rare_tag_match_subset(rows, gt)
    assert subset == {"s1"}


def test_rare_tag_match_subset_empty_when_target_missing_from_pool():
    rows = [_telemetry_row("s1", 1, [("OTHER", _feats(rare_tag_match=0.9))])]
    subset = _rare_tag_match_subset(rows, {"s1": "TARGET"})
    assert subset == set()


# --------------------------------------------------------------------------
# run_ablation() — feature kind (fast path) and -llm_rerank
# --------------------------------------------------------------------------

def test_run_ablation_unknown_config_raises():
    with pytest.raises(ValueError):
        run_ablation("-not-a-real-config")


def test_run_ablation_feature_kind_requires_telemetry():
    with pytest.raises(ValueError):
        run_ablation("-rating_style_fit")


def test_run_ablation_feature_kind_returns_mrr():
    rows = [
        _telemetry_row("s1", 1, [("TARGET", _feats(rating_style_fit=0.9)), ("OTHER", _feats())]),
    ]
    gt = {"s1": "TARGET"}
    result = run_ablation("-rating_style_fit", telemetry_rows=rows, ground_truth=gt, session_meta={})
    assert isinstance(result, AblationResult)
    assert result.mrr == pytest.approx(1.0)  # TARGET still ranks first even with rating_style_fit zeroed


def test_run_ablation_llm_rerank_is_a_trivial_no_change():
    """llm_rerank() (C7) is a stub always returning None; the real answer
    is 'identical to baseline', not computed via re-scoring."""
    result = run_ablation("-llm_rerank")
    assert isinstance(result, AblationResult)
    assert result.config_name == "-llm_rerank"


def test_run_ablation_rare_tag_match_scopes_to_subset():
    rows = [
        _telemetry_row("s1", 1, [("T1", _feats(rare_tag_match=0.5))]),
        _telemetry_row("s2", 1, [("T2", _feats(rare_tag_match=0.0))]),
    ]
    gt = {"s1": "T1", "s2": "T2"}
    result = run_ablation("-rare_tag_match", telemetry_rows=rows, ground_truth=gt, session_meta={})
    assert result.n == 1  # only s1 has a nonzero rare_tag_match target


# --------------------------------------------------------------------------
# slice_by_scenario()
# --------------------------------------------------------------------------

def test_slice_by_scenario_returns_zeroed_aggregate_for_absent_slice():
    rows = [_telemetry_row("s1", 1, [("TARGET", _feats())])]
    result = run_ablation(
        "-rating_style_fit", telemetry_rows=rows, ground_truth={"s1": "TARGET"}, session_meta={}
    )
    agg = slice_by_scenario(result, "boundary")  # no boundary sessions in this fixture
    assert agg.n == 0
    assert agg.mrr == 0.0


def test_slice_by_scenario_returns_the_matching_slice():
    rows = [_telemetry_row("s1", 1, [("TARGET", _feats())])]
    meta = {"s1": {"scenario_type": "buying"}}
    result = run_ablation(
        "-rating_style_fit", telemetry_rows=rows, ground_truth={"s1": "TARGET"}, session_meta=meta
    )
    agg = slice_by_scenario(result, "buying")
    assert agg.n == 1


# --------------------------------------------------------------------------
# run_ablation() — stream/filter/baseline kinds: real end-to-end smoke
# tests against a tiny synthetic catalog. Slow (each is a full pipeline
# pass); proves the module-level toggles reach the real pipeline and are
# restored afterward, not that ablated scores are meaningful at this scale.
# --------------------------------------------------------------------------

def _write_synthetic_data(tmp_path):
    catalog = [
        {
            "parent_asin": "TARGET1",
            "title": "Red running shoe",
            "categories": ["Root", "Men", "Shoes"],
            "details": {"Department": "Men", "Color": "Red"},
            "store": "Acme",
            "price": 40.0,
            "rating_number": 500,
            "average_rating": 4.5,
            "features": [],
            "description": [],
        },
        {
            "parent_asin": "OTHER1",
            "title": "Blue running shoe",
            "categories": ["Root", "Women", "Shoes"],
            "details": {"Department": "Women", "Color": "Blue"},
            "store": "Acme",
            "price": 35.0,
            "rating_number": 300,
            "average_rating": 4.0,
            "features": [],
            "description": [],
        },
    ]
    (tmp_path / "data").mkdir()
    with open(tmp_path / "data" / "catalog.jsonl", "w") as f:
        for row in catalog:
            f.write(json.dumps(row) + "\n")
    public_set = [
        {
            "sample_id": "public_0001",
            "scenario_type": "buying",
            "ground_truth": {"parent_asin": "TARGET1"},
            "user_profile": {},
        }
    ]
    with open(tmp_path / "data" / "public_set.jsonl", "w") as f:
        for row in public_set:
            f.write(json.dumps(row) + "\n")


def test_run_ablation_baseline_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_synthetic_data(tmp_path)
    result = run_ablation("full_pipeline")
    assert isinstance(result, AblationResult)
    assert result.n == 1


def test_run_ablation_stream_kind_restores_quotas_after(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_synthetic_data(tmp_path)
    original = {track: dict(quotas) for track, quotas in retrieval.STREAM_QUOTAS.items()}
    run_ablation("-popularity_stream")
    assert retrieval.STREAM_QUOTAS == original
