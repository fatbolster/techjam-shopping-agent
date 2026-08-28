"""
Contract tests for evaluate.py's Hit Rate@10/MRR/MTTC/Efficiency scorer
(design doc §6.1, §8.5 step E3).

score_session()/score_transcripts() are tested against fabricated
transcript dicts shaped like simulate.run_session()'s real output, not a
real corpus run (that's what test_owner_b_e2e_audit.py-style integration
coverage and the ablation harness are for). record_baseline() gets one
small end-to-end smoke test against a synthetic catalog/public_set on disk.
"""

import json

import pytest

from evaluate import MAX_TURNS, record_baseline, score_session, score_transcripts


def _session(session_id, scenario_type, turns, converged_turn=None):
    """turns: list of target_rank values, one per turn (None = absent)."""
    transcript = [
        {"turn": i + 1, "ask_attribute": None, "recommended": [], "target_rank": rank}
        for i, rank in enumerate(turns)
    ]
    return {
        "session_id": session_id,
        "scenario_type": scenario_type,
        "target_asin": "X",
        "turns": len(turns),
        "converged_turn": converged_turn,
        "transcript": transcript,
    }


# --------------------------------------------------------------------------
# score_session() — per-session §6.1 formulas
# --------------------------------------------------------------------------

def test_score_session_hit_when_target_in_final_top_10():
    s = score_session(_session("s1", "buying", [5, 3], converged_turn=2))
    assert s.hit is True
    assert s.reciprocal_rank == pytest.approx(1 / 3)


def test_score_session_no_hit_when_target_absent_and_rank_none():
    """target_rank is only ever 1-10 or None in real transcripts — it's
    computed as recommended.index(target)+1 against a top_k=10 list
    (simulate.run_session()), so "beyond top 10" and "absent" are the same
    observable state: None."""
    s = score_session(_session("s1", "buying", [None]))
    assert s.hit is False
    assert s.reciprocal_rank == 0.0


def test_score_session_no_hit_when_target_absent_from_final_turn():
    s = score_session(_session("s1", "buying", [3, None]))
    assert s.hit is False
    assert s.reciprocal_rank == 0.0


def test_score_session_rank_1_scores_full_reciprocal_rank():
    s = score_session(_session("s1", "buying", [1]))
    assert s.reciprocal_rank == pytest.approx(1.0)


def test_score_session_rank_5_scores_point_2():
    s = score_session(_session("s1", "buying", [5]))
    assert s.reciprocal_rank == pytest.approx(0.2)


def test_score_session_turns_to_conversion_uses_converged_turn():
    s = score_session(_session("s1", "buying", [1], converged_turn=1))
    assert s.turns_to_conversion == 1.0


def test_score_session_unconverged_scores_zero_mttc():
    """§6.1: 'Sessions exceeding 10 turns terminate with zero score.'"""
    s = score_session(_session("s1", "buying", [None] * 10, converged_turn=None))
    assert s.turns_to_conversion == 0.0


def test_score_session_converged_turn_beyond_max_turns_scores_zero():
    s = score_session(_session("s1", "buying", [1], converged_turn=MAX_TURNS + 1))
    assert s.turns_to_conversion == 0.0


def test_score_session_empty_transcript_no_hit_no_crash():
    s = score_session({"session_id": "s1", "scenario_type": "buying", "transcript": []})
    assert s.hit is False
    assert s.reciprocal_rank == 0.0


def test_score_session_default_efficiency_is_zero():
    s = score_session(_session("s1", "buying", [1]))
    assert s.efficiency == 0.0


# --------------------------------------------------------------------------
# score_transcripts() — aggregate + by_scenario (§6.1, §6.4)
# --------------------------------------------------------------------------

def test_score_transcripts_aggregate_hit_rate_is_mean_across_sessions():
    transcripts = [
        _session("s1", "buying", [1]),  # hit
        _session("s2", "buying", [20]),  # miss
    ]
    result = score_transcripts(transcripts)
    assert result["overall"].hit_rate_at_10 == pytest.approx(0.5)
    assert result["overall"].n == 2


def test_score_transcripts_aggregate_mrr_is_mean_reciprocal_rank():
    transcripts = [_session("s1", "buying", [1]), _session("s2", "buying", [4])]
    result = score_transcripts(transcripts)
    assert result["overall"].mrr == pytest.approx((1.0 + 0.25) / 2)


def test_score_transcripts_slices_by_scenario_type():
    transcripts = [
        _session("s1", "buying", [1]),
        _session("s2", "browsing", [20]),
        _session("s3", "buying", [1]),
    ]
    result = score_transcripts(transcripts)
    assert set(result["by_scenario"]) == {"buying", "browsing"}
    assert result["by_scenario"]["buying"].n == 2
    assert result["by_scenario"]["buying"].hit_rate_at_10 == pytest.approx(1.0)
    assert result["by_scenario"]["browsing"].n == 1
    assert result["by_scenario"]["browsing"].hit_rate_at_10 == pytest.approx(0.0)


def test_score_transcripts_empty_list_returns_zeroed_aggregate():
    result = score_transcripts([])
    assert result["overall"].n == 0
    assert result["overall"].hit_rate_at_10 == 0.0
    assert result["by_scenario"] == {}


def test_score_transcripts_overall_covers_all_sessions_regardless_of_scenario():
    transcripts = [_session("s1", "buying", [1]), _session("s2", "browsing", [1])]
    result = score_transcripts(transcripts)
    assert result["overall"].n == 2


# --------------------------------------------------------------------------
# record_baseline() — small end-to-end smoke test
# --------------------------------------------------------------------------

def test_record_baseline_end_to_end_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
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
        }
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

    result = record_baseline()
    assert result["baseline"]["overall"].n == 1
    assert "recall" in result
