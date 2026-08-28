"""Tests for scripts/report_ranker.py (design doc §8.3 step C8)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import report_ranker  # noqa: E402

from features import FEATURE_NAMES
from rank import FittedRanker, save_fitted_ranker

pytest.importorskip("pandas")


def _write_features(path, rows):
    """rows: list of (features_dict, label)."""
    with open(path, "w") as f:
        for feats, label in rows:
            row = {
                "session_id": "s1",
                "turn": 1,
                "n_hard_slots": 1,
                "parent_asin": "X",
                "features": [feats.get(name, 0.0) for name in FEATURE_NAMES],
                "label": label,
            }
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def synthetic_features(tmp_path):
    rows = [({name: 0.5 for name in FEATURE_NAMES}, 1), ({name: 0.1 for name in FEATURE_NAMES}, 0)] * 20
    path = tmp_path / "features.jsonl"
    _write_features(path, rows)
    return path


def test_load_feature_dataframe_has_one_row_per_candidate_and_a_label_column(synthetic_features):
    df = report_ranker.load_feature_dataframe(str(synthetic_features))
    assert len(df) == 40
    assert set(df.columns) == {*FEATURE_NAMES, "label"}


def test_weight_report_flags_near_zero_relative_to_max(tmp_path):
    ranker = FittedRanker(weights={**dict.fromkeys(FEATURE_NAMES, 0.0), "pop": 10.0, "rating": 0.1})
    path = tmp_path / "ranker.json"
    save_fitted_ranker(ranker, str(path))

    df = report_ranker.weight_report(str(path))
    pop_row = df[df["feature"] == "pop"].iloc[0]
    rating_row = df[df["feature"] == "rating"].iloc[0]
    assert pop_row["near_zero"] == False  # noqa: E712 (pandas bool, not Python bool)
    assert rating_row["near_zero"] == True  # noqa: E712 — 0.1 is < 5% of 10.0


def test_weight_report_sorted_by_magnitude_descending(tmp_path):
    ranker = FittedRanker(weights={**dict.fromkeys(FEATURE_NAMES, 0.0), "pop": -10.0, "rating": 2.0})
    path = tmp_path / "ranker.json"
    save_fitted_ranker(ranker, str(path))
    df = report_ranker.weight_report(str(path))
    assert df.iloc[0]["feature"] == "pop"  # |-10| > |2|


def test_correlation_report_is_symmetric_with_unit_diagonal(synthetic_features):
    corr = report_ranker.correlation_report(str(synthetic_features))
    assert list(corr.index) == list(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert corr.loc[name, name] == pytest.approx(1.0)


def test_cross_check_doc_predictions_reports_both_attributes(tmp_path):
    ranker = FittedRanker(weights={**dict.fromkeys(FEATURE_NAMES, 0.0), "rare_tag_match": 0.5, "rating_style_fit": 0.01})
    path = tmp_path / "ranker.json"
    save_fitted_ranker(ranker, str(path))
    weights = report_ranker.weight_report(str(path))
    lines = report_ranker.cross_check_doc_predictions(weights)
    assert any("rare_tag_match" in line and "as predicted" in line for line in lines)
    assert any("rating_style_fit" in line for line in lines)


def test_main_runs_end_to_end_without_raising(tmp_path, synthetic_features, capsys):
    ranker = FittedRanker(weights={**dict.fromkeys(FEATURE_NAMES, 0.1), "pop": 5.0})
    ranker_path = tmp_path / "ranker.json"
    save_fitted_ranker(ranker, str(ranker_path))

    report_ranker.main(str(synthetic_features), str(ranker_path))
    out = capsys.readouterr().out
    assert "Fitted weights" in out
    assert "Pairwise feature correlations" in out
