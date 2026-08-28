"""Tests for scripts/fit_ranker.py (design doc §8.3 step C5's execution)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fit_ranker  # noqa: E402

from features import FEATURE_NAMES
from rank import FittedRanker, load_fitted_ranker

sklearn = pytest.importorskip("sklearn")


def _write_rows(path, n_sessions=8, rows_per_session=4):
    """Synthetic rows where `pop` alone separates target from non-target,
    same construction as tests/test_rank.py's fit_logistic_regression
    fixtures — enough sessions for GroupKFold, real (if trivial) signal."""
    import random

    rng = random.Random(0)
    with open(path, "w") as f:
        for s in range(n_sessions):
            for r in range(rows_per_session):
                is_target = r == 0
                feats = {name: rng.uniform(0.0, 1.0) for name in FEATURE_NAMES}
                feats["pop"] = rng.uniform(0.8, 1.0) if is_target else rng.uniform(0.0, 0.2)
                row = {
                    "session_id": f"session-{s}",
                    "turn": 1,
                    "n_hard_slots": 1,
                    "parent_asin": f"asin-{s}-{r}",
                    "features": [feats[name] for name in FEATURE_NAMES],
                    "label": 1 if is_target else 0,
                    "scenario_type": "buying",
                }
                f.write(json.dumps(row) + "\n")


def test_load_feature_rows_reads_jsonl(tmp_path):
    path = tmp_path / "features.jsonl"
    _write_rows(path)
    rows = fit_ranker.load_feature_rows(str(path))
    assert len(rows) == 32  # 8 sessions * 4 rows


def test_main_fits_and_persists_a_real_ranker(tmp_path):
    features_path = tmp_path / "features.jsonl"
    ranker_path = tmp_path / "ranker.json"
    _write_rows(features_path)

    fit_ranker.main(str(features_path), str(ranker_path))

    ranker = load_fitted_ranker(str(ranker_path))
    assert isinstance(ranker, FittedRanker)
    assert set(ranker.weights) == set(FEATURE_NAMES)
    assert ranker.weights["pop"] > 0  # recovers the synthetic signal
    assert ranker.cv_accuracy is not None
