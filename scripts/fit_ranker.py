"""Fit the real logistic regression ranker on data/features.jsonl (C5).

Design doc §8.3 step C5. Run after data/features.jsonl exists (produced by
telemetry.run_instrumented_corpus() / evaluate.record_baseline()):

    python3 scripts/fit_ranker.py

Reads every row of data/features.jsonl (telemetry.build_training_rows()'s
schema: session_id, turn, n_hard_slots, parent_asin, features, label,
scenario_type), adapts it via rank.rows_to_training_arrays(), fits with
rank.fit_logistic_regression() (real sklearn, GroupKFold by session_id),
and persists the result to models/ranker.json (rank.save_fitted_ranker()).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank import DEFAULT_RANKER_PATH, fit_logistic_regression, rows_to_training_arrays, save_fitted_ranker


def load_feature_rows(path: str = "data/features.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(feature_matrix_path: str = "data/features.jsonl", ranker_path: str = DEFAULT_RANKER_PATH) -> None:
    rows = load_feature_rows(feature_matrix_path)
    n_positive = sum(r["label"] for r in rows)
    n_sessions = len({r["session_id"] for r in rows})
    print(f"Loaded {len(rows)} rows, {n_positive} positive, {n_sessions} sessions.")

    feature_matrix, labels, groups = rows_to_training_arrays(rows)
    ranker = fit_logistic_regression(feature_matrix, labels, groups)
    save_fitted_ranker(ranker, ranker_path)

    print(f"Fitted and saved to {ranker_path}.")
    print(f"cv_accuracy (GroupKFold by session_id): {ranker.cv_accuracy}")
    print("weights:")
    for name, weight in sorted(ranker.weights.items(), key=lambda p: -abs(p[1])):
        print(f"  {name:20s} {weight:+.4f}")
    print(f"intercept: {ranker.intercept:+.4f}")


if __name__ == "__main__":
    main()
