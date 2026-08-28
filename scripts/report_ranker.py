"""Report fitted feature weights and pairwise correlations (C8).

Design doc §8.3 step C8: "Once C5 delivers a real FittedRanker, write a
small notebook/script reporting each feature's fitted coefficient and
pairwise correlations between features. Flag any near-zero-weight feature
explicitly — per §6.3, a near-zero weight is itself evidence a feature is
inert, without needing a full ablation re-run. Cross-check against the
doc's own predictions (§2.4/§2.4.1): expect rare_tag_match and
rating_style_fit to have small-but-nonzero weights, not necessarily zero."

Run after scripts/fit_ranker.py (needs both models/ranker.json and
data/features.jsonl):

    python3 scripts/report_ranker.py

Kept as a plain script rather than a Jupyter notebook (§8.3's "Technology:
Jupyter, pandas" is a suggestion, not a hard requirement) — every number it
prints is reproducible by re-running it, which a notebook's saved output
cells are not automatically; a notebook that imports and calls this
module's functions is a straightforward follow-up if the team wants the
inline-plot / cell-by-cell format for the writeup instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from features import FEATURE_NAMES
from rank import DEFAULT_RANKER_PATH, load_fitted_ranker

# §6.3: "a near-zero weight is itself evidence a feature is inert, without
# needing a full ablation re-run." No exact threshold is given in the doc;
# this is this script's judgment call, relative to the largest-magnitude
# weight rather than an absolute cutoff, so it stays meaningful regardless
# of the overall scale a given fit lands on (StandardScaler-fitted
# coefficients don't have an inherent absolute unit to compare against).
NEAR_ZERO_RELATIVE_THRESHOLD = 0.05


def load_feature_dataframe(path: str = "data/features.jsonl") -> pd.DataFrame:
    """One row per (session, turn, candidate), one column per feature."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(dict(zip(FEATURE_NAMES, row["features"]), label=row["label"]))
    return pd.DataFrame(rows, columns=[*FEATURE_NAMES, "label"])


def weight_report(ranker_path: str = DEFAULT_RANKER_PATH) -> pd.DataFrame:
    """Fitted weight per feature, sorted by magnitude, with a near-zero flag."""
    ranker = load_fitted_ranker(ranker_path)
    max_abs = max((abs(w) for w in ranker.weights.values()), default=0.0)
    rows = []
    for name in FEATURE_NAMES:
        weight = ranker.weights.get(name, 0.0)
        near_zero = max_abs > 0 and abs(weight) < NEAR_ZERO_RELATIVE_THRESHOLD * max_abs
        rows.append({"feature": name, "weight": weight, "near_zero": near_zero})
    df = pd.DataFrame(rows).sort_values("weight", key=abs, ascending=False).reset_index(drop=True)
    return df


def correlation_report(features_path: str = "data/features.jsonl") -> pd.DataFrame:
    """Pairwise Pearson correlation between every feature (design doc §6.3)."""
    df = load_feature_dataframe(features_path)
    return df[list(FEATURE_NAMES)].corr()


def cross_check_doc_predictions(weights: pd.DataFrame) -> list[str]:
    """§2.4/§2.4.1: "expect rare_tag_match and rating_style_fit to have
    small-but-nonzero weights, not necessarily zero." Returns human-
    readable lines reporting whether that held for this fit.
    """
    lines = []
    for name in ("rare_tag_match", "rating_style_fit"):
        row = weights[weights["feature"] == name].iloc[0]
        verdict = "near-zero (contradicts the doc's prediction)" if row["near_zero"] else "nonzero, as predicted"
        lines.append(f"{name}: weight={row['weight']:+.4f} -> {verdict}")
    return lines


def main(features_path: str = "data/features.jsonl", ranker_path: str = DEFAULT_RANKER_PATH) -> None:
    weights = weight_report(ranker_path)
    print("Fitted weights (sorted by |weight|):")
    print(weights.to_string(index=False))

    near_zero = weights[weights["near_zero"]]
    if len(near_zero):
        print(f"\nNear-zero (< {NEAR_ZERO_RELATIVE_THRESHOLD:.0%} of max |weight|) — inert without an ablation re-run:")
        for _, row in near_zero.iterrows():
            print(f"  {row['feature']}: {row['weight']:+.4f}")
    else:
        print("\nNo near-zero-weight features.")

    print("\nCross-check against §2.4/§2.4.1's predictions:")
    for line in cross_check_doc_predictions(weights):
        print(f"  {line}")

    corr = correlation_report(features_path)
    print("\nPairwise feature correlations:")
    print(corr.round(2).to_string())

    # Highlight the strongest off-diagonal correlations — pairs a near-zero
    # weight might be explained by (redundant with a stronger feature)
    # rather than genuinely uninformative on its own.
    pairs = corr.where(~corr.isna()).stack()
    pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]
    strongest = pairs.abs().sort_values(ascending=False).head(5)
    print("\nStrongest pairwise correlations:")
    for (a, b), _ in strongest.items():
        print(f"  {a} <-> {b}: {corr.loc[a, b]:+.2f}")


if __name__ == "__main__":
    main()
