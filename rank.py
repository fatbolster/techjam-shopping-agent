"""Scoring (hand-set weights + fitted logistic regression) and the LLM rerank.

Design doc §3.4 Step 6 (Ranking) and §6.6 (Ranker training protocol).

Owner: Emerson (Ranking). §8.3, step C3 (hand-set weighted sum), step C5 (fit
logistic regression on D6's matrix, GroupKFold by session_id), step C6
(optional pairwise objective comparison), step C7 (flagged LLM rerank with
defensive parsing).

score_candidates()/rank() (C3) do real weighted-sum arithmetic over real
features.py feature values as of issues #20-#23 (C1-C4). fit_logistic_regression()
(C5) now does a real sklearn fit; see its docstring for the raw-feature-space
weight fold-in that keeps score_candidates() unchanged. Real corpus not yet
available (blocked on Owner A's retrieval, §8.1 A2-A9, being real rather
than fixture — see rows_to_training_arrays() for the D6 adapter, exercised
here only against synthetic fixtures until then). llm_rerank() (C7) remains
a deliberate stub — out of scope here (closed as won't-do, no LLM access
provided).
sklearn is imported lazily inside fit_logistic_regression() so this module
is importable without the dependency installed (§9: "Pipeline is fully
functional with zero LLM calls").
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from features import FEATURE_NAMES, extract_features
from indexes import Indexes
from state import SessionState
from utils import Candidate

# Where the fitted model is persisted (gitignored, like the embedding
# matrix — §8.3 step C5: "Persist the fitted model/scaler to disk ... so
# it doesn't need refitting every run.").
DEFAULT_RANKER_PATH = "models/ranker.json"

# Hand-set weights, one per FEATURE_NAMES entry (§8.3 step C3: "Constants
# at module top for single-line tuning. Produces a complete ranking on day
# one."). Starting values below (issue #22) are not fit — C5's logistic
# regression supersedes them once it exists (rank() falls back to these
# only when no FittedRanker is supplied). Relative magnitudes follow the
# strength of evidence in §2: pop carries the single strongest documented
# prior (§2.1: 63%/81%/96% of targets in the top 1%/5%/20% by review
# count), so it is weighted above the two retrieval-relevance features.
# rating_style_fit is deliberately small — §2.4.1 measured a real but
# modest effect (0.131 stars, ~16% of one catalogue IQR) — and price_fit
# is muted because it is neutral (0.5) for 78.9% of the catalogue (§2.2),
# so most candidates get no signal from it either way.
HANDSET_WEIGHTS: dict[str, float] = {
    "bm25_norm": 1.5,
    "cos_sim": 1.5,
    "pop": 2.0,
    "rating": 1.0,
    "price_fit": 0.5,
    "category_match": 1.5,
    "brand_match": 1.0,
    "slot_coverage": 1.0,
    "rare_tag_match": 0.5,
    "rating_style_fit": 0.3,
}

TOP_K_RETURN = 10
TOP_K_TRUNCATE = 30  # §3.4 Step 6: "truncate to 30, optionally rerank with an LLM"


@dataclass
class FittedRanker:
    """A fitted logistic regression ranker plus its feature scaler.

    Design doc §3.4 Step 6: "Logistic regression (scikit-learn,
    class_weight='balanced', L2 regularisation) over standardised
    features. Eleven parameters including the intercept." Owner Emerson, step C5.

    Attributes:
        weights: Fitted coefficient per feature, in FEATURE_NAMES order.
            Already folded together with the scaler (raw-feature-space, so
            score_candidates() needs no separate scaling step — see
            fit_logistic_regression()).
        intercept: Fitted intercept term, same raw-feature-space fold-in.
            Unused by score_candidates() today (a per-candidate-turn
            constant does not change relative order), kept for probability
            interpretation / future use.
        feature_means: Per-feature mean used by the persisted StandardScaler.
        feature_stds: Per-feature std used by the persisted StandardScaler.
        cv_accuracy: Mean GroupKFold-by-session_id held-out accuracy from
            fitting (§6.6 step 5), or None when fit with hand-set weights
            (never fitted) or too few distinct sessions to form 2+ folds.

            RETAINED FOR CONTINUITY, BUT DO NOT QUOTE IT. The training
            corpus is ~2.7% positive (632 of 23,012 rows), so always
            predicting "not the target" scores 0.973 while the fitted model
            scores 0.924 — accuracy sits *below* the majority-class
            baseline and reads as "worse than useless" for a model that
            ranks well. Accuracy is the wrong family of metric here: this
            is a ranking problem scored by where the target lands, not a
            classification problem scored by how many rows got the right
            side of 0.5. Use `cv_auc`/`cv_ap` below.
        cv_auc: Mean held-out ROC-AUC over the same folds — the probability
            that a randomly chosen target outranks a randomly chosen pool
            negative. 0.5 is chance, and it is unaffected by class
            imbalance, which is exactly why it belongs here.
        cv_ap: Mean held-out average precision (PR-AUC) over the same
            folds. Chance equals the positive rate (~0.0275), so this is
            the honest "how much better than nothing" figure, and the one
            most sensitive to the head of the ranking that Hit@10 actually
            scores.
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(HANDSET_WEIGHTS))
    intercept: float = 0.0
    feature_means: dict[str, float] = field(default_factory=dict)
    feature_stds: dict[str, float] = field(default_factory=dict)
    cv_accuracy: Optional[float] = None
    cv_auc: Optional[float] = None
    cv_ap: Optional[float] = None


def score_candidates(
    candidates: list[Candidate], state: SessionState, indexes: Indexes, weights: Optional[dict[str, float]] = None
) -> list[tuple[str, float]]:
    """Score every candidate as a weighted sum of the ten features.

    Design doc §3.4 Step 6; §8.3 step C3. Falls back to HANDSET_WEIGHTS
    when `weights` is None, i.e. before a FittedRanker exists (§7.4
    descoping order #4: "Falls back to hand-set weights.").

    STUB: uses extract_features() (itself all fixture math), so the
    weighted-sum arithmetic here is real but every input feature is not.

    Args:
        candidates: The (truncated or full) candidate pool for this turn.
        state: Current session state.
        indexes: Offline indexes bundle.
        weights: Feature name -> weight, or None for HANDSET_WEIGHTS.

    Returns:
        (asin, score) pairs, unsorted.
    """
    weights = weights if weights is not None else HANDSET_WEIGHTS
    scored = []
    for cand in candidates:
        feats = extract_features(cand, candidates, indexes, state)
        total = sum(weights.get(name, 0.0) * value for name, value in feats.items())
        scored.append((cand.asin, total))
    return scored


def fit_logistic_regression(
    feature_matrix: list[dict[str, float]], labels: list[int], groups: list[str]
) -> FittedRanker:
    """Fit the nine-feature (+intercept) logistic regression ranker.

    Design doc §6.6 Ranker training protocol and §3.4 Step 6: "class_weight
    = 'balanced', L2 regularisation ... validate with GroupKFold grouped by
    session_id." Owner Emerson, §8.3 step C5. Input comes from D6's instrumented
    run (telemetry.py / simulate.py), joined against ground truth offline —
    see `rows_to_training_arrays()` for the adapter from D6's row schema to
    the three parallel arguments here.

    sklearn is imported here, not at module level, so rank.py stays
    importable in environments without it installed (§1.2: "No hosted
    model access provided ... Pipeline is fully functional with zero LLM
    calls"; the regression is a separate, optional-at-import dependency).

    Fits `sklearn.preprocessing.StandardScaler` + `LogisticRegression(
    class_weight="balanced", penalty="l2")` on the full input, then folds
    the scaler into the returned weights/intercept so `score_candidates()`
    can keep applying `sum(weight * raw_feature_value)` unchanged — no
    scaling step needed at scoring time (§6.6: "raw_weight_i = coef_i /
    scale_i; raw_intercept = intercept_ - sum(coef_i * mean_i / scale_i)",
    the standard fold-in for `intercept + coef . ((x - mean) / scale)`).
    `feature_means`/`feature_stds` are still persisted on the FittedRanker
    per "scaler persisted alongside the model", for inspection/C8's
    coefficient report and so `rank.py` never has to reconstruct them from
    training data.

    A GroupKFold(session_id) validation pass also runs, purely to report
    out-of-session performance — the returned model itself is refit on every
    row (§6.6 step 5 validates the *approach*, it does not select which
    fold's model ships). Nothing downstream reads these numbers: they are
    persisted and printed for inspection, and no weight, threshold, or
    shipping decision is conditioned on them.

    Three figures come back. `cv_accuracy` is retained for continuity with
    §6.6 but is misleading on this corpus (see FittedRanker's docstring:
    it lands below the majority-class baseline). `cv_auc` and `cv_ap` are
    the ones to read — ranking metrics for what is, at inference, purely a
    ranking task.

    Args:
        feature_matrix: One dict per training row, feature name (per
            FEATURE_NAMES) -> value. Missing feature keys default to 0.0.
        labels: 1 for the session's target ASIN, 0 otherwise (§6.6 step 3).
        groups: session_id per row, for GroupKFold (§6.6 step 5).

    Returns:
        A FittedRanker with real fitted weights/intercept (raw-feature
        space) and the persisted scaler's per-feature means/stds.

    Raises:
        ValueError: fewer than 2 rows, or labels are all one class (sklearn
            cannot fit a decision boundary from a single class).
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    if len(feature_matrix) != len(labels) or len(feature_matrix) != len(groups):
        raise ValueError(
            f"feature_matrix ({len(feature_matrix)}), labels ({len(labels)}), and "
            f"groups ({len(groups)}) must be the same length"
        )
    if len(set(labels)) < 2:
        raise ValueError("labels must contain both classes (need at least one 0 and one 1)")

    X = np.array([[row.get(name, 0.0) for name in FEATURE_NAMES] for row in feature_matrix])
    y = np.array(labels)
    groups_arr = np.array(groups)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def _new_model() -> LogisticRegression:
        return LogisticRegression(class_weight="balanced", penalty="l2")

    # GroupKFold validation (§6.6 step 5) — session_id groups, not a plain
    # random split, since rows within a session are not independent.
    # Report-only: skipped (rather than erroring) when there are too few
    # distinct sessions to form at least 2 folds.
    n_unique_groups = len(set(groups))
    cv_accuracy: Optional[float] = None
    cv_auc: Optional[float] = None
    cv_ap: Optional[float] = None
    if n_unique_groups >= 2:
        n_splits = min(5, n_unique_groups)
        fold_scores: list[float] = []
        fold_aucs: list[float] = []
        fold_aps: list[float] = []
        for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X_scaled, y, groups=groups_arr):
            if len(set(y[train_idx].tolist())) < 2:
                continue  # a fold whose training split lost one class can't fit
            fold_model = _new_model().fit(X_scaled[train_idx], y[train_idx])
            fold_scores.append(fold_model.score(X_scaled[test_idx], y[test_idx]))
            # Ranking metrics need both classes present in the held-out
            # split to be defined at all (AUC is undefined with one class,
            # and average precision degenerates). A single-class test fold
            # is possible under GroupKFold when every session in it had its
            # target miss the pool, so guard rather than assume.
            if len(set(y[test_idx].tolist())) >= 2:
                fold_probs = fold_model.predict_proba(X_scaled[test_idx])[:, 1]
                fold_aucs.append(float(roc_auc_score(y[test_idx], fold_probs)))
                fold_aps.append(float(average_precision_score(y[test_idx], fold_probs)))
        if fold_scores:
            cv_accuracy = sum(fold_scores) / len(fold_scores)
        if fold_aucs:
            cv_auc = sum(fold_aucs) / len(fold_aucs)
        if fold_aps:
            cv_ap = sum(fold_aps) / len(fold_aps)

    model = _new_model().fit(X_scaled, y)

    # Fold the scaler into the weights so score_candidates() keeps scoring
    # raw feature values directly: intercept + coef . ((x - mean) / scale)
    # == (intercept - sum(coef * mean / scale)) + sum((coef / scale) * x).
    coef = model.coef_[0]
    raw_weights = coef / scaler.scale_
    raw_intercept = float(model.intercept_[0] - np.sum(coef * scaler.mean_ / scaler.scale_))

    return FittedRanker(
        weights=dict(zip(FEATURE_NAMES, (float(w) for w in raw_weights))),
        intercept=raw_intercept,
        feature_means=dict(zip(FEATURE_NAMES, (float(m) for m in scaler.mean_))),
        feature_stds=dict(zip(FEATURE_NAMES, (float(s) for s in scaler.scale_))),
        cv_accuracy=cv_accuracy,
        cv_auc=cv_auc,
        cv_ap=cv_ap,
    )


def rows_to_training_arrays(
    rows: list[dict],
) -> tuple[list[dict[str, float]], list[int], list[str]]:
    """Adapt D6's training-row schema into fit_logistic_regression()'s inputs.

    D6 (telemetry.py `build_training_rows()`) emits one dict per row:
    `session_id, turn, n_hard_slots, parent_asin, features` (a list of ten
    floats in FEATURE_NAMES order, per §6.6 step 2), `label`. This unpacks
    that into the three parallel arguments `fit_logistic_regression()`
    expects, matching FEATURE_NAMES order back onto feature names.

    Args:
        rows: Output of `telemetry.build_training_rows()`.

    Returns:
        (feature_matrix, labels, groups) — same length, same row order.
    """
    feature_matrix = [dict(zip(FEATURE_NAMES, row["features"])) for row in rows]
    labels = [row["label"] for row in rows]
    groups = [row["session_id"] for row in rows]
    return feature_matrix, labels, groups


def save_fitted_ranker(ranker: FittedRanker, path: str = DEFAULT_RANKER_PATH) -> None:
    """Persist a FittedRanker to disk as JSON (§8.3 step C5, "so it doesn't
    need refitting every run"). Plain JSON, not pickle/joblib: a
    FittedRanker holds only the folded-in weights/intercept and the
    scaler's means/stds, all plain floats — no sklearn objects to
    serialize, so this stays readable and dependency-free to load.

    Args:
        ranker: The FittedRanker to persist.
        path: Destination path (default gitignored, like the embedding
            matrix — see `models/` in .gitignore).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(ranker), f, indent=2, sort_keys=True)


def load_fitted_ranker(path: str = DEFAULT_RANKER_PATH) -> FittedRanker:
    """Load a FittedRanker previously written by save_fitted_ranker().

    Args:
        path: Source path (default matches save_fitted_ranker()'s default).

    Returns:
        The persisted FittedRanker.

    Raises:
        FileNotFoundError: no ranker has been persisted at `path` yet.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return FittedRanker(**data)


def llm_rerank(
    candidates: list[Candidate], dialogue: list[str], slots: dict[str, str]
) -> Optional[list[str]]:
    """Optional final LLM pass over the top 30 candidates.

    Design doc §3.4 Step 6: "one LLM call carrying the dialogue, slot
    dictionary and 30 candidate titles, returning a reordering; parsed
    defensively with fallback to the regression ordering." Owner Emerson, §8.3
    step C7. Behind a flag; token counts must surface in `usage` (§6.1
    Efficiency).

    STUB: makes no LLM call and always returns None, signalling "fall back
    to the regression ordering" per the defensive-parsing contract.

    Args:
        candidates: Up to 30 candidates, already truncated and scored.
        dialogue: The turn-by-turn utterance history for this session.
        slots: The current slot dictionary.

    Returns:
        A reordered list of ASINs, or None on failure/malformed output/no
        LLM configured (fixture: always None).
    """
    return None


def rank(
    candidates: list[Candidate],
    state: SessionState,
    indexes: Indexes,
    ranker: Optional[FittedRanker] = None,
    use_llm_rerank: bool = False,
    top_k: int = TOP_K_RETURN,
) -> list[str]:
    """Score, truncate, optionally rerank, and return the top `top_k` ASINs.

    Design doc §7.2 interface contract sketch: `rank(candidates, state) ->
    [asin] * 10`. §3.4 Step 6: "Score all candidates on ten features,
    truncate to 30, optionally rerank with an LLM, return 10." `top_k` is
    exposed here (rather than hardcoded) because the supplied kit's
    `Agent.respond(..., top_k)` passes it in per call.

    Args:
        candidates: The full candidate pool for this turn.
        state: Current session state.
        indexes: Offline indexes bundle.
        ranker: A FittedRanker to use in place of HANDSET_WEIGHTS, or None.
        use_llm_rerank: Whether to attempt the flagged LLM rerank pass.
        top_k: How many ASINs to return (the kit passes 10).

    Returns:
        Up to `top_k` ASINs, highest-ranked first.
    """
    weights = ranker.weights if ranker is not None else None
    scored = score_candidates(candidates, state, indexes, weights)
    scored.sort(key=lambda pair: pair[1], reverse=True)
    top = [asin for asin, _ in scored[:TOP_K_TRUNCATE]]

    if use_llm_rerank:
        top_candidates = [c for c in candidates if c.asin in set(top)]
        reranked = llm_rerank(top_candidates, [state.canonical_intent], state.slots)
        if reranked is not None:
            top = reranked

    return top[:top_k]
