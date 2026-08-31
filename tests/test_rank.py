"""
Contract tests for rank.py's hand-set weighted scoring (design doc §3.4
Step 6, §8.3 step C3) and the fitted logistic regression (§8.3 step C5).

Scope: score_candidates()/rank() and their supporting weighted-sum/sort/
truncate/top_k machinery — the part that is actually implemented and
running today (verified live via `python3 agent.py`). Individual feature
*values* (price_fit, category_match, etc.) are features.py's contract,
already covered by tests/test_features.py; this file only tests how
rank.py combines and orders those values.

fit_logistic_regression() (C5) is tested against small synthetic feature
matrices fabricated in this file, not the real ~4,200-row corpus — that
corpus is still blocked on Owner A's retrieval (§8.1, A2-A9) being real
rather than fixture (D6's `run_instrumented_corpus` exists per Owner D's
PR, but a run today would pool almost entirely negatives: keyword_search()
still ignores its query and returns a fixed handful of fixture ASINs
regardless of the real 50k-row catalogue, so the real target is almost
never in the pool). These tests protect the *fitting mechanics* — sklearn
wiring, GroupKFold-by-session_id, the scaler fold-in — so that once a real
corpus exists, only the input changes, not this code. Likewise llm_rerank()
(C7, closed as won't-do — no LLM access provided) is tested only for its
current "always None, always falls back" stub contract.
"""

import json
import math
import random

import pytest

from features import FEATURE_NAMES
from indexes import Indexes
from rank import (
    _fit_sign_constrained,
    HANDSET_WEIGHTS,
    TOP_K_RETURN,
    TOP_K_TRUNCATE,
    FittedRanker,
    fit_logistic_regression,
    llm_rerank,
    load_fitted_ranker,
    rank,
    rows_to_training_arrays,
    save_fitted_ranker,
    score_candidates,
)
from state import SessionState
from utils import Candidate


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def facts():
    """Three products, distinct pop, everything else held equal.

    Isolates ordering to the one feature that varies, so ranking tests can
    assert exact expected order without needing to override weights.
    """
    def record(pop):
        return {
            "dept": "Men",
            "cat3": "Clothing, Shoes & Jewelry > Men > Clothing > Jackets & Coats",
            "store": "generic",
            "price": None,
            "rating_number": 0,
            "pop": pop,
            "rating": 4.0,
            "blob": "a generic product with no special terms",
        }

    return {
        "HIGH_POP": record(0.9),
        "MID_POP": record(0.5),
        "LOW_POP": record(0.1),
    }


@pytest.fixture
def indexes(facts):
    return Indexes(
        catalog=[], fts_conn=None, embedding_matrix=None, embedding_asins=[],
        facts=facts, category_lists={},
    )


@pytest.fixture
def state_empty():
    """No slots, no profile terms — every constraint-satisfaction feature
    (category_match, brand_match, slot_coverage, rare_tag_match,
    rating_style_fit) evaluates to 0.0 for every candidate, so only pop/
    rating/bm25_norm/cos_sim/price_fit can move the score."""
    return SessionState(session_id="s")


@pytest.fixture
def pool():
    """Three candidates, no stream scores — isolates ordering to pop/rating."""
    return [
        Candidate(asin="HIGH_POP"),
        Candidate(asin="MID_POP"),
        Candidate(asin="LOW_POP"),
    ]


# --------------------------------------------------------------------------
# HANDSET_WEIGHTS — §8.3 step C3: "Constants at module top for single-line
# tuning."
# --------------------------------------------------------------------------

def test_fit_never_assigns_a_weight_that_contradicts_the_feature_lift():
    """A feature scoring higher on targets must not fit to a negative weight.

    With eleven collinear features and few positives, an unconstrained fit
    produces suppressor coefficients: category_match measured a clear
    positive lift on the real corpus yet fitted to -0.31, because
    slot_coverage already carries the same "matches the stated slots"
    signal through text. That is defensible for in-sample likelihood and
    indefensible as ranking behaviour -- it demotes products matching the
    category the shopper asked for.
    """
    import random

    rng = random.Random(20240531)
    rows, labels, groups = [], [], []
    for session in range(40):
        for cand in range(20):
            target = cand == 0
            # `helper` is pure noise; `signal` is genuinely higher on
            # targets, and `echo` duplicates it to force collinearity.
            signal = 1.0 if target else 0.0
            row = {name: rng.random() * 0.1 for name in FEATURE_NAMES}
            row["slot_coverage"] = signal + rng.random() * 0.05
            row["category_match"] = signal * 0.9 + rng.random() * 0.05
            rows.append(row)
            labels.append(1 if target else 0)
            groups.append(f"s{session}")

    ranker = fit_logistic_regression(rows, labels, groups)

    import statistics
    for name in ("slot_coverage", "category_match"):
        pos = statistics.mean(r[name] for r, y in zip(rows, labels) if y == 1)
        neg = statistics.mean(r[name] for r, y in zip(rows, labels) if y == 0)
        assert pos > neg, f"{name} should have positive lift in this fixture"
        assert ranker.weights[name] >= 0.0, (
            f"{name} has positive lift but fitted to {ranker.weights[name]:+.4f}"
        )


def test_sign_constrained_fit_matches_sklearn_when_unbounded():
    """The bounded optimiser must solve sklearn's own objective.

    Guards the whole constraint: if this diverges, every fitted weight the
    project ships is coming from a different objective than the one §3.4
    documents.
    """
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    rng = numpy.random.default_rng(11)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    pos = y == 1
    sw = numpy.where(pos, len(y) / (2.0 * pos.sum()), len(y) / (2.0 * (~pos).sum()))

    mine, intercept = _fit_sign_constrained(X, y, sw, [(None, None)] * 4)
    sk = LogisticRegression(class_weight="balanced", penalty="l2", tol=1e-10, max_iter=5000).fit(X, y)

    assert numpy.abs(sk.coef_[0] - mine).max() < 1e-4
    assert abs(sk.intercept_[0] - intercept) < 1e-4


def test_handset_weights_has_one_entry_per_feature():
    assert set(HANDSET_WEIGHTS) == set(FEATURE_NAMES)


def test_handset_weights_are_all_floats():
    for name, weight in HANDSET_WEIGHTS.items():
        assert isinstance(weight, float), f"{name} weight is {type(weight)}"


# --------------------------------------------------------------------------
# score_candidates() — the weighted-sum arithmetic itself
# --------------------------------------------------------------------------

def test_score_candidates_returns_one_pair_per_candidate(indexes, state_empty, pool):
    scored = score_candidates(pool, state_empty, indexes)
    assert len(scored) == len(pool)
    assert {asin for asin, _ in scored} == {c.asin for c in pool}


def test_score_candidates_defaults_to_handset_weights(indexes, state_empty, pool):
    """weights=None must produce the identical result to passing HANDSET_WEIGHTS explicitly."""
    default = dict(score_candidates(pool, state_empty, indexes))
    explicit = dict(score_candidates(pool, state_empty, indexes, HANDSET_WEIGHTS))
    assert default == explicit


def test_score_candidates_respects_custom_weights(indexes, state_empty, pool):
    """Zero every weight except pop: the score must equal facts[asin]['pop'] exactly."""
    isolate_pop = {name: 0.0 for name in FEATURE_NAMES}
    isolate_pop["pop"] = 1.0
    scored = dict(score_candidates(pool, state_empty, indexes, isolate_pop))
    assert scored["HIGH_POP"] == pytest.approx(0.9)
    assert scored["MID_POP"] == pytest.approx(0.5)
    assert scored["LOW_POP"] == pytest.approx(0.1)


def test_score_candidates_missing_weight_entries_default_to_zero_contribution(
    indexes, state_empty, pool
):
    """A weights dict that omits some FEATURE_NAMES must not crash or count them."""
    only_rating = {"rating": 1.0}  # every other feature implicitly weight 0
    scored = dict(score_candidates(pool, state_empty, indexes, only_rating))
    # All three candidates share rating=4.0/5.0=0.8 in the fixture, so scores
    # must all be equal — proves the other nine features contributed nothing.
    assert scored["HIGH_POP"] == pytest.approx(0.8)
    assert scored["MID_POP"] == pytest.approx(0.8)
    assert scored["LOW_POP"] == pytest.approx(0.8)


def test_score_candidates_survives_empty_pool(indexes, state_empty):
    assert score_candidates([], state_empty, indexes) == []


def test_score_candidates_survives_asin_absent_from_facts(indexes, state_empty):
    """A candidate feature_vector can't compute for shouldn't crash the whole turn."""
    scored = score_candidates(
        [Candidate(asin="DOES_NOT_EXIST")], state_empty, indexes
    )
    assert len(scored) == 1
    asin, value = scored[0]
    assert asin == "DOES_NOT_EXIST"
    assert isinstance(value, float)
    assert not math.isnan(value)


# --------------------------------------------------------------------------
# rank() — sort, truncate, top_k, ranker override, llm_rerank fallback
# --------------------------------------------------------------------------

def test_rank_orders_by_score_descending(indexes, state_empty, pool):
    """HANDSET_WEIGHTS weights pop positively; higher pop must rank first."""
    ranked = rank(pool, state_empty, indexes, top_k=3)
    assert ranked == ["HIGH_POP", "MID_POP", "LOW_POP"]


def test_rank_returns_exactly_top_k(indexes, state_empty, pool):
    ranked = rank(pool, state_empty, indexes, top_k=2)
    assert len(ranked) == 2
    assert ranked == ["HIGH_POP", "MID_POP"]


def test_rank_returns_fewer_than_top_k_if_pool_is_smaller(indexes, state_empty, pool):
    ranked = rank(pool, state_empty, indexes, top_k=10)
    assert len(ranked) == 3  # pool only has 3 candidates


def test_rank_default_top_k_matches_top_k_return_constant(indexes, state_empty, pool):
    ranked = rank(pool, state_empty, indexes)
    assert len(ranked) <= TOP_K_RETURN


def test_rank_truncates_to_top_k_truncate_before_top_k_slice(indexes, facts):
    """§3.4 Step 6: 'truncate to 30 ... return 10.' A pool bigger than
    TOP_K_TRUNCATE must never return more than TOP_K_TRUNCATE, even if a
    caller asks for more via top_k."""
    big_facts = {f"ASIN_{i}": {**facts["HIGH_POP"], "pop": i / 100} for i in range(40)}
    big_indexes = Indexes(
        catalog=[], fts_conn=None, embedding_matrix=None, embedding_asins=[],
        facts=big_facts, category_lists={},
    )
    big_pool = [Candidate(asin=asin) for asin in big_facts]
    state = SessionState(session_id="s")
    ranked = rank(big_pool, state, big_indexes, top_k=35)
    assert len(ranked) == TOP_K_TRUNCATE


def test_rank_survives_empty_pool(indexes, state_empty):
    assert rank([], state_empty, indexes) == []


def test_rank_uses_ranker_weights_when_supplied(indexes, state_empty, pool):
    """A FittedRanker weighting only rating (equal across the fixture) must
    flatten the ordering that pop alone would otherwise produce."""
    isolate_rating = {name: 0.0 for name in FEATURE_NAMES}
    isolate_rating["rating"] = 1.0
    ranker = FittedRanker(weights=isolate_rating)
    ranked = rank(pool, state_empty, indexes, ranker=ranker, top_k=3)
    # All three tie at rating=0.8, so order falls back to Python's stable
    # sort over the original pool order rather than pop-descending.
    assert set(ranked) == {"HIGH_POP", "MID_POP", "LOW_POP"}
    assert ranked == ["HIGH_POP", "MID_POP", "LOW_POP"]  # stable sort, input order preserved on ties


def test_rank_without_ranker_falls_back_to_handset_weights(indexes, state_empty, pool):
    with_none = rank(pool, state_empty, indexes, ranker=None, top_k=3)
    with_handset = rank(
        pool, state_empty, indexes, ranker=FittedRanker(weights=dict(HANDSET_WEIGHTS)), top_k=3
    )
    assert with_none == with_handset


def test_rank_llm_rerank_flag_off_by_default_does_not_change_order(indexes, state_empty, pool):
    without_flag = rank(pool, state_empty, indexes, top_k=3)
    with_flag_off = rank(pool, state_empty, indexes, use_llm_rerank=False, top_k=3)
    assert without_flag == with_flag_off


def test_rank_llm_rerank_stub_falls_back_to_regression_ordering(indexes, state_empty, pool):
    """llm_rerank() is a stub that always returns None (§8.3 step C7's
    defensive-parsing contract: malformed/no output -> fall back). Since it
    always returns None right now, use_llm_rerank=True must currently
    produce the identical order to use_llm_rerank=False."""
    without_flag = rank(pool, state_empty, indexes, top_k=3)
    with_flag_on = rank(pool, state_empty, indexes, use_llm_rerank=True, top_k=3)
    assert without_flag == with_flag_on


# --------------------------------------------------------------------------
# FittedRanker — default shape, independent per-instance weights
# --------------------------------------------------------------------------

def test_fitted_ranker_defaults_to_handset_weights():
    ranker = FittedRanker()
    assert ranker.weights == HANDSET_WEIGHTS
    assert ranker.intercept == 0.0
    assert ranker.feature_means == {}
    assert ranker.feature_stds == {}


def test_fitted_ranker_default_weights_are_an_independent_copy():
    """Mutating one instance's weights must not leak into HANDSET_WEIGHTS
    or into another instance — field(default_factory=...) must actually be
    called per-instance, not shared as one mutable default."""
    a = FittedRanker()
    b = FittedRanker()
    a.weights["pop"] = 999.0
    assert b.weights["pop"] != 999.0
    assert HANDSET_WEIGHTS["pop"] != 999.0


# --------------------------------------------------------------------------
# fit_logistic_regression() — real sklearn fit (§8.3 step C5), exercised
# against small synthetic feature matrices fabricated below, not the real
# corpus (still blocked on Owner A's retrieval — see module docstring).
# --------------------------------------------------------------------------

sklearn = pytest.importorskip("sklearn")  # C5's dependency; skip this section without it


def _synthetic_rows(n_sessions: int = 12, rows_per_session: int = 6, seed: int = 0):
    """Fabricate a feature matrix where `pop` alone predicts the label.

    One positive (high pop) and several negatives (low pop, plus noise on
    every other feature) per fake session — enough sessions for GroupKFold
    to form multiple folds, and a signal strong enough that a real fit
    should recover a positive `pop` weight and near-zero weight elsewhere.
    """
    rng = random.Random(seed)
    feature_matrix, labels, groups = [], [], []
    for session_i in range(n_sessions):
        session_id = f"session-{session_i}"
        for row_i in range(rows_per_session):
            is_target = row_i == 0
            row = {name: rng.uniform(0.0, 1.0) for name in FEATURE_NAMES}
            row["pop"] = rng.uniform(0.8, 1.0) if is_target else rng.uniform(0.0, 0.2)
            feature_matrix.append(row)
            labels.append(1 if is_target else 0)
            groups.append(session_id)
    return feature_matrix, labels, groups


def test_fit_logistic_regression_returns_fitted_ranker_not_handset():
    feature_matrix, labels, groups = _synthetic_rows()
    ranker = fit_logistic_regression(feature_matrix, labels, groups)
    assert isinstance(ranker, FittedRanker)
    assert ranker.weights != HANDSET_WEIGHTS  # actually fit, not the stub passthrough


def test_fit_logistic_regression_recovers_the_dominant_signal():
    """pop alone separates the classes here; its fitted weight must be the
    largest in magnitude and must be positive (higher pop -> more likely
    the target), same sign convention as HANDSET_WEIGHTS."""
    feature_matrix, labels, groups = _synthetic_rows()
    ranker = fit_logistic_regression(feature_matrix, labels, groups)
    assert ranker.weights["pop"] > 0
    assert ranker.weights["pop"] == max(ranker.weights.values(), key=abs)


def test_fit_logistic_regression_weights_cover_every_feature():
    feature_matrix, labels, groups = _synthetic_rows()
    ranker = fit_logistic_regression(feature_matrix, labels, groups)
    assert set(ranker.weights) == set(FEATURE_NAMES)
    assert set(ranker.feature_means) == set(FEATURE_NAMES)
    assert set(ranker.feature_stds) == set(FEATURE_NAMES)


def test_fit_logistic_regression_persists_scaler_stats():
    """feature_means/feature_stds must reflect the real training data, not
    defaults — spot-check pop's mean lands strictly between the two
    clusters this fixture draws it from ([0, 0.2] negatives, [0.8, 1.0]
    positive)."""
    feature_matrix, labels, groups = _synthetic_rows()
    ranker = fit_logistic_regression(feature_matrix, labels, groups)
    assert 0.2 < ranker.feature_means["pop"] < 0.8
    assert ranker.feature_stds["pop"] > 0


def test_fit_logistic_regression_reports_cv_accuracy():
    """12 distinct sessions -> GroupKFold should actually run and report a
    score; the synthetic signal is strong enough to expect well above
    chance (0.5) on held-out sessions."""
    feature_matrix, labels, groups = _synthetic_rows()
    ranker = fit_logistic_regression(feature_matrix, labels, groups)
    assert ranker.cv_accuracy is not None
    assert 0.0 <= ranker.cv_accuracy <= 1.0
    assert ranker.cv_accuracy > 0.5


def test_fit_logistic_regression_cv_accuracy_none_with_too_few_sessions():
    """A single session can't form 2 GroupKFold folds; fitting must still
    succeed (the whole-data fit doesn't need multiple groups), just without
    a cv_accuracy figure."""
    feature_matrix, labels, _ = _synthetic_rows(n_sessions=1)
    ranker = fit_logistic_regression(feature_matrix, labels, groups=["only-session"] * len(labels))
    assert ranker.cv_accuracy is None


def test_fit_logistic_regression_reports_ranking_metrics():
    """The same folds must also report AUC and average precision, the two
    figures that survive this corpus's class imbalance (FittedRanker's
    docstring). Both are computed on held-out sessions, so the synthetic
    signal should put them well above their respective chance rates."""
    feature_matrix, labels, groups = _synthetic_rows()
    ranker = fit_logistic_regression(feature_matrix, labels, groups)
    assert ranker.cv_auc is not None
    assert 0.0 <= ranker.cv_auc <= 1.0
    assert ranker.cv_auc > 0.5  # chance
    assert ranker.cv_ap is not None
    assert 0.0 <= ranker.cv_ap <= 1.0


def test_fit_logistic_regression_ranking_metrics_none_with_too_few_sessions():
    """No folds means no held-out ranking metrics either — but the fit
    itself must still succeed, same contract as cv_accuracy."""
    feature_matrix, labels, _ = _synthetic_rows(n_sessions=1)
    ranker = fit_logistic_regression(feature_matrix, labels, groups=["only-session"] * len(labels))
    assert ranker.cv_auc is None
    assert ranker.cv_ap is None


def test_load_fitted_ranker_accepts_legacy_json_without_ranking_metrics(tmp_path):
    """A ranker.json persisted before cv_auc/cv_ap existed must still load,
    so an existing models/ranker.json is not invalidated by the new fields
    — the scoring path only ever reads `weights`."""
    legacy = {
        "weights": dict(HANDSET_WEIGHTS),
        "intercept": -1.5,
        "feature_means": {},
        "feature_stds": {},
        "cv_accuracy": 0.9,
    }
    path = tmp_path / "legacy_ranker.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    ranker = load_fitted_ranker(str(path))
    assert ranker.weights == HANDSET_WEIGHTS
    assert ranker.cv_accuracy == 0.9
    assert ranker.cv_auc is None
    assert ranker.cv_ap is None


def test_fit_logistic_regression_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_logistic_regression(feature_matrix=[{"pop": 0.5}], labels=[1, 0], groups=["s1"])


def test_fit_logistic_regression_rejects_single_class_labels():
    feature_matrix, _, groups = _synthetic_rows()
    with pytest.raises(ValueError):
        fit_logistic_regression(feature_matrix, labels=[1] * len(feature_matrix), groups=groups)


def test_fit_logistic_regression_missing_feature_keys_default_to_zero():
    """A row dict that omits a feature name must not crash — matches
    score_candidates()'s existing missing-weight convention."""
    feature_matrix = [{"pop": 0.9}, {"pop": 0.1}] * 6
    labels = [1, 0] * 6
    groups = [f"session-{i}" for i in range(6) for _ in range(2)]
    ranker = fit_logistic_regression(feature_matrix, labels, groups)
    assert set(ranker.weights) == set(FEATURE_NAMES)


def test_fitted_ranker_changes_rank_order_vs_handset(indexes, state_empty, pool):
    """A ranker fit to prefer the opposite of pop must be able to invert
    the HANDSET_WEIGHTS ordering rank() otherwise produces for `pool`."""
    invert_pop = {name: 0.0 for name in FEATURE_NAMES}
    invert_pop["pop"] = -1.0
    ranker = FittedRanker(weights=invert_pop)
    ranked = rank(pool, state_empty, indexes, ranker=ranker, top_k=3)
    assert ranked == ["LOW_POP", "MID_POP", "HIGH_POP"]


# --------------------------------------------------------------------------
# rows_to_training_arrays() — the D6 row-schema adapter (§6.6 step 2)
# --------------------------------------------------------------------------

def test_rows_to_training_arrays_unpacks_d6_row_schema():
    rows = [
        {
            "session_id": "s1",
            "turn": 1,
            "n_hard_slots": 2,
            "parent_asin": "TARGET",
            "features": [float(i) for i in range(len(FEATURE_NAMES))],
            "label": 1,
        },
        {
            "session_id": "s1",
            "turn": 1,
            "n_hard_slots": 2,
            "parent_asin": "OTHER",
            "features": [0.0] * len(FEATURE_NAMES),
            "label": 0,
        },
    ]
    feature_matrix, labels, groups = rows_to_training_arrays(rows)
    assert feature_matrix[0] == dict(zip(FEATURE_NAMES, range(len(FEATURE_NAMES))))
    assert labels == [1, 0]
    assert groups == ["s1", "s1"]


def test_rows_to_training_arrays_output_feeds_fit_logistic_regression_directly():
    """The adapter's output must be accepted as-is by fit_logistic_regression
    — this is the whole point of the adapter."""
    feature_matrix, labels, groups = _synthetic_rows()
    rows = [
        {"session_id": g, "features": [row.get(name, 0.0) for name in FEATURE_NAMES], "label": label}
        for row, label, g in zip(feature_matrix, labels, groups)
    ]
    adapted_matrix, adapted_labels, adapted_groups = rows_to_training_arrays(rows)
    ranker = fit_logistic_regression(adapted_matrix, adapted_labels, adapted_groups)
    assert isinstance(ranker, FittedRanker)


# --------------------------------------------------------------------------
# save_fitted_ranker() / load_fitted_ranker() — persistence (§8.3 step C5:
# "Persist the fitted model/scaler to disk ... so it doesn't need
# refitting every run.")
# --------------------------------------------------------------------------

def test_save_then_load_fitted_ranker_round_trips(tmp_path):
    feature_matrix, labels, groups = _synthetic_rows()
    ranker = fit_logistic_regression(feature_matrix, labels, groups)
    path = str(tmp_path / "ranker.json")
    save_fitted_ranker(ranker, path)
    loaded = load_fitted_ranker(path)
    assert loaded == ranker


def test_save_fitted_ranker_creates_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "ranker.json")
    save_fitted_ranker(FittedRanker(), path)
    assert load_fitted_ranker(path) == FittedRanker()


def test_load_fitted_ranker_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_fitted_ranker("/nonexistent/path/ranker.json")


def test_llm_rerank_stub_always_returns_none():
    result = llm_rerank(candidates=[], dialogue=["hi"], slots={"category": "jacket"})
    assert result is None
