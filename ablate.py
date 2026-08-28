"""Ablation harness: per-component configs, scored over all 200 sessions.

Design doc §6.3 (Ablation protocol) and §6.4 (Slicing).

Owner: Marcus (Evaluation and integration). §8.5, step E5 ("Config flags +
evaluator. Feature ablations re-score logged candidates without re-running
retrieval; stream and filter ablations trigger a full pass."), step E6
("Populate the ablation table; slice by scenario_type.").

Two real code paths, matching §6.3's "two protocol notes" exactly:
- "feature" kind (-rating_style_fit, -popularity_feature, -rare_tag_match):
  re-ranks each session's already-logged final-turn candidate pool
  (data/telemetry.jsonl's per-candidate feature vectors) with one feature's
  HANDSET_WEIGHTS entry zeroed. No re-retrieval — the pool is unchanged,
  only the ranking within it — so only MRR is meaningful (§6.3's table:
  every feature-kind row reads "MRR", never Hit@10/MTTC, since Hit@10 is
  bounded by pool recall, not ranking, and MTTC depends on turn-by-turn
  dynamics this re-score can't reconstruct from one turn's snapshot).
- "stream"/"filter" kind (-popularity_stream, -semantic_stream,
  -keyword_stream, -department_filter): changes pool composition, so needs
  a full pass. Runs through evaluate.record_baseline() unmodified —
  Agent.respond()'s signature is fixed to the evaluator's baseline exactly
  (agent.py's docstring), so there's no parameter to carry an ablation flag
  through it. Instead, retrieval.STREAM_QUOTAS / DEPARTMENT_FILTER_ENABLED
  are toggled for the duration of the run (see _stream_disabled() /
  _department_filter_disabled() below) — the real, unmodified pipeline
  just sees different module-level constants, the same way a config file
  would change them.

"baseline" and "-llm_rerank" need no special-casing: baseline is
record_baseline() with nothing overridden; -llm_rerank's real answer is
"no change" without further computation, since llm_rerank() (C7) is a
stub that always returns None (falls back to the regression ordering) —
there being no LLM access provided, per its own docstring.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import retrieval
from evaluate import AggregateScore, record_baseline, score_transcripts
from features import FEATURE_NAMES
from rank import HANDSET_WEIGHTS

# One row per §6.3's ablation table. `kind` distinguishes the two protocol
# notes: "feature" ablations re-score logged candidates (fast, no
# re-retrieval); "stream"/"filter" ablations change pool composition and
# require a full pass (§6.3 "Two protocol notes").
ABLATION_CONFIGS: tuple[dict, ...] = (
    {"name": "full_pipeline", "kind": "baseline", "reads": "all"},
    {"name": "-rating_style_fit", "kind": "feature", "reads": "MRR"},
    {"name": "-popularity_feature", "kind": "feature", "reads": "MRR"},
    {"name": "-popularity_stream", "kind": "stream", "reads": "Hit@10"},
    {"name": "-semantic_stream", "kind": "stream", "reads": "Hit@10"},
    {"name": "-keyword_stream", "kind": "stream", "reads": "Hit@10"},
    {"name": "-department_filter", "kind": "filter", "reads": "Hit@10 and MRR"},
    {"name": "-llm_rerank", "kind": "feature", "reads": "MRR vs Efficiency"},
    {"name": "-rare_tag_match", "kind": "feature", "reads": "MRR (22% subset)"},
)

# Slices reported alongside every ablation row (§6.4).
SCENARIO_SLICES: tuple[str, ...] = ("buying", "browsing", "intent_override", "boundary")

# -config name -> (FEATURE_NAMES entry to zero, restrict to sessions where
# the target's logged value for that feature is nonzero?). rare_tag_match
# is scoped to "the 22% subset" (§6.3) — sessions where the signal could
# possibly matter; the other two feature ablations run over every session.
_FEATURE_ABLATIONS: dict[str, tuple[str, bool]] = {
    "-rating_style_fit": ("rating_style_fit", False),
    "-popularity_feature": ("pop", False),
    "-rare_tag_match": ("rare_tag_match", True),
}


@contextmanager
def _stream_disabled(stream: str):
    """Zero one stream's quota on both tracks for the duration of the block.

    Mutates retrieval.STREAM_QUOTAS in place (not a reassignment) so every
    module that already imported the dict (agent.py holds no reference to
    it directly, but retrieval.retrieve() reads it by module-global lookup
    at call time either way) sees the override; restores the original
    values on exit even if the block raises.
    """
    original = {track: dict(quotas) for track, quotas in retrieval.STREAM_QUOTAS.items()}
    for quotas in retrieval.STREAM_QUOTAS.values():
        if stream in quotas:
            quotas[stream] = 0
    try:
        yield
    finally:
        for track, quotas in original.items():
            retrieval.STREAM_QUOTAS[track].clear()
            retrieval.STREAM_QUOTAS[track].update(quotas)


@contextmanager
def _department_filter_disabled():
    original = retrieval.DEPARTMENT_FILTER_ENABLED
    retrieval.DEPARTMENT_FILTER_ENABLED = False
    try:
        yield
    finally:
        retrieval.DEPARTMENT_FILTER_ENABLED = original


def _empty_aggregate() -> AggregateScore:
    return AggregateScore(n=0, hit_rate_at_10=0.0, mrr=0.0, mttc=0.0, efficiency=0.0)


@dataclass
class AblationResult:
    """One configuration's scored metrics, aggregate and sliced.

    Design doc §6.3/§6.4.

    Attributes:
        config_name: One of ABLATION_CONFIGS' `name` values.
        hit_rate_at_10: Aggregate Hit Rate@10 (§6.1). 0.0 for a "feature"
            kind config — not recomputed by the fast re-score path; read
            the config's `reads` entry in ABLATION_CONFIGS before trusting
            a field this config doesn't claim to measure.
        mrr: Aggregate MRR (§6.1).
        mttc: Aggregate MTTC (§6.1). Same caveat as hit_rate_at_10.
        efficiency: Aggregate token/usage cost (§6.1).
        by_scenario: scenario_type -> AggregateScore (§6.4).
        n: Number of sessions actually scored (may be smaller than 200 for
            a subset config like -rare_tag_match's "22% subset").
    """

    config_name: str
    hit_rate_at_10: float
    mrr: float
    mttc: float
    efficiency: float
    by_scenario: dict[str, "AggregateScore"]
    n: int


def _feature_ablation_transcripts(telemetry_rows: list[dict], ground_truth: dict, session_meta: dict, weights: dict) -> list[dict]:
    """Re-rank each session's last logged turn with `weights`, no re-retrieval.

    One "transcript" per session_id, shaped just enough for
    evaluate.score_transcripts() to consume: a single-turn transcript
    holding only the re-ranked `target_rank`. converged_turn/turns are
    deliberately absent (None/0) — this path recomputes MRR only, not
    Hit@10/MTTC (see module docstring).

    Args:
        telemetry_rows: telemetry.read_telemetry()'s output.
        ground_truth: session_id -> target parent_asin.
        session_meta: session_id -> {"scenario_type": ...}.
        weights: Feature name -> weight (a copy of HANDSET_WEIGHTS with
            one entry zeroed, by the caller).

    Returns:
        One transcript dict per session_id present in telemetry_rows.
    """
    last_turn_by_session: dict[str, dict] = {}
    for row in telemetry_rows:
        sid = row["session_id"]
        if sid not in last_turn_by_session or row["turn"] > last_turn_by_session[sid]["turn"]:
            last_turn_by_session[sid] = row

    transcripts = []
    for sid, row in last_turn_by_session.items():
        scored = []
        for cand in row.get("candidates", []):
            feats = cand.get("features", {})
            total = sum(weights.get(name, 0.0) * feats.get(name, 0.0) for name in FEATURE_NAMES)
            scored.append((cand.get("asin"), total))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top10 = [asin for asin, _ in scored[:10]]
        target = ground_truth.get(sid)
        target_rank = top10.index(target) + 1 if target is not None and target in top10 else None

        transcripts.append(
            {
                "session_id": sid,
                "scenario_type": session_meta.get(sid, {}).get("scenario_type"),
                "target_asin": target,
                "turns": row["turn"],
                "converged_turn": None,
                "transcript": [{"turn": row["turn"], "target_rank": target_rank}],
            }
        )
    return transcripts


def _rare_tag_match_subset(telemetry_rows: list[dict], ground_truth: dict) -> set[str]:
    """session_ids whose target candidate had a nonzero rare_tag_match in
    its last logged turn — "the 22% subset" §6.3 scopes this ablation to.
    """
    rare_tag_index = FEATURE_NAMES.index("rare_tag_match")
    subset = set()
    last_turn_by_session: dict[str, dict] = {}
    for row in telemetry_rows:
        sid = row["session_id"]
        if sid not in last_turn_by_session or row["turn"] > last_turn_by_session[sid]["turn"]:
            last_turn_by_session[sid] = row
    for sid, row in last_turn_by_session.items():
        target = ground_truth.get(sid)
        for cand in row.get("candidates", []):
            if cand.get("asin") == target:
                features = cand.get("features")
                value = (
                    features.get("rare_tag_match")
                    if isinstance(features, dict)
                    else (features[rare_tag_index] if features else None)
                )
                if value:
                    subset.add(sid)
                break
    return subset


def run_ablation(
    config_name: str,
    *,
    telemetry_rows: list[dict] | None = None,
    ground_truth: dict[str, str] | None = None,
    session_meta: dict[str, dict] | None = None,
    catalog_path: str = "data/catalog.jsonl",
    public_set_path: str = "data/public_set.jsonl",
) -> AblationResult:
    """Score one ablation configuration.

    Design doc §6.3: "Each configuration is run over all 200 public
    sessions and scored." Owner Marcus, step E5.

    Args:
        config_name: One of ABLATION_CONFIGS' `name` values.
        telemetry_rows, ground_truth, session_meta: Pre-loaded inputs for
            a "feature"-kind config's fast re-score path (avoids re-running
            the full corpus once per feature ablation when the caller —
            run_all_ablations() — already has them loaded). Required for
            "feature" kind, ignored for "baseline"/"stream"/"filter" (those
            call evaluate.record_baseline(), a full pass, themselves).
        catalog_path, public_set_path: Passed through to
            evaluate.record_baseline() for "baseline"/"stream"/"filter" kinds.

    Returns:
        An AblationResult.

    Raises:
        ValueError: `config_name` isn't in ABLATION_CONFIGS, or a
            "feature"-kind config is called without telemetry_rows/
            ground_truth/session_meta.
    """
    config = next((c for c in ABLATION_CONFIGS if c["name"] == config_name), None)
    if config is None:
        raise ValueError(f"unknown ablation config: {config_name!r}")

    if config["kind"] == "baseline":
        result = record_baseline(catalog_path=catalog_path, public_set_path=public_set_path)
        agg = result["baseline"]["overall"]
        return AblationResult(
            config_name=config_name,
            hit_rate_at_10=agg.hit_rate_at_10,
            mrr=agg.mrr,
            mttc=agg.mttc,
            efficiency=agg.efficiency,
            by_scenario=result["baseline"]["by_scenario"],
            n=agg.n,
        )

    if config["kind"] == "stream":
        stream = config_name.lstrip("-").removesuffix("_stream")
        with _stream_disabled(stream):
            result = record_baseline(catalog_path=catalog_path, public_set_path=public_set_path)
        agg = result["baseline"]["overall"]
        return AblationResult(
            config_name=config_name,
            hit_rate_at_10=agg.hit_rate_at_10,
            mrr=agg.mrr,
            mttc=agg.mttc,
            efficiency=agg.efficiency,
            by_scenario=result["baseline"]["by_scenario"],
            n=agg.n,
        )

    if config["kind"] == "filter":
        with _department_filter_disabled():
            result = record_baseline(catalog_path=catalog_path, public_set_path=public_set_path)
        agg = result["baseline"]["overall"]
        return AblationResult(
            config_name=config_name,
            hit_rate_at_10=agg.hit_rate_at_10,
            mrr=agg.mrr,
            mttc=agg.mttc,
            efficiency=agg.efficiency,
            by_scenario=result["baseline"]["by_scenario"],
            n=agg.n,
        )

    # kind == "feature"
    if config_name == "-llm_rerank":
        # llm_rerank() (C7) is a stub that always returns None -> rank()
        # always falls back to the regression ordering already, so this
        # ablation's real answer is "identical to baseline, computed
        # without re-running anything" — not a fixture placeholder, the
        # documented actual behavior of a stub with no LLM access.
        return AblationResult(
            config_name=config_name, hit_rate_at_10=0.0, mrr=0.0, mttc=0.0, efficiency=0.0, by_scenario={}, n=0
        )

    if telemetry_rows is None or ground_truth is None:
        raise ValueError(f"{config_name!r} is a 'feature'-kind config: telemetry_rows/ground_truth are required")
    session_meta = session_meta or {}

    feature_name, subset_only = _FEATURE_ABLATIONS[config_name]
    weights = dict(HANDSET_WEIGHTS)
    weights[feature_name] = 0.0

    rows = telemetry_rows
    if subset_only:
        subset = _rare_tag_match_subset(telemetry_rows, ground_truth)
        rows = [r for r in telemetry_rows if r["session_id"] in subset]

    transcripts = _feature_ablation_transcripts(rows, ground_truth, session_meta, weights)
    scored = score_transcripts(transcripts)
    agg = scored["overall"]
    return AblationResult(
        config_name=config_name,
        hit_rate_at_10=agg.hit_rate_at_10,
        mrr=agg.mrr,
        mttc=agg.mttc,
        efficiency=agg.efficiency,
        by_scenario=scored["by_scenario"],
        n=agg.n,
    )


def slice_by_scenario(result: AblationResult, scenario_type: str) -> "AggregateScore":
    """One ablation result's metrics for a single scenario_type slice (§6.4).

    Args:
        result: An AblationResult from run_ablation().
        scenario_type: One of SCENARIO_SLICES.

    Returns:
        That slice's AggregateScore, or an all-zero one if this result has
        no sessions in that slice.
    """
    return result.by_scenario.get(scenario_type, _empty_aggregate())


def run_all_ablations(
    catalog_path: str = "data/catalog.jsonl", public_set_path: str = "data/public_set.jsonl"
) -> list[AblationResult]:
    """Run every configuration in ABLATION_CONFIGS and collect results.

    Design doc §6.3: "Nine configurations x four metrics, plus four
    slices. The core evidence in the writeup." Owner Marcus, step E6.

    Loads telemetry/ground-truth/session-meta once (from a fresh baseline
    run) and reuses them for every "feature"-kind config's fast re-score,
    rather than re-running the full corpus nine times; "baseline"/"stream"/
    "filter" kinds each do their own full pass (they change pool
    composition, so logged telemetry from a different config can't be reused).

    Args:
        catalog_path, public_set_path: Passed through to every full-pass
            config; the "feature"-kind fast path reuses one such run's
            telemetry rather than calling these again.

    Returns:
        One AblationResult per entry in ABLATION_CONFIGS, in that order.
    """
    from telemetry import read_telemetry, run_instrumented_corpus

    corpus = run_instrumented_corpus(catalog_path=catalog_path, public_set_path=public_set_path)
    telemetry_rows = read_telemetry("data/telemetry.jsonl")
    ground_truth = {t["session_id"]: t["target_asin"] for t in corpus["transcripts"] if t.get("target_asin")}
    session_meta = {t["session_id"]: {"scenario_type": t.get("scenario_type")} for t in corpus["transcripts"]}

    results = []
    for config in ABLATION_CONFIGS:
        if config["kind"] == "feature":
            results.append(
                run_ablation(
                    config["name"],
                    telemetry_rows=telemetry_rows,
                    ground_truth=ground_truth,
                    session_meta=session_meta,
                )
            )
        else:
            results.append(run_ablation(config["name"], catalog_path=catalog_path, public_set_path=public_set_path))
    return results
