"""Our own local scorer: Hit Rate@10, MRR, MTTC, Efficiency (§6.1).

Design doc §6.1 (Metric definitions) and §8.5 step E3 ("Stand up the
evaluator; record the baseline score").

Owner: Marcus (Evaluation and integration).

This is deliberately NOT the organizer-supplied evaluator — that kit lives
in kit/ (gitignored) and, per D1 (§6.5.1), was never present in this repo
until the team supplies it. "Headline figures come only from the supplied
evaluator" (§8.4 D7's note) still holds; this module exists so a baseline
number and the ablation harness (E5/E6) don't have to sit idle until the
kit lands, and every formula here is copied verbatim from §6.1, not
guessed — swapping in the real evaluator later is a matter of pointing
run_ablation()/record_baseline() at its scoring call instead of
score_transcripts(), not redesigning this module.

Input shape: one dict per session, as returned by
simulate.run_session()/telemetry.run_instrumented_corpus()'s "transcripts"
list — {session_id, scenario_type, target_asin, turns, converged_turn,
transcript: [{turn, ask_attribute, recommended, target_rank}, ...]}.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# §1.2: "Max 10 turns; exceeding scores zero." Sessions that never converge
# within this many turns score 0 on MTTC (§6.1: "Sessions exceeding 10
# turns terminate with zero score.").
MAX_TURNS = 10


@dataclass
class ScoredSession:
    """One session's four §6.1 metrics.

    Attributes:
        session_id: The session's identifier.
        scenario_type: buying/browsing/intent_override/boundary, for
            §6.4's per-slice reporting.
        hit: Whether the target appears in the final turn's top 10
            (Hit Rate@10's per-session indicator).
        reciprocal_rank: 1/rank of the target in the final turn's
            recommendations, or 0.0 if absent (MRR's per-session term).
        turns_to_conversion: The turn `converged_turn` first held, or 0.0
            if the session never converged within MAX_TURNS (§6.1's "zero
            score" rule — not `MAX_TURNS` itself, an unconverged session
            contributes nothing, not a penalty proportional to length).
        efficiency: Token/usage cost for this session (§6.1: "derived from
            the usage object ... our pipeline reports zero tokens with LLM
            components disabled").
    """

    session_id: str
    scenario_type: Optional[str]
    hit: bool
    reciprocal_rank: float
    turns_to_conversion: float
    efficiency: float = 0.0


def score_session(session_result: dict) -> ScoredSession:
    """Score one session's transcript against the four §6.1 metrics.

    Args:
        session_result: One entry from run_instrumented_corpus()'s
            "transcripts" list (see module docstring for the shape).

    Returns:
        A ScoredSession.
    """
    transcript = session_result.get("transcript") or []
    final_rank = transcript[-1].get("target_rank") if transcript else None
    # target_rank is always 1-10 or None in practice (simulate.run_session()
    # computes it as recommended.index(target)+1 against a top_k=10 list,
    # so "beyond top 10" and "absent" are the same observable state: None)
    # — the explicit <= 10 check below is defensive, not load-bearing, in
    # case a future caller's rank isn't pre-bounded to top-10 the same way.
    hit = final_rank is not None and final_rank <= 10
    reciprocal_rank = (1.0 / final_rank) if hit else 0.0

    converged_turn = session_result.get("converged_turn")
    turns_to_conversion = (
        float(converged_turn) if converged_turn is not None and converged_turn <= MAX_TURNS else 0.0
    )

    return ScoredSession(
        session_id=session_result.get("session_id", ""),
        scenario_type=session_result.get("scenario_type"),
        hit=hit,
        reciprocal_rank=reciprocal_rank,
        turns_to_conversion=turns_to_conversion,
        efficiency=float(session_result.get("efficiency", 0.0)),
    )


@dataclass
class AggregateScore:
    """Metrics averaged over N sessions (§6.1's (1/N) Σ formulas).

    Attributes:
        n: Number of sessions this aggregate covers.
        hit_rate_at_10: (1/N) Σ 1[target_i ∈ top10_i].
        mrr: (1/N) Σ 1/rank_i.
        mttc: (1/N) Σ turns_to_conversion_i.
        efficiency: (1/N) Σ efficiency_i.
    """

    n: int
    hit_rate_at_10: float
    mrr: float
    mttc: float
    efficiency: float


def _aggregate(scored: list[ScoredSession]) -> AggregateScore:
    n = len(scored)
    if n == 0:
        return AggregateScore(n=0, hit_rate_at_10=0.0, mrr=0.0, mttc=0.0, efficiency=0.0)
    return AggregateScore(
        n=n,
        hit_rate_at_10=sum(1.0 for s in scored if s.hit) / n,
        mrr=sum(s.reciprocal_rank for s in scored) / n,
        mttc=sum(s.turns_to_conversion for s in scored) / n,
        efficiency=sum(s.efficiency for s in scored) / n,
    )


def score_transcripts(transcripts: list[dict]) -> dict:
    """Score every session and aggregate overall + by scenario_type (§6.4).

    Args:
        transcripts: run_instrumented_corpus()'s "transcripts" list (every
            public session, not just intent_override/boundary — those are
            D8's separate, filtered plain-text export).

    Returns:
        {"overall": AggregateScore, "by_scenario": {scenario_type:
        AggregateScore}}.
    """
    scored = [score_session(s) for s in transcripts]
    by_scenario: dict[str, list[ScoredSession]] = defaultdict(list)
    for s in scored:
        if s.scenario_type is not None:
            by_scenario[s.scenario_type].append(s)

    return {
        "overall": _aggregate(scored),
        "by_scenario": {scenario: _aggregate(rows) for scenario, rows in by_scenario.items()},
    }


def record_baseline(
    catalog_path: str = "data/catalog.jsonl", public_set_path: str = "data/public_set.jsonl"
) -> dict:
    """Run the full pipeline over every public session and score it.

    Design doc §8.5 step E3: "Stand up the evaluator; record the baseline
    score." Runs telemetry.run_instrumented_corpus() (D6) for the
    transcripts, then scores them here — see the module docstring for why
    this is our own scorer, not the organizer-supplied one.

    Args:
        catalog_path: Path to catalog.jsonl.
        public_set_path: Path to public_set.jsonl.

    Returns:
        {"baseline": score_transcripts()'s result, "recall":
        run_instrumented_corpus()'s per-stream recall report (§8.4 D7) —
        the internal diagnostic §6.2 says to read alongside any score:
        "if pool recall is 0.55 and Hit@10 is 0.30, ranking is the
        bottleneck"}.
    """
    from telemetry import run_instrumented_corpus

    result = run_instrumented_corpus(catalog_path=catalog_path, public_set_path=public_set_path)
    return {
        "baseline": score_transcripts(result["transcripts"]),
        "recall": result["recall"],
    }


if __name__ == "__main__":
    summary = record_baseline()
    overall = summary["baseline"]["overall"]
    print(
        f"n={overall.n} hit_rate_at_10={overall.hit_rate_at_10:.3f} "
        f"mrr={overall.mrr:.3f} mttc={overall.mttc:.3f} efficiency={overall.efficiency:.3f}"
    )
    print("pool_recall:", summary["recall"]["overall"]["pool_recall"])
    for scenario, agg in summary["baseline"]["by_scenario"].items():
        print(
            f"  {scenario}: n={agg.n} hit_rate_at_10={agg.hit_rate_at_10:.3f} "
            f"mrr={agg.mrr:.3f} mttc={agg.mttc:.3f}"
        )
