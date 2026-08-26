"""Ablation harness: per-component configs, scored over all 200 sessions.

Design doc §6.3 (Ablation protocol) and §6.4 (Slicing).

Owner: Chellpapan (Evaluation and integration). §8.5, step E5 ("Config flags +
evaluator. Feature ablations re-score logged candidates without re-running
retrieval; stream and filter ablations trigger a full pass."), step E6
("Populate the ablation table; slice by scenario_type.").

Everything below is a stub. Function bodies return fixture values only.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class AblationResult:
    """One configuration's scored metrics, aggregate and sliced.

    Design doc §6.3/§6.4.

    Attributes:
        config_name: One of ABLATION_CONFIGS' `name` values.
        hit_rate_at_10: Aggregate Hit Rate@10 (§6.1).
        mrr: Aggregate MRR (§6.1).
        mttc: Aggregate MTTC (§6.1).
        efficiency: Aggregate token/usage cost (§6.1).
        by_scenario: scenario_type -> {metric_name: value} (§6.4).
    """

    config_name: str
    hit_rate_at_10: float
    mrr: float
    mttc: float
    efficiency: float
    by_scenario: dict[str, dict[str, float]]


def run_ablation(config_name: str, sessions: list[dict]) -> AblationResult:
    """Score one ablation configuration over the given sessions.

    Design doc §6.3: "Each configuration is run over all 200 public
    sessions and scored with the supplied evaluator." Owner Chellpapan, step E5.

    STUB: does not run the evaluator or re-score anything; returns fixture
    metrics tagged with `config_name` and the session count.

    Args:
        config_name: One of ABLATION_CONFIGS' `name` values.
        sessions: The public sessions to score over (§6.3: all 200).

    Returns:
        An AblationResult with fixture metric values.
    """
    n = len(sessions)
    return AblationResult(
        config_name=config_name,
        hit_rate_at_10=0.0,
        mrr=0.0,
        mttc=0.0,
        efficiency=0.0,
        by_scenario={slice_name: {"hit_rate_at_10": 0.0, "mrr": 0.0, "n": 0} for slice_name in SCENARIO_SLICES},
    )


def slice_by_scenario(results: list[dict], scenario_type: str) -> list[dict]:
    """Filter scored session results down to one scenario_type slice.

    Design doc §6.4: "Every metric is additionally reported by
    scenario_type. Aggregate scores conceal which capability is failing."

    STUB: returns `results` unfiltered rather than checking each row's
    scenario_type field.

    Args:
        results: Per-session scored results (session_id, scenario_type,
            metrics...).
        scenario_type: One of SCENARIO_SLICES.

    Returns:
        The subset of `results` matching `scenario_type`.
    """
    return results


def run_all_ablations(sessions: list[dict]) -> list[AblationResult]:
    """Run every configuration in ABLATION_CONFIGS and collect results.

    Design doc §6.3: "Nine configurations x four metrics, plus four
    slices. The core evidence in the writeup." Owner Chellpapan, step E6.

    Args:
        sessions: The public sessions to score over.

    Returns:
        One AblationResult per entry in ABLATION_CONFIGS.
    """
    return [run_ablation(config["name"], sessions) for config in ABLATION_CONFIGS]
