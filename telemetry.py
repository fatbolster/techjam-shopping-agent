"""Append-only JSONL telemetry: the only source of ranker training data.

Design doc §3.4 Step 7 (Telemetry), §3.3 ("Telemetry log ... Append-only"),
§6.6 (Ranker training protocol) and §8.4:

* step D2 — append-only telemetry: per-turn state, canonical intent string,
  pool sizes, per-candidate features and stream provenance.
* step D6 — own loop over reset()/respond() producing the feature matrix
  (`run_instrumented_corpus`, driven by `simulate.run_session`).
* step D7 — per-stream recall report (`per_stream_recall_report`), which
  establishes the pool-recall ceiling that bounds Hit Rate@10.

Owner: Chellappan (Simulator and training corpus). §8.4.

Ground-truth labels are joined *offline* by `session_id` (and `scenario_type`
for slicing, likewise joined offline). Nothing here is passed into
`respond()` — `log_turn` records only what the agent already computed for its
own ranking, plus each candidate's stream `sources`, from which
"which streams contained the target" is derived after the run.

WORKING ASSUMPTION — configuration C (§6.5.1): see `simulate.py`.
"""

from __future__ import annotations

import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from features import FEATURE_NAMES, feature_vector
from indexes import Indexes
from state import SessionState
from utils import Candidate

DEFAULT_TELEMETRY_PATH = "data/telemetry.jsonl"
DEFAULT_FEATURE_MATRIX_PATH = "data/features.jsonl"

# The active telemetry file. `log_turn()` writes here whenever it is called
# without an explicit `path` — which is how `agent.respond()` calls it. The
# instrumented run (and tests) redirect it via `telemetry_path(...)` so the
# `log_turn` calls buried inside `respond()` land somewhere the runner
# controls, without `agent.py` needing a path parameter it does not have.
_ACTIVE_TELEMETRY_PATH = DEFAULT_TELEMETRY_PATH


@contextmanager
def telemetry_path_ctx(path: str) -> Generator[str, None, None]:
    """Redirect `log_turn`'s default output to `path` for the duration."""
    global _ACTIVE_TELEMETRY_PATH
    previous = _ACTIVE_TELEMETRY_PATH
    _ACTIVE_TELEMETRY_PATH = path
    try:
        yield path
    finally:
        _ACTIVE_TELEMETRY_PATH = previous

# Negatives sampled per *turn* from that turn's candidate pool (§8.4 STEP 3:
# "Sample ~20 negatives PER TURN from the candidate pool"), never at random
# from the catalogue — random catalogue negatives are trivially separable and
# the model would learn only "popular = good" (§6.6 "Why negatives are
# sampled from the pool").
NEGATIVES_PER_TURN = 20

# Slots that authorise irreversible filtering (§2.3 well-populated structured
# fields; §3.4 Step 3 routing rule). `n_hard_slots` counts these so the
# corpus / router diagnostics can key on the same definition the router uses.
HARD_SLOT_KEYS: tuple[str, ...] = (
    "department",
    "category",
    "brand",
    "price_min",
    "price_max",
    "price_target",
)


def n_hard_slots(state: Optional[SessionState]) -> int:
    """Number of filter-authorising slots currently set (§3.4 Step 3)."""
    if state is None:
        return 0
    return sum(1 for key in HARD_SLOT_KEYS if state.slots.get(key))


def log_turn(
    session_id: str,
    turn: int,
    candidates: list[Candidate],
    state: Optional[SessionState] = None,
    indexes: Optional[Indexes] = None,
    ask_attribute: Optional[str] = None,
    target_in_pool: Optional[bool] = None,
    pool_size_before: Optional[int] = None,
    path: Optional[str] = None,
) -> None:
    """Append one turn's telemetry row to the JSONL log.

    Design doc §7.2 interface contract `log_turn(session_id, turn,
    candidates) -> None`; §3.4 Step 7 record contents. Owner Chellappan, D2.

    One row per turn (not per session): features change as slots accumulate,
    so turn 1 and turn 4 are different data (§8.4 STEP 3). Each candidate
    carries its ten feature values in `FEATURE_NAMES` order and the set of
    streams that surfaced it (`sources`); "which streams contained the
    target" is computed from the latter offline, so no label is needed here.

    `pool_size_before` (pre-filter pool size) is accepted but currently
    `None`: `retrieval.retrieve()` (Owner A) returns only the final pool. It
    is wired through so a later `retrieve()` that surfaces both sizes needs
    no telemetry change. `pool_size` is the post-filter size and is the
    signal that actually bounds recall.

    Args:
        session_id: Session identifier (joined to ground truth offline).
        turn: 1-indexed turn number.
        candidates: The turn's final candidate pool.
        state: Current session state, for track / canonical intent /
            n_hard_slots.
        indexes: Offline indexes bundle, for per-candidate feature extraction.
        ask_attribute: The attribute asked this turn, if any.
        target_in_pool: Optional diagnostic, computed by the caller offline
            (never from a label available at inference); usually left None
            and derived later by `per_stream_recall_report`.
        pool_size_before: Optional pre-filter pool size (see above).
        path: JSONL file to append to; defaults to the active telemetry path
            (see `telemetry_path_ctx`).
    """
    path = path or _ACTIVE_TELEMETRY_PATH
    record = {
        "session_id": session_id,
        "turn": turn,
        "track": state.track if state is not None else None,
        "canonical_intent": state.canonical_intent if state is not None else None,
        "n_hard_slots": n_hard_slots(state),
        "pool_size_before": pool_size_before,
        "pool_size": len(candidates),
        "ask_attribute": ask_attribute,
        "target_in_pool": target_in_pool,
        "feature_names": list(FEATURE_NAMES),
        "candidates": [
            {
                "asin": cand.asin,
                "sources": sorted(cand.sources),
                "features": (
                    list(feature_vector(cand, candidates, indexes, state))
                    if indexes is not None and state is not None
                    else [None] * len(FEATURE_NAMES)
                ),
            }
            for cand in candidates
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_telemetry(path: str = DEFAULT_TELEMETRY_PATH) -> list[dict]:
    """Read back every logged turn, in file order (empty list if absent)."""
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _seeded_rng(*parts: object) -> random.Random:
    """A reproducible RNG keyed by `parts` (so a re-run samples identically)."""
    return random.Random("|".join(str(p) for p in parts))


def sample_pool_negatives(
    candidates: list[dict],
    target_asin: Optional[str],
    rng: random.Random,
    k: int = NEGATIVES_PER_TURN,
) -> list[dict]:
    """Pick up to `k` non-target candidates from one turn's pool.

    Design doc §6.6 step 4 / §8.4 STEP 3. Sampling is without replacement
    from the pool itself; the target (if present) is excluded.
    """
    pool = [c for c in candidates if c.get("asin") != target_asin]
    if len(pool) <= k:
        return list(pool)
    return rng.sample(pool, k)


def build_training_rows(
    telemetry_rows: list[dict],
    ground_truth: dict[str, str],
    negatives_per_turn: int = NEGATIVES_PER_TURN,
    session_meta: Optional[dict[str, dict]] = None,
) -> list[dict]:
    """Join telemetry against ground truth into labelled training rows.

    Design doc §6.6 steps 2-4, §8.4 STEP 3 row schema:
    `session_id, turn, n_hard_slots, parent_asin, [ten features in
    FEATURE_NAMES order], label`. One positive (the target, when it is in
    that turn's pool) plus ~`negatives_per_turn` pool negatives per turn.

    Args:
        telemetry_rows: Output of `read_telemetry()`.
        ground_truth: session_id -> target parent_asin.
        negatives_per_turn: Pool negatives to sample per turn.
        session_meta: Optional session_id -> {"scenario_type": ...}, joined
            offline for slicing; added to each row as `scenario_type` when
            available.

    Returns:
        One dict per (session, turn, candidate) with `features` as a list of
        ten floats in FEATURE_NAMES order and `label` (1 for the target).
    """
    session_meta = session_meta or {}
    rows: list[dict] = []
    for turn_row in telemetry_rows:
        session_id = turn_row["session_id"]
        turn = turn_row["turn"]
        target = ground_truth.get(session_id)
        cands = turn_row.get("candidates", [])
        by_asin = {c.get("asin"): c for c in cands}

        selected: list[tuple[dict, int]] = []
        if target is not None and target in by_asin:
            selected.append((by_asin[target], 1))
        rng = _seeded_rng(session_id, turn)
        for neg in sample_pool_negatives(cands, target, rng, negatives_per_turn):
            selected.append((neg, 0))

        scenario_type = session_meta.get(session_id, {}).get("scenario_type")
        for cand, label in selected:
            row = {
                "session_id": session_id,
                "turn": turn,
                "n_hard_slots": turn_row.get("n_hard_slots"),
                "parent_asin": cand.get("asin"),
                "features": cand.get("features"),
                "label": label,
            }
            if scenario_type is not None:
                row["scenario_type"] = scenario_type
            rows.append(row)
    return rows


def write_feature_matrix(rows: list[dict], path: str = DEFAULT_FEATURE_MATRIX_PATH) -> None:
    """Write the training corpus as JSONL, overwriting any prior matrix."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def per_stream_recall_report(
    telemetry_rows: list[dict],
    ground_truth: dict[str, str],
    session_meta: Optional[dict[str, dict]] = None,
) -> dict:
    """Per-stream recall + the pool-recall ceiling (§8.4 D7, §6.2).

    For every logged turn: was the target in the pool, and if so which
    stream(s) surfaced it (and did any single stream do so *uniquely*).
    "If pool recall is 0.35, no amount of ranking work helps" (§6.2) — this
    report is what tells you which half of the pipeline to work on.

    Returns a dict with an `overall` block and a `by_scenario` block (the
    latter only populated where `session_meta` supplies `scenario_type`).
    """
    session_meta = session_meta or {}
    streams = ("keyword", "semantic", "popularity")

    def blank() -> dict:
        return {
            "turns": 0,
            "turns_with_target": 0,
            "present": {s: 0 for s in streams},
            "unique": {s: 0 for s in streams},
        }

    overall = blank()
    by_scenario: dict[str, dict] = {}

    for turn_row in telemetry_rows:
        session_id = turn_row["session_id"]
        target = ground_truth.get(session_id)
        scenario = session_meta.get(session_id, {}).get("scenario_type")
        buckets = [overall]
        if scenario is not None:
            by_scenario.setdefault(scenario, blank())
            buckets.append(by_scenario[scenario])

        for bucket in buckets:
            bucket["turns"] += 1

        if target is None:
            continue
        target_cand = next(
            (c for c in turn_row.get("candidates", []) if c.get("asin") == target),
            None,
        )
        if target_cand is None:
            continue
        sources = set(target_cand.get("sources", []))
        for bucket in buckets:
            bucket["turns_with_target"] += 1
            for s in streams:
                if s in sources:
                    bucket["present"][s] += 1
            if len(sources) == 1:
                (only,) = tuple(sources)
                if only in bucket["unique"]:
                    bucket["unique"][only] += 1

    def finalize(bucket: dict) -> dict:
        turns = bucket["turns"] or 1
        bucket["pool_recall"] = bucket["turns_with_target"] / turns
        return bucket

    finalize(overall)
    for bucket in by_scenario.values():
        finalize(bucket)
    return {"overall": overall, "by_scenario": by_scenario}


def format_recall_report(report: dict) -> str:
    """Render `per_stream_recall_report` output as a readable text block (D7).

    Design doc §8.4 D7 / §6.2: shows the pool-recall ceiling and, per stream,
    how often it carried the target and how often it was the *only* stream
    that did (its unique catches).
    """
    streams = ("keyword", "semantic", "popularity")
    lines: list[str] = []

    def block(title: str, bucket: dict) -> None:
        turns = bucket["turns"]
        with_target = bucket["turns_with_target"]
        lines.append(
            f"{title}: pool recall {bucket['pool_recall']:.3f} "
            f"({with_target}/{turns} turns had the target in the pool)"
        )
        denom = with_target or 1
        for s in streams:
            present = bucket["present"][s]
            unique = bucket["unique"][s]
            lines.append(
                f"    {s:<11} present {present:>4}/{with_target} "
                f"({present / denom:.2f})   unique {unique:>4}"
            )

    block("overall", report["overall"])
    for scenario in sorted(report.get("by_scenario", {})):
        lines.append("")
        block(scenario, report["by_scenario"][scenario])
    return "\n".join(lines)


def format_transcript(session_result: dict) -> str:
    """Render one `simulate.run_session` result as a plain-text transcript (D8)."""
    header = (
        f"=== {session_result.get('session_id')}  "
        f"[{session_result.get('scenario_type')}]  "
        f"target={session_result.get('target_asin')}  "
        f"turns={session_result.get('turns')}  "
        f"converged={session_result.get('converged_turn')}"
    )
    lines = [header]
    for step in session_result.get("transcript", []):
        lines.append(f"  turn {step['turn']}  user: {step['user']}")
        if step.get("ask_attribute"):
            lines.append(f"          agent asked -> {step['ask_attribute']}")
        rank = step.get("target_rank")
        rank_text = f"target #{rank}" if rank is not None else "target not in top-k"
        lines.append(f"          {rank_text}  |  returned: {', '.join(step.get('recommended', []))}")
    return "\n".join(lines)


def export_transcripts(
    session_results: list[dict],
    path: str,
    scenario_types: tuple[str, ...] = ("intent_override", "boundary"),
) -> int:
    """Write the plain-text transcripts for the named scenario types (D8).

    Design doc §8.4 D8: "Circulate override and boundary transcripts ...
    Feeds B5's negation patterns and E's answerability priors." Returns the
    number of sessions written.
    """
    selected = [
        r for r in session_results if r.get("scenario_type") in scenario_types
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# transcripts for {', '.join(scenario_types)} "
            f"({len(selected)} sessions)\n\n"
        )
        f.write("\n\n".join(format_transcript(r) for r in selected))
        f.write("\n")
    return len(selected)


def run_instrumented_corpus(
    catalog_path: str = "data/catalog.jsonl",
    public_set_path: str = "data/public_set.jsonl",
    *,
    telemetry_path: str = DEFAULT_TELEMETRY_PATH,
    feature_matrix_path: str = DEFAULT_FEATURE_MATRIX_PATH,
    transcript_path: str = "data/transcripts.txt",
    max_turns: int = 10,
    limit: Optional[int] = None,
) -> dict:
    """Run every public session through the simulator and build the corpus.

    Design doc §8.4 D6 ("Own loop over reset()/respond()"). Steps:

    1. Load the catalogue and `public_set.jsonl`.
    2. Build one `Agent` (offline indexes once).
    3. For each session, `simulate.run_session()` drives the conversation;
       the `log_turn` call inside `respond()` writes one telemetry row per
       turn to `telemetry_path` (truncated first so a re-run is clean).
    4. Join `ground_truth` and `scenario_type` offline by `session_id`.
    5. Build the labelled feature matrix (pool negatives per turn) and write
       it to `feature_matrix_path`.
    6. Produce the per-stream recall report (D7).
    7. Export the intent_override / boundary transcripts (D8).

    Imports `agents.our_agent` / `simulate` lazily to avoid an import cycle
    (`agents.our_agent` imports `telemetry`).

    Returns `{sessions, turns, feature_rows, recall, transcripts,
    transcripts_exported}`.
    """
    from agents.our_agent import Agent
    from simulate import build_catalog_index, load_jsonl, run_session

    catalog = load_jsonl(catalog_path)
    if not catalog:
        from utils import load_catalog

        catalog = load_catalog(catalog_path)
    catalog_index = build_catalog_index(catalog)

    sessions = load_jsonl(public_set_path)
    if limit is not None:
        sessions = sessions[:limit]

    # Truncate the telemetry log so this run's rows are the only ones read
    # back, then redirect the log_turn calls inside respond() to it.
    Path(telemetry_path).parent.mkdir(parents=True, exist_ok=True)
    Path(telemetry_path).write_text("", encoding="utf-8")

    ground_truth: dict[str, str] = {}
    session_meta: dict[str, dict] = {}
    transcripts: list[dict] = []
    with telemetry_path_ctx(telemetry_path):
        agent = Agent(catalog_path)
        for session in sessions:
            session_id = session.get("sample_id") or session.get("session_id")
            asin = None
            gt = session.get("ground_truth")
            if isinstance(gt, dict):
                asin = gt.get("parent_asin")
            if session_id is not None and asin is not None:
                ground_truth[session_id] = asin
            if session_id is not None:
                session_meta[session_id] = {
                    "scenario_type": session.get("scenario_type")
                }
            transcripts.append(
                run_session(agent, session, catalog_index, max_turns=max_turns)
            )

    telemetry_rows = read_telemetry(telemetry_path)
    feature_rows = build_training_rows(
        telemetry_rows, ground_truth, session_meta=session_meta
    )
    write_feature_matrix(feature_rows, feature_matrix_path)
    recall = per_stream_recall_report(telemetry_rows, ground_truth, session_meta)
    transcripts_exported = export_transcripts(transcripts, transcript_path)

    return {
        "sessions": len(sessions),
        "turns": len(telemetry_rows),
        "feature_rows": len(feature_rows),
        "recall": recall,
        "transcripts": transcripts,
        "transcripts_exported": transcripts_exported,
    }


if __name__ == "__main__":
    summary = run_instrumented_corpus()
    print(
        f"sessions={summary['sessions']} turns={summary['turns']} "
        f"feature_rows={summary['feature_rows']}"
    )
    print("pool recall:", summary["recall"]["overall"]["pool_recall"])
