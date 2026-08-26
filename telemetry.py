"""Append-only JSONL telemetry: the only source of ranker training data.

Design doc §3.4 Step 7 (Telemetry) and §3.3 ("Telemetry log ... Append-only").

Owner: Chellappan (Simulator and training corpus). §8.4, step D2 ("Logs state,
canonical intent string, pool sizes, and whether the target was in the
pool. Labels joined offline by session_id; ground truth never reaches
respond()."), step D6 (produce the ~4,200-row feature matrix), step D7
(per-stream recall report).

Everything below is a stub. Function bodies return fixture values only, or
perform file I/O whose *shape* matches the design (append-only JSONL) but
whose *content* is a fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from features import FEATURE_NAMES, extract_features
from indexes import Indexes
from state import SessionState
from utils import Candidate

DEFAULT_TELEMETRY_PATH = "data/telemetry.jsonl"

# Negatives sampled per session from the candidate pool, not the catalogue
# at large (§6.6 step 4, §6.6 "Why negatives are sampled from the pool").
NEGATIVES_PER_SESSION = 20


def log_turn(
    session_id: str,
    turn: int,
    candidates: list[Candidate],
    state: Optional[SessionState] = None,
    indexes: Optional[Indexes] = None,
    ask_attribute: Optional[str] = None,
    target_in_pool: Optional[bool] = None,
    path: str = DEFAULT_TELEMETRY_PATH,
) -> None:
    """Append one turn's telemetry row to the JSONL log.

    Design doc §7.2 interface contract: `log_turn(session_id, turn,
    candidates) -> None`. §3.4 Step 7: "Records session_id, turn, track,
    canonical intent string, pool size before and after filtering,
    ask_attribute, whether the target was present in the pool, and the ten
    feature values per candidate." Owner Chellappan, step D2.

    Ground truth is never passed into respond() (§3.4 Step 7); the
    `target_in_pool` boolean is computed by the *caller* offline or during
    an instrumented run, never derived from a label available at inference.

    STUB: writes one real JSON line via the real append-only mechanism
    (this is simple, unambiguous I/O, not a design decision), but the
    per-candidate feature values inside it come from features.py's stubs.

    Args:
        session_id: Session identifier.
        turn: 1-indexed turn number.
        candidates: The turn's final candidate pool.
        state: Current session state, for track/canonical_intent/slots.
        indexes: Offline indexes bundle, for feature extraction.
        ask_attribute: The attribute asked this turn, if any.
        target_in_pool: Whether the session's ground-truth ASIN was in
            `candidates` (diagnostic only, §6.2; None if unknown/live).
        path: JSONL file to append to.

    Returns:
        None.
    """
    record = {
        "session_id": session_id,
        "turn": turn,
        "track": state.track if state is not None else None,
        "canonical_intent": state.canonical_intent if state is not None else None,
        "pool_size": len(candidates),
        "ask_attribute": ask_attribute,
        "target_in_pool": target_in_pool,
        "candidates": [
            {
                "asin": cand.asin,
                "features": (
                    extract_features(cand, candidates, indexes, state)
                    if indexes is not None and state is not None
                    else {name: None for name in FEATURE_NAMES}
                ),
            }
            for cand in candidates
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_telemetry(path: str = DEFAULT_TELEMETRY_PATH) -> list[dict]:
    """Read back every logged turn.

    Design doc §6.5 ("training data for the ranker must be produced by an
    instrumented run rather than read from disk") and §6.6 step 2.
    Chellappan, supports step D6/D7.

    STUB: reads whatever is actually on disk at `path`; returns an empty
    list if the file does not exist, rather than raising.

    Args:
        path: JSONL file to read.

    Returns:
        One dict per logged turn, in file order.
    """
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_training_rows(telemetry_rows: list[dict], ground_truth: dict[str, str]) -> list[dict]:
    """Join telemetry against ground truth to build labelled training rows.

    Design doc §6.6 Ranker training protocol, steps 2-4: "Each turn logs
    one row per candidate ... Join ground_truth offline by session_id to
    assign labels ... Sample ~20 negatives per session from the candidate
    pool, not from the catalogue at large." Owner Chellappan, step D6 (BLOCKING —
    "~4,200 labelled rows delivered to C").

    STUB: joins for real (a simple dict lookup + list comprehension, not a
    design decision) but does not implement the ~20-negatives-per-session
    sampling; returns every candidate as a row, unsampled.

    Args:
        telemetry_rows: Output of read_telemetry().
        ground_truth: session_id -> target parent_asin.

    Returns:
        One dict per (session, turn, candidate) with a `label` field
        (1 for the target, 0 otherwise).
    """
    rows = []
    for turn_row in telemetry_rows:
        target = ground_truth.get(turn_row["session_id"])
        for cand in turn_row["candidates"]:
            rows.append(
                {
                    "session_id": turn_row["session_id"],
                    "turn": turn_row["turn"],
                    "parent_asin": cand["asin"],
                    "features": cand["features"],
                    "label": int(cand["asin"] == target),
                }
            )
    return rows
