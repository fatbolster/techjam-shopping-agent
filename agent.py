"""The Agent: wires every stub into a runnable reset()/respond() loop.

Design doc §4 (System diagram, the full per-turn pipeline) and §7.2's
interface-contract sketch. The exact `reset`/`respond` signatures below
follow the supplied starter kit's baseline agent (§5.1: "The supplied
starter agent conflates retrieval and ranking — it issues one FTS5 query
with LIMIT top_k and returns that order directly"), which the evaluator
drives directly, matched here exactly (no added/removed/reordered
parameters):

    Agent(catalog_path: str | Path = "data/catalog.jsonl") -> None
    reset(session_id: str, user_profile: dict) -> None
    respond(session_id: str, user_message: str, turn: int, top_k: int) -> dict

Note two consequences of that contract: (1) the harness is the source of
truth for `turn`, not an internal counter — Agent.respond() stores
whatever it is given, it does not increment its own; and (2) a single
Agent instance serves many concurrent sessions, so state is a
session_id -> SessionState map, not one SessionState field.

Owner: Chellpapan (Evaluation and integration). §8.5, step E1 (repo skeleton,
BLOCKING), step E2 (BLOCKING — "Wire stubs into a runnable Agent ... End-to-
end run with fixture data before any component is real."), step E8
(integrate modules, keep main green).

This module is the one place every other module is imported together. All
of *those* modules are stubs; this module's job is only orchestration, so
it is the closest thing here to "real" — the turn loop itself follows §3.4
exactly, even though every step it calls returns fixture data.
"""

from __future__ import annotations

from pathlib import Path

from clarify import pick_attribute
from extract import update_slots
from indexes import Indexes, build_indexes, embed_text
from rank import rank
from retrieval import retrieve
from state import SessionState, init_state, pick_track, reconstruct_canonical
from telemetry import log_turn
from utils import load_catalog

MAX_TURNS = 10  # §1.2: "Max 10 turns; exceeding scores zero."


class Agent:
    """A headless conversational shopping agent (§1, "The task").

    Holds the offline indexes for the lifetime of the process (built once,
    §3.2) and one SessionState per session_id, created by reset() and
    mutated turn-by-turn by respond() — matching the kit's contract of one
    Agent instance serving many sessions.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        """Build the offline indexes once.

        Design doc §3.2: "Runs once at startup, in roughly 5 seconds plus
        encoding time." §8.5 step E1. Signature and default path match the
        supplied kit's baseline agent exactly, so the evaluator can
        construct this Agent the same way it constructs that one.

        A fitted ranker (§3.4 Step 6, §8.3 step C5) is not threaded through
        the constructor, because the baseline contract has no such
        parameter; once C5 delivers one, rank.py's `rank()` should default
        to loading it internally rather than this method growing a
        baseline-incompatible argument.

        Args:
            catalog_path: Path to catalog.jsonl. load_catalog() is a stub
                and ignores this, returning utils.FIXTURE_CATALOG instead.
        """
        self.catalog_path = Path(catalog_path)
        self.indexes: Indexes = build_indexes(load_catalog(str(self.catalog_path)))
        self.sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Start a new session (§3.3: "Loaded once in reset()").

        Args:
            session_id: Identifier for the new session, joined against
                ground truth offline (§3.4 Step 7) — never used at
                inference time to look anything up.
            user_profile: The session's raw profile dict (§2.4/§2.4.1).
                Filtered into `profile_terms` by state.derive_profile_terms();
                kept in full for rating_style_fit.

        Returns:
            None.
        """
        self.sessions[session_id] = init_state(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        """Run one full turn of the pipeline (§3.4, §4).

        Order: state update (§3.4 Step 1) -> canonical reconstruction
        (Step 2) -> intent routing (Step 3) -> multi-stream retrieval
        (Step 4) -> clarification decision (Step 5) -> ranking (Step 6) ->
        telemetry (Step 7) -> return.

        Args:
            session_id: Must have been passed to reset() first.
            user_message: The user's utterance for this turn.
            turn: The 1-indexed turn number, supplied by the caller (§1.2's
                10-turn cap is enforced by the harness, not here).
            top_k: How many recommendations to return.

        Raises:
            RuntimeError: If reset() was not called for `session_id` first.

        Returns:
            {"message": str, "ask_attribute": str | None,
             "recommendations": [{"parent_asin": str}, ...] (<= top_k),
             "usage": dict} per §4's system diagram: "RETURN {message,
            ask_attribute, recommendations[10], usage}", with
            `recommendations` shaped as the kit's baseline agent shapes it.
        """
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]
        state.turn = turn

        update_slots(state, user_message)
        reconstruct_canonical(state, embed_text)
        track = pick_track(state)
        pool = retrieve(state, track, self.indexes)

        ask_attribute = pick_attribute(pool, state)
        if ask_attribute is not None:
            state.asked_attributes.add(ask_attribute)

        ranked_asins = rank(pool, state, self.indexes, top_k=top_k)

        log_turn(
            session_id=state.session_id,
            turn=state.turn,
            candidates=pool,
            state=state,
            indexes=self.indexes,
            ask_attribute=ask_attribute,
        )

        return {
            "message": f"[STUB reply] turn={state.turn} track={track} pool_size={len(pool)}",
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": asin} for asin in ranked_asins],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


if __name__ == "__main__":
    # Smoke test: an end-to-end run over three turns, exercising a write,
    # an overwrite, and a negation, per §8.6's "what the demo must show".
    agent = Agent()
    session_id = "demo-session"
    agent.reset(
        session_id,
        user_profile={"preference_tags": ["comfort", "warmth"], "rating_style": "usually positive"},
    )

    for turn_number, turn_message in enumerate(
        [
            "I'm looking for a men's jacket",
            "something for cooler weather, black",
            "actually not black, blue",
        ],
        start=1,
    ):
        result = agent.respond(session_id, turn_message, turn=turn_number, top_k=10)
        print(f"user: {turn_message}")
        print(f"agent: {result}\n")
