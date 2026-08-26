"""Session state: the slot dictionary, scenario buffer, and canonical render.

Design doc §3.3 (Session state) and §3.4 Steps 2-3 (canonical intent
reconstruction, intent routing).

Owner: Qikun (State and routing). §8.2, step B1 — BLOCKING (slot schema),
step B6 (scenario buffer), step B7 (render + canonical reconstruction),
step B8 (routing rule).

Invariant carried over from §3.3: inferred preference (profile terms) never
enters the slot dictionary. The slot dictionary holds exactly what the user
said; `SessionState.profile_terms` is a separate, read-only field.

Everything below is a stub. Function bodies return fixture values only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Fixed key set (§8.2 B1: "Fixed key set. A knows which slots may filter;
# C knows which slots contribute to slot_coverage."). Values are always
# strings the user stated verbatim or a normalized form of them; price is
# split into min/max so a one-sided budget ("under $50") is representable.
SLOT_KEYS: tuple[str, ...] = (
    "department",
    "category",
    "brand",
    "price_min",
    "price_max",
    "color",
    "material",
    "style",
    "size",
)

# The three preference tags §2.4 measured real lift for, against a
# popularity-matched baseline: "Only three rare tags show real lift" out
# of eight (fit, material, comfort, style, durability, performance,
# warmth, weather). The other five sit within +/-2 of chance and are
# deliberately never carried into profile_terms.
RARE_TAGS: tuple[str, ...] = ("performance", "warmth", "weather")


def derive_profile_terms(user_profile: dict) -> list[str]:
    """Filter a raw user_profile down to the three tags with measured lift.

    Design doc §2.4 ("The four tags carried by most profiles sit within
    +/-2 of chance. Only three rare tags show real lift.") and §2.4
    Consequence: "reduced to two ranking features: a three-tag term list
    active in ~22% of sessions." Implemented for real — it is a filter
    against a constant named in the design, not a scoring decision.

    Args:
        user_profile: The raw profile dict passed to Agent.reset(). Per
            data/public_set.jsonl, carries a "preference_tags" list among
            other (mostly non-discriminative, per §2.4) fields —
            average_prior_rating, purchase_frequency, rating_style,
            summary.

    Returns:
        The subset of `user_profile["preference_tags"]` that are in
        RARE_TAGS.
    """
    tags = user_profile.get("preference_tags", []) if user_profile else []
    return [tag for tag in tags if tag in RARE_TAGS]


@dataclass
class SessionState:
    """Everything the agent carries across turns within one session.

    Design doc §3.3. Four objects: slot dictionary, scenario buffer, profile
    terms (read-only), and (elsewhere) the telemetry log.

    Attributes:
        session_id: Identifier joined offline against ground truth (§3.4
            Step 7); never used to look up ground truth at inference time.
        turn: 1-indexed turn counter within the session, capped at 10 (§1.2).
        slots: The slot dictionary — attributes the user explicitly stated.
            Keys are a subset of SLOT_KEYS.
        slot_override_flags: Whether each currently-set slot was written by
            an overwrite (§3.3: "overwrite on conflict, setting an override
            flag") rather than a first write. Diagnostic for override
            transcripts (§8.6, D8).
        scenario_buffer: Un-slotted intent, e.g. "for a beach trip". Replaced
            wholesale by a new scenario statement, never appended to (§3.3).
        profile_terms: Rare-tag term list derived once in reset() (§2.4,
            §3.3). Read-only for the lifetime of the session.
        user_profile: The raw profile dict passed to Agent.reset(), kept
            for rating_style_fit (§2.4.1), which needs the profile's
            rating skew, not just its rare tags.
        asked_attributes: Attributes already asked about this session, so
            the clarification policy (clarify.py) does not repeat a
            question (§3.4 Step 5).
        canonical_intent: The rendered + embedded request string from the
            most recent call to reconstruct_canonical() (§3.4 Step 2).
            Logged verbatim every turn (§3.3, §8.2 B7).
        canonical_vector: The embedding of `canonical_intent`, or None
            before the first reconstruction.
        track: "buy" or "browse", set by pick_track() (§3.4 Step 3).
    """

    session_id: str
    turn: int = 0
    slots: dict[str, str] = field(default_factory=dict)
    slot_override_flags: dict[str, bool] = field(default_factory=dict)
    scenario_buffer: str = ""
    profile_terms: list[str] = field(default_factory=list)
    user_profile: dict = field(default_factory=dict)
    asked_attributes: set[str] = field(default_factory=set)
    canonical_intent: str = ""
    canonical_vector: Optional[np.ndarray] = None
    track: str = "browse"


def init_state(session_id: str, user_profile: Optional[dict] = None) -> SessionState:
    """Construct a fresh SessionState for the start of a session.

    Design doc §3.3 ("Profile terms ... Loaded once in reset(); no code path
    writes to it") and §8.2 step B1. Signature matches the supplied kit's
    `Agent.reset(session_id, user_profile)` contract (§5.1's baseline
    agent), which passes a raw profile dict rather than a pre-filtered
    term list.

    Args:
        session_id: Identifier for the new session.
        user_profile: The raw profile dict from Agent.reset(), or None.
            Filtered down to profile_terms via derive_profile_terms();
            kept in full on `user_profile` for rating_style_fit (§2.4.1).

    Returns:
        A new, empty SessionState.
    """
    user_profile = user_profile or {}
    return SessionState(
        session_id=session_id,
        profile_terms=derive_profile_terms(user_profile),
        user_profile=user_profile,
    )


def set_scenario(state: SessionState, text: str) -> SessionState:
    """Replace the scenario buffer with a new un-slotted intent statement.

    Design doc §3.3: "A new scenario statement replaces the previous one
    rather than appending." Owner Qikun, §8.2 step B6.

    STUB: overwrites `state.scenario_buffer` with `text` verbatim; the real
    implementation is exactly this (replace, not append) so there is little
    "real logic" left to add here beyond what extract.py decides to pass in.

    Args:
        state: The session state to update.
        text: The un-slotted portion of the user's utterance.

    Returns:
        The same SessionState, mutated in place, for chaining.
    """
    state.scenario_buffer = text
    return state


def render(state: SessionState) -> str:
    """Render the canonical request string from current slot/scenario state.

    Design doc §3.4 Step 2: "canonical = render(slots) + scenario_buffer".
    Owner Qikun, §8.2 step B7: "Output string is logged verbatim every turn,
    making override behaviour directly auditable."

    STUB: returns a fixture string summarizing slot count and scenario
    presence rather than the real "key: value, key: value ... scenario"
    join, so the shape (a single deterministic string) is exercisable
    end-to-end before the real renderer lands.

    Args:
        state: The session state to render.

    Returns:
        The canonical request string, not yet embedded.
    """
    return f"[STUB render: {len(state.slots)} slot(s), scenario={state.scenario_buffer!r}]"


def reconstruct_canonical(
    state: SessionState, embed_fn: Callable[[str], np.ndarray]
) -> SessionState:
    """Re-render the canonical string and embed it fresh (§3.4 Step 2).

    Design doc §3.4 Step 2: "Re-render the full request from current state
    and embed it fresh ... encoded by the same MiniLM model used for the
    catalog, so query and products occupy one space." Owner Qikun, step B7.

    Takes `embed_fn` as a parameter (rather than importing indexes.py
    directly) to keep state.py free of a dependency on the offline index
    build, per the acyclic module layout in this scaffold.

    STUB: calls `embed_fn` (itself a stub in indexes.py) on the rendered
    string and stores both; no real embedding happens yet.

    Args:
        state: The session state to update.
        embed_fn: A function mapping text to a 384-d vector, e.g.
            `indexes.embed_text`.

    Returns:
        The same SessionState, with `canonical_intent` and
        `canonical_vector` refreshed.
    """
    state.canonical_intent = render(state)
    state.canonical_vector = embed_fn(state.canonical_intent)
    return state


def pick_track(state: SessionState) -> str:
    """Classify the turn as "buy" or "browse" from slot-dictionary state.

    Design doc §3.4 Step 3: "Rule over slot state: presence of a department
    slot, or two or more hard constraints, or a leaf-category noun ->
    BUYING; otherwise BROWSING. Re-evaluated every turn so the track can
    flip mid-session." Owner Qikun, §8.2 step B8.

    STUB: returns "buy" whenever two or more slots are set (a rough,
    fixture stand-in for the real department/leaf-category rule), else
    "browse". Also writes the result onto `state.track`.

    Args:
        state: The session state to classify.

    Returns:
        "buy" or "browse".
    """
    track = "buy" if len(state.slots) >= 2 else "browse"
    state.track = track
    return track
