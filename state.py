"""Session state: the slot dictionary, scenario buffer, and canonical render.

Design doc §3.3 (Session state) and §3.4 Steps 2-3 (canonical intent
reconstruction, intent routing).

Owner: Qikun (State and routing). §8.2, step B1 — BLOCKING (slot schema),
step B6 (scenario buffer), step B7 (render + canonical reconstruction),
step B8 (routing rule).

Invariant carried over from §3.3: inferred preference (profile terms) never
enters the slot dictionary. The slot dictionary holds exactly what the user
said; `SessionState.profile_terms` is a separate, read-only field.

The B1 schema, B4 clarification lifecycle, B6 scenario assignment, B7
canonical reconstruction, and B8 deterministic routing are implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, TypeAlias

import numpy as np

# Internal explicit-state keys. These are deliberately not the evaluator's
# `ask_attribute` vocabulary: department is useful internally but cannot be
# requested through that interface, `budget` maps to internal price fields,
# and `other` is a clarification request channel rather than a state slot.
SingleValueSlotKey: TypeAlias = Literal[
    "department",
    "category",
    "brand",
    "price_min",
    "price_max",
    "price_target",
]
MultiValueSlotKey: TypeAlias = Literal[
    "color",
    "material",
    "style",
    "size",
    "feature",
    "use_case",
]
SlotKey: TypeAlias = SingleValueSlotKey | MultiValueSlotKey
SlotValue: TypeAlias = str | tuple[str, ...]
ExplicitSlots: TypeAlias = dict[SlotKey, SlotValue]
SlotOverrideFlags: TypeAlias = dict[SlotKey, bool]
OverrideReferenceValues: TypeAlias = dict[SlotKey, tuple[str, ...]]
Track: TypeAlias = Literal["buy", "browse"]

# Exact evaluator-facing clarification vocabulary. It has a separate type so
# an internal key such as `department` can never be returned accidentally and
# `other` does not masquerade as a structured slot.
ClarificationAttribute: TypeAlias = Literal[
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
]
CLARIFICATION_ATTRIBUTES: tuple[ClarificationAttribute, ...] = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)

# A SlotKey may appear here only when the user stated that constraint in the
# current conversation. Profile terms, semantic guesses, and ranking signals
# remain separate. Single-value slots store a string. Descriptive slots can
# store an ordered tuple because the simulator may disclose two simultaneously
# active constraints for one requested attribute.
SLOT_KEYS: tuple[SlotKey, ...] = (
    "department",
    "category",
    "brand",
    "price_min",
    "price_max",
    "price_target",
    "color",
    "material",
    "style",
    "size",
    "feature",
    "use_case",
)
MULTI_VALUE_SLOT_KEYS: tuple[MultiValueSlotKey, ...] = (
    "color",
    "material",
    "style",
    "size",
    "feature",
    "use_case",
)

# Retrieval-facing presentation order. Price fields are rendered together
# after descriptive constraints, so they are handled separately by render().
CANONICAL_SLOT_ORDER: tuple[SlotKey, ...] = (
    "department",
    "category",
    "brand",
    "style",
    "color",
    "material",
    "size",
    "feature",
    "use_case",
)
_CANONICAL_SLOT_LABELS: dict[SlotKey, str] = {
    "department": "department",
    "category": "category",
    "brand": "brand",
    "style": "style",
    "color": "color",
    "material": "material",
    "size": "size",
    "feature": "features",
    "use_case": "use case",
}

# Broad nodes visible in the repository's catalogue category paths. B8 has no
# catalogue/index dependency, and B3 currently stores only the category label
# rather than its path depth, so this conservative denylist is the complete
# hierarchy distinction available through the agreed ``pick_track(state)``
# interface. Unknown non-empty categories are treated as specific.
BROAD_CATEGORY_VALUES: frozenset[str] = frozenset(
    {
        "root",
        "clothing, shoes & jewelry",
        "clothing, shoes and jewelry",
        "men",
        "mens",
        "women",
        "womens",
        "boys",
        "girls",
        "kids",
        "children",
        "clothing",
        "shoes",
    }
)

# The three preference tags §2.4 measured real lift for, against a
# popularity-matched baseline: "Only three rare tags show real lift" out
# of eight (fit, material, comfort, style, durability, performance,
# warmth, weather). The other five sit within +/-2 of chance and are
# deliberately never carried into profile_terms.
RARE_TAGS: tuple[str, ...] = ("performance", "warmth", "weather")


def derive_profile_terms(user_profile: dict) -> list[str]:
    """Filter a raw user_profile down to the three tags with measured lift.

    Revised design §2.4: common tags sit near chance while only three rare
    tags show measured lift. The state contract retains that term list and
    no other profile-derived personalization state.

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
            a non-additive revision since its latest creation: scalar
            conflict, explicit replacement, or partial value deletion.
            Additive multi-value updates do not set it. Removing a slot also
            removes its flag. Diagnostic for override transcripts (§8.6, D8).
        scenario_buffer: Current un-slotted intent, e.g. "for a beach trip".
            A genuine new scenario replaces it; B6 may retain the base plus
            only the latest explicitly additive detail, never turn history.
        profile_terms: Rare-tag term list derived once in reset() (§2.4,
            §3.3). Read-only for the lifetime of the session.
        asked_attributes: Attributes already asked about this session, so
            the clarification policy (clarify.py) does not repeat a
            question (§3.4 Step 5). Uses the evaluator-facing
            ClarificationAttribute vocabulary rather than SlotKey.
        pending_clarification: The one evaluator-facing question the next
            incoming message answers. Unlike asked_attributes, this is
            one-shot and ordered by the actual turn flow.
        override_reference_values: Minimal value-level provenance for the
            explicit non-category preference in the evaluator's initial
            intent-override message. B5 uses it only to invalidate those
            matching active values when an exact evaluator override arrives;
            it is not live intent, history, or an input to retrieval/ranking.
        canonical_intent: The rendered current-intent string from the most
            recent call to reconstruct_canonical() (§3.4 Step 2).
            Logged verbatim every turn (§3.3, §8.2 B7).
        canonical_vector: The embedding of `canonical_intent`, or None
            before the first reconstruction.
        track: "buy" or "browse", set by pick_track() (§3.4 Step 3).
    """

    session_id: str
    turn: int = 0
    slots: ExplicitSlots = field(default_factory=dict)
    slot_override_flags: SlotOverrideFlags = field(default_factory=dict)
    scenario_buffer: str = ""
    profile_terms: list[str] = field(default_factory=list)
    asked_attributes: set[ClarificationAttribute] = field(default_factory=set)
    pending_clarification: Optional[ClarificationAttribute] = None
    override_reference_values: OverrideReferenceValues = field(
        default_factory=dict
    )
    canonical_intent: str = ""
    canonical_vector: Optional[np.ndarray] = None
    track: Track = "browse"


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
            Only measured rare tags are retained in profile_terms; the raw
            mapping is not session state.

    Returns:
        A new, empty SessionState.
    """
    user_profile = user_profile or {}
    return SessionState(
        session_id=session_id,
        profile_terms=derive_profile_terms(user_profile),
    )


def set_pending_clarification(
    state: SessionState, attribute: ClarificationAttribute
) -> SessionState:
    """Record the clarification that the next incoming turn answers.

    The Owner E orchestrator calls this after emitting ``ask_attribute``.
    It separately records the attribute in the historical asked set.
    """
    state.pending_clarification = attribute
    return state


def consume_pending_clarification(
    state: SessionState,
) -> Optional[ClarificationAttribute]:
    """Return and clear the one-shot context before extracting the next turn."""
    attribute = state.pending_clarification
    state.pending_clarification = None
    return attribute


def clear_pending_clarification(state: SessionState) -> SessionState:
    """Clear clarification context without consuming a user turn."""
    state.pending_clarification = None
    return state


def set_scenario(state: SessionState, text: str) -> SessionState:
    """Replace the scenario buffer with a new un-slotted intent statement.

    Design doc §3.3: "A new scenario statement replaces the previous one
    rather than appending." Owner Qikun, §8.2 step B6.

    B6's pure transition logic lives beside ``ExtractionResult`` in
    extract.py; this helper performs only the resulting state assignment.

    Args:
        state: The session state to update.
        text: The un-slotted portion of the user's utterance.

    Returns:
        The same SessionState, mutated in place, for chaining.
    """
    state.scenario_buffer = text
    return state


def _render_slot_values(value: SlotValue) -> str:
    values = value if isinstance(value, tuple) else (value,)
    return ", ".join(item.strip() for item in values if item.strip())


def _render_money(value: str) -> str:
    cleaned = value.strip()
    return cleaned if cleaned.startswith("$") else f"${cleaned}"


def render(state: SessionState) -> str:
    """Render the canonical request string from current slot/scenario state.

    Design doc §3.4 Step 2: "canonical = render(slots) + scenario_buffer".
    Owner Qikun, §8.2 step B7: "Output string is logged verbatim every turn,
    making override behaviour directly auditable."

    Only current ``slots`` and ``scenario_buffer`` are read. Attribute clauses
    use ``CANONICAL_SLOT_ORDER`` rather than mapping insertion order; active
    multi-values retain their tuple order. Price bounds and approximate targets
    retain their distinct semantics.

    Args:
        state: The session state to render.

    Returns:
        The canonical request string, or ``""`` for empty current state.
    """
    clauses: list[str] = []
    for slot in CANONICAL_SLOT_ORDER:
        value = state.slots.get(slot)
        if value is None:
            continue
        rendered = _render_slot_values(value)
        if rendered:
            clauses.append(f"{_CANONICAL_SLOT_LABELS[slot]}: {rendered}")

    price_min = state.slots.get("price_min")
    price_max = state.slots.get("price_max")
    if isinstance(price_min, str) and price_min.strip():
        if isinstance(price_max, str) and price_max.strip():
            clauses.append(
                f"between {_render_money(price_min)} and {_render_money(price_max)}"
            )
        else:
            clauses.append(f"at least {_render_money(price_min)}")
    elif isinstance(price_max, str) and price_max.strip():
        clauses.append(f"under {_render_money(price_max)}")

    price_target = state.slots.get("price_target")
    if isinstance(price_target, str) and price_target.strip():
        clauses.append(f"around {_render_money(price_target)}")

    scenario = state.scenario_buffer.strip()
    if scenario:
        clauses.append(scenario)
    return "; ".join(clauses)


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

    Args:
        state: The session state to update.
        embed_fn: A function mapping text to a 384-d vector, e.g.
            `indexes.embed_text`.

    Returns:
        The same SessionState, with `canonical_intent` and
        `canonical_vector` refreshed.
    """
    state.canonical_intent = render(state)
    state.canonical_vector = (
        embed_fn(state.canonical_intent) if state.canonical_intent else None
    )
    return state


def pick_track(state: SessionState) -> Track:
    """Authorize the restrictive buying path from current explicit state.

    The current Owner A stub changes keyword/semantic quotas by track but does
    not yet apply its documented buying-only department/category filters.
    Because category path depth is not carried in SessionState, B8 treats only
    a non-empty category outside the known broad catalogue nodes as sufficient
    authorization. Department alone and arbitrary slot counts remain browse.

    The result is derived fresh and replaces ``state.track`` every call; no
    prior track, scenario text, profile, or transition metadata participates.

    Args:
        state: The session state to classify.

    Returns:
        ``"buy"`` for a specific explicit category, otherwise ``"browse"``.
    """
    category = state.slots.get("category")
    has_specific_category = (
        isinstance(category, str)
        and bool(category.strip())
        and category.strip().casefold() not in BROAD_CATEGORY_VALUES
    )
    track: Track = "buy" if has_specific_category else "browse"
    state.track = track
    return track
