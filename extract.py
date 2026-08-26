"""Slot extraction and negation: turn text into slot writes/overwrites/deletes.

Design doc §3.4 Step 1 (State update) and §3.3's update rule ("Write on new
key; overwrite on conflict, setting an override flag; delete on detected
negation.").

Owner: Qikun (State and routing). §8.2, step B2 (gazetteer), step B3 (rule-based
extraction — "the fallback the whole system rests on"), step B4 (merge
policy), step B5 (negation detector — "All 30 intent_override sessions
carry difficulty_bucket: hard ... Qikun owns the highest-value defect surface"),
step B9 (optional LLM path).

Everything below is a stub. Function bodies return fixture values only.
"""

from __future__ import annotations

from state import SLOT_KEYS, SessionState, set_scenario

# Built from the catalogue's own attribute vocabulary (§3.4 Step 1: "a
# gazetteer built from the catalogue's own attribute vocabulary, which
# guarantees every extracted value exists in the data"). Owner Qikun, step B2.
# STUB: a tiny fixture gazetteer, not the real Counter-over-product_text()
# vocabulary of ~50,000 rows.
GAZETTEER: dict[str, list[str]] = {
    "department": ["mens", "womens", "kids"],
    "category": ["jacket", "running shoe", "cover-up"],
    "brand": ["london fog", "saucony", "sundaze"],
    "color": ["black", "blue", "auburn"],
    "material": ["polyester", "mesh"],
    "style": ["golf jacket", "running shoe", "cover-up"],
}

# Negation constructions the simulator is expected to produce (§3.4 Step 1:
# "not X, instead of X, actually Y"). Owner Qikun, step B5, refined from D8
# transcripts before being finalized — this fixture list is a placeholder
# for that refinement.
NEGATION_PATTERNS: tuple[str, ...] = ("not ", "instead of ", "actually ")


def extract_slots(message: str) -> dict[str, str]:
    """Extract flat attribute writes from one user utterance.

    Design doc §3.4 Step 1: "Primary path: one constrained-JSON LLM call
    returning a flat attribute object. Fallback path: regex plus a
    gazetteer ... The fallback path alone yields a fully functional agent."
    Owner Qikun, §8.2 step B3.

    STUB: returns a fixed single-slot fixture regardless of `message`,
    tagged with a truncated copy of the input for traceability. Neither the
    LLM primary path nor the regex/gazetteer fallback is implemented yet.

    Args:
        message: The raw user utterance for this turn.

    Returns:
        A flat dict of slot_key -> value for attributes newly stated in
        `message` (does not include attributes the negation detector would
        remove — see detect_negation()).
    """
    return {"category": f"[STUB extracted from: {message[:30]!r}]"}


def detect_negation(message: str) -> list[str]:
    """Detect which slot keys the utterance negates or overrides away.

    Design doc §3.4 Step 1: "Negation detection is pattern-based (not X,
    instead of X, actually Y)." Owner Qikun, §8.2 step B5 — the highest-value
    defect surface: all 30 intent_override sessions are difficulty: hard.

    STUB: returns an empty list unless one of NEGATION_PATTERNS appears as
    a substring, in which case it returns a fixture guess of ["color"]. Not
    the real pattern-to-slot-key mapping.

    Args:
        message: The raw user utterance for this turn.

    Returns:
        Slot keys (a subset of SLOT_KEYS) that should be deleted from the
        slot dictionary, not merely overwritten.
    """
    lowered = message.lower()
    if any(pattern in lowered for pattern in NEGATION_PATTERNS):
        return ["color"]
    return []


def update_slots(state: SessionState, message: str) -> SessionState:
    """Merge one utterance's extracted attributes into the slot dictionary.

    Design doc §3.3 update rule and §3.4 Step 1. Owner Qikun, §8.2 step B4:
    "Overwrite sets an override flag. Negation removes the key outright
    rather than adding a negative value." This is the `update_slots`
    signature named in §7.2's interface contract table.

    Un-slotted content (i.e. `message` when extract_slots() finds nothing
    in it) replaces the scenario buffer via state.set_scenario(), per §3.4
    Step 1: "Un-slotted content replaces the scenario buffer."

    STUB: applies detect_negation() deletes, then extract_slots() writes
    (marking overwrite=True when the key already existed), then always
    also updates the scenario buffer with the raw message — the real
    implementation only does the last step when extraction yields nothing.

    Args:
        state: The session state to update, mutated in place.
        message: The raw user utterance for this turn.

    Returns:
        The same SessionState, with slots/scenario_buffer refreshed.
    """
    for key in detect_negation(message):
        state.slots.pop(key, None)
        state.slot_override_flags.pop(key, None)

    for key, value in extract_slots(message).items():
        if key not in SLOT_KEYS:
            continue
        state.slot_override_flags[key] = key in state.slots
        state.slots[key] = value

    set_scenario(state, message)
    return state
