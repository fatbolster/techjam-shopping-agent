"""Clarification decision: entropy x answerability.

Design doc §3.4 Step 5 (Clarification decision): "score(a) =
P(answerable | a) x H(a), H(a) = -Sum_v p(v) log2 p(v). Ask about argmax if
the score clears a threshold."

Owner: Marcus (Evaluation and integration). §8.5, step E4 ("Answerability priors
initialised by judgement, then re-estimated from D8 transcripts. Per-session
cap on clarifications.").

(This line was briefly regressed to a broken "Owner: Owner D" placeholder in
8c64053, undoing the correct Chellappan<->Marcus D/E-lane swap from f98422b —
restored here to match README.md's still-correct "Marcus — Evaluation and
integration" section, which is the tie-breaker for current ownership.)

Everything below is a stub except shannon_entropy(), which is implemented
for real: it is a pure, unambiguous formula (§3.4 Step 5) with no design
judgement left to defer, unlike the answerability prior it is weighted by.
"""

from __future__ import annotations

import math
from collections import Counter

from state import CLARIFICATION_ATTRIBUTES, SessionState
from utils import Candidate

# Hand-set answerability priors (§3.4 Step 5: "Answerability is initially
# hand-set and subsequently estimated from instrumented simulator runs.").
# Design doc §3.4 Step 5 worked example gives combined scores of category
# 2.84, department 1.67, brand 0.73 on one real pool; these priors are a
# fixture placeholder, not derived from that example.
#
# Keyed by the evaluator-facing ClarificationAttribute vocabulary (state.py:
# category/material/color/size/style/brand/budget/feature/use_case/other),
# not the internal SlotKey vocabulary — pick_attribute()'s return value is
# handed straight to the evaluator and later fed back through
# extract.py's _CLARIFICATION_SLOT_MAP, which only recognises
# ClarificationAttribute keys and raises KeyError on anything else.
# "department" has no entry: it is useful internally but, per state.py,
# cannot be requested through this interface at all. "other" (the free-text
# fallback channel) is also excluded — it is never the *best* question to
# proactively ask, only a landing spot for an unclassifiable reply.
ANSWERABILITY_PRIOR: dict[str, float] = {
    "category": 0.9,
    "brand": 0.3,
    "color": 0.5,
    "material": 0.5,
    "style": 0.6,
    "size": 0.4,
    "budget": 0.4,
    "feature": 0.45,
    "use_case": 0.45,
}
assert set(ANSWERABILITY_PRIOR) <= set(CLARIFICATION_ATTRIBUTES)

# Score must clear this threshold to trigger a question (§3.4 Step 5:
# "Ask about argmax if the score clears a threshold.").
ASK_THRESHOLD = 1.0

# Per-session cap on clarifications (§3.4 Step 5, §7.4 descoping order #3).
MAX_CLARIFICATIONS_PER_SESSION = 3

# ANSWERABILITY_PRIOR's keys (ClarificationAttribute) and state.slots' keys
# (SlotKey) are two different vocabularies (state.py) that mostly, but not
# entirely, coincide: "budget" maps to three internal slots, not one
# literally named "budget". Needed so _already_filled() below can check
# the right internal slot(s) for each externally-askable attribute.
_ATTRIBUTE_TO_SLOTS: dict[str, tuple[str, ...]] = {
    "category": ("category",),
    "brand": ("brand",),
    "color": ("color",),
    "material": ("material",),
    "style": ("style",),
    "size": ("size",),
    "budget": ("price_min", "price_max", "price_target"),
    "feature": ("feature",),
    "use_case": ("use_case",),
}


def _already_filled(attribute: str, state: SessionState) -> bool:
    """Whether the user has already stated this attribute, in any form."""
    return any(state.slots.get(slot) for slot in _ATTRIBUTE_TO_SLOTS.get(attribute, ()))


def shannon_entropy(values: list[str]) -> float:
    """H(a) = -Sum_v p(v) log2 p(v) over an attribute's value distribution.

    Design doc §3.4 Step 5. Implemented for real: this is the one piece of
    the clarification formula that is a pure, unambiguous calculation, in
    contrast to the answerability prior it is weighted by.

    Args:
        values: The attribute's value for every candidate in the pool
            (e.g. every candidate's `store` field, one entry per
            candidate, duplicates included).

    Returns:
        Entropy in bits. 0.0 for an empty list or a single-valued list.
    """
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def score_attribute(attribute: str, pool: list[Candidate], state: SessionState) -> float:
    """score(a) = P(answerable | a) x H(a) for one unfilled attribute.

    Design doc §3.4 Step 5: "Weighting by answerability inverts the
    ranking correctly: category 2.84, department 1.67, brand 0.73."
    Marcus, step E4.

    STUB: computes real entropy over a fixture value distribution (not the
    real per-candidate attribute values, since facts lookups here are
    stubbed elsewhere) and multiplies by ANSWERABILITY_PRIOR.

    Args:
        attribute: The slot key being considered, e.g. "brand".
        pool: The current candidate pool.
        state: Current session state (would exclude already-filled slots).

    Returns:
        The combined score; higher means a more worthwhile question.
    """
    fixture_values = [f"[STUB value {i % 3}]" for i in range(len(pool))]
    return ANSWERABILITY_PRIOR.get(attribute, 0.0) * shannon_entropy(fixture_values)


def pick_attribute(pool: list[Candidate], state: SessionState) -> str | None:
    """Pick the best attribute to ask about, or None if none clears the bar.

    Design doc §7.2 interface contract: `pick_attribute(pool, state) -> str
    | None`. §3.4 Step 5: "ask_attribute and recommendations occupy the
    same return object ... the decision is therefore not 'ask or answer'
    but 'answer, and additionally ask when useful'."

    Respects `state.asked_attributes` (don't repeat a question) and
    MAX_CLARIFICATIONS_PER_SESSION (protects MTTC, §3.4 Step 5).

    Args:
        pool: The current candidate pool.
        state: Current session state.

    Returns:
        The slot key to ask about, or None.
    """
    if len(state.asked_attributes) >= MAX_CLARIFICATIONS_PER_SESSION:
        return None

    candidates = [
        attr
        for attr in ANSWERABILITY_PRIOR
        if not _already_filled(attr, state) and attr not in state.asked_attributes
    ]
    if not candidates:
        return None

    scored = [(attr, score_attribute(attr, pool, state)) for attr in candidates]
    best_attr, best_score = max(scored, key=lambda pair: pair[1])
    return best_attr if best_score >= ASK_THRESHOLD else None
