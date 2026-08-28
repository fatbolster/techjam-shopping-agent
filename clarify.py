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

shannon_entropy() was always real (§3.4 Step 5, a pure formula). score_attribute()
now computes real per-candidate value distributions too, via an optional
`indexes` parameter (keyword-only, defaulting to None — pick_attribute(pool,
state) -> str | None, §7.2's interface sketch, still works unchanged for
any caller that doesn't pass one; see score_attribute()'s docstring for
what None falls back to). ANSWERABILITY_PRIOR stays hand-set — §3.4 Step 5:
"initially hand-set and subsequently estimated from instrumented simulator
runs" — re-estimating it is D8/E4's transcript-driven follow-up, not done
here.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from indexes import Indexes
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
#
# Lowered from 1.0 after measuring the supplied evaluator's actual
# behaviour (docs/DESIGN_AUDIT.md, DC-3): when `ask_attribute` is None,
# evaluator.py's customer_reply() returns a content-free string ("Ask me
# about one specific attribute.") and discloses nothing. Asking is
# therefore not a cost to be justified — it is the only channel through
# which the agent acquires information, and it is additive (a full ranked
# 10 is still returned and scored on the same turn). At 1.0 we ran 71.7%
# of turns silent (3,224 of 4,497 logged), each of which could not narrow
# the pool. §1.2's "every question must justify its cost" framing predates
# sight of the evaluator.
ASK_THRESHOLD = 0.15

# Per-session cap on clarifications (§3.4 Step 5, §7.4 descoping order #3).
#
# Raised from 3 for the same reason: the cap exists to "protect MTTC", but
# MTTC is only harmed by a question when the session would otherwise have
# converged sooner — and a silent turn cannot converge at all, because the
# simulated user discloses nothing without a question. At 3, 203 sessions
# exhausted the cap and then spent 679 further turns (15.1% of all turns)
# unable to learn anything. Set to the evaluator's own 10-turn ceiling so
# the cap never binds before the session does.
MAX_CLARIFICATIONS_PER_SESSION = 10

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


# feature/use_case have no dedicated structured facts field (facts.py's
# Index 3 carries dept/cat3/store/price/brand/color/material/style/size —
# "feature" is free text, not a controlled few-valued attribute anyone
# catalogued). Bucketing each candidate by which of a small controlled
# phrase set appears in facts[asin]['blob'] (§3.2's lowercased
# product_text()) gives a real, if coarse, categorical distribution rather
# than a fixture placeholder — reusing the same "small hand-picked phrase
# list matched against text" approach extract.py's B3 already uses for
# these two attributes (its _FEATURE_PHRASES/_use_case_findings), rather
# than inventing a third vocabulary.
_FEATURE_PHRASE_BUCKETS: tuple[str, ...] = (
    "water resistant",
    "waterproof",
    "machine washable",
    "lightweight",
    "breathable",
)
_USE_CASE_PHRASE_BUCKETS: tuple[str, ...] = (
    "running",
    "hiking",
    "yoga",
    "travel",
    "gym",
    "outdoor",
    "casual",
    "formal",
)

# Budget has a real continuous facts field (price) but every candidate's
# exact price is almost always unique, so literal per-candidate entropy
# would sit near log2(pool size) regardless of whether prices actually
# cluster — uninformative in the specific way §3.4 Step 5's H(a) is meant
# to avoid. Bucketing into $20 bands turns it into the same kind of
# few-valued categorical distribution as every other attribute here.
_BUDGET_BUCKET_WIDTH = 20.0


def _attribute_value(attribute: str, asin: str, facts: dict[str, dict]) -> Optional[str]:
    """One candidate's bucketed value for `attribute`, or None if unknown.

    None entries are dropped before shannon_entropy() (a candidate with no
    known value contributes no information either way, rather than being
    forced into a synthetic "unknown" bucket that would understate the
    attribute's true informativeness among candidates that do have it).
    """
    row = facts.get(asin, {})
    if attribute == "category":
        return row.get("cat3") or row.get("dept")
    if attribute == "brand":
        return row.get("brand") or row.get("store")
    if attribute in ("color", "material", "style", "size"):
        return row.get(attribute)
    if attribute == "budget":
        price = row.get("price")
        # The real catalogue's price field is 78.9% null (§2.2) and, among
        # the rest, not always numeric — junk values like "—" or a
        # from-price string like "from 12.99" appear (never cleaned,
        # matching price_fit()'s own scope: §2.2/features.py's price_fit()
        # only implements the null-safety branch, not a real budget-fit
        # formula, for the identical reason). Anything not already a plain
        # number is treated the same as null: excluded from the
        # distribution, not coerced or parsed.
        if not isinstance(price, (int, float)):
            return None
        return str(int(price // _BUDGET_BUCKET_WIDTH))
    if attribute in ("feature", "use_case"):
        blob = row.get("blob") or ""
        phrases = _FEATURE_PHRASE_BUCKETS if attribute == "feature" else _USE_CASE_PHRASE_BUCKETS
        matched = tuple(p for p in phrases if p in blob)
        return ",".join(matched) if matched else "none"
    return None


def score_attribute(
    attribute: str, pool: list[Candidate], state: SessionState, *, indexes: Optional[Indexes] = None
) -> float:
    """score(a) = P(answerable | a) x H(a) for one unfilled attribute.

    Design doc §3.4 Step 5: "Weighting by answerability inverts the
    ranking correctly: category 2.84, department 1.67, brand 0.73."
    Marcus, step E4.

    Computes real per-candidate values via `indexes.facts` (see
    `_attribute_value()` for the attribute -> facts-field mapping,
    including the two attributes — feature, use_case — with no dedicated
    facts field) and takes their Shannon entropy, weighted by the hand-set
    `ANSWERABILITY_PRIOR`. `state` is accepted (per the interface sketch)
    but not read yet — excluding already-filled slots is pick_attribute()'s
    job, not this scoring function's.

    Args:
        attribute: The ClarificationAttribute being considered, e.g. "brand".
        pool: The current candidate pool.
        state: Current session state.
        indexes: Offline indexes bundle, keyword-only. None falls back to
            an empty value distribution (entropy 0, so this attribute
            never wins) — the same "no facts, no signal" fallback
            extract.py/rank.py's other real functions use when their
            optional catalogue-derived argument is omitted, rather than a
            fixture placeholder that could win a question it has no real
            basis to ask.

    Returns:
        The combined score; higher means a more worthwhile question.
    """
    if indexes is None:
        return 0.0
    values = [v for v in (_attribute_value(attribute, c.asin, indexes.facts) for c in pool) if v is not None]
    return ANSWERABILITY_PRIOR.get(attribute, 0.0) * shannon_entropy(values)


def pick_attribute(
    pool: list[Candidate], state: SessionState, *, indexes: Optional[Indexes] = None
) -> str | None:
    """Pick the best attribute to ask about, or None if none clears the bar.

    Design doc §7.2 interface contract: `pick_attribute(pool, state) -> str
    | None` — still satisfied exactly by any caller that omits `indexes`
    (keyword-only, defaulting to None); Agent passes its own indexes bundle
    for the real per-candidate scoring (see score_attribute()). §3.4 Step
    5: "ask_attribute and recommendations occupy the same return object
    ... the decision is therefore not 'ask or answer' but 'answer, and
    additionally ask when useful'."

    Respects `state.asked_attributes` (don't repeat a question) and
    MAX_CLARIFICATIONS_PER_SESSION (protects MTTC, §3.4 Step 5).

    Args:
        pool: The current candidate pool.
        state: Current session state.
        indexes: Offline indexes bundle, keyword-only — see
            score_attribute()'s docstring for the None fallback.

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

    scored = [(attr, score_attribute(attr, pool, state, indexes=indexes)) for attr in candidates]
    best_attr, best_score = max(scored, key=lambda pair: pair[1])
    return best_attr if best_score >= ASK_THRESHOLD else None
