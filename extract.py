"""Slot extraction and negation: turn text into slot writes/overwrites/deletes.

Design doc §3.4 Step 1 (State update) and §3.3's update rule ("Write on new
key; overwrite on conflict, setting an override flag; delete on detected
negation.").

Owner: Qikun (State and routing). §8.2, step B2 (gazetteer), step B3 (rule-based
extraction — "the fallback the whole system rests on"), step B4 (merge
policy), step B5 (negation detector — "All 30 intent_override sessions
carry difficulty_bucket: hard ... Qikun owns the highest-value defect surface"),
step B9 (optional LLM path).

B2's gazetteer builder, B3's deterministic single-utterance extraction,
B4's structured state transitions, B5's conservative V1 negation planner,
and B6's deterministic scenario-buffer transitions are implemented.
B9 (extract_slots_llm(), behind USE_LLM_EXTRACTION) is a deliberate stub —
no LLM access is provided in this environment (§1.2), same as rank.py's
llm_rerank() (C7, closed as won't-do for the identical reason). It always
reports "missing credentials" so update_slots()'s fallback path is real and
tested even though the LLM call itself never runs.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal, Optional, TypeAlias

from state import (
    ClarificationAttribute,
    MULTI_VALUE_SLOT_KEYS,
    SLOT_KEYS,
    ExplicitSlots,
    OverrideReferenceValues,
    SessionState,
    SlotKey,
    consume_pending_clarification,
    set_scenario,
)

AttributeGazetteer = dict[SlotKey, dict[str, str]]

# Only these slots have a precision-safe controlled vocabulary at B2. Size
# and price need patterns; feature needs conversational cues; none may be
# populated by tokenising product_text(), titles, or descriptions.
GAZETTEER_BACKED_SLOTS: tuple[SlotKey, ...] = (
    "department",
    "category",
    "brand",
    "color",
    "material",
    "style",
    "use_case",
)

_FIXED_COLOR_LOOKUPS: dict[str, str] = {
    "black": "black",
    "white": "white",
    "blue": "blue",
    "red": "red",
    "pink": "pink",
    "green": "green",
    "brown": "brown",
    "gray": "gray",
    "grey": "gray",
    "purple": "purple",
    "yellow": "yellow",
    "orange": "orange",
}
_FIXED_MATERIALS: tuple[str, ...] = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
_FIXED_USE_CASES: tuple[str, ...] = (
    "hiking",
    "running",
    "gym",
    "winter",
    "outdoor",
    "work",
)
_DEPARTMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "men": ("mens",),
    "women": ("womens",),
    "boys": ("boy",),
    "girls": ("girl",),
    "kids": ("kid", "children"),
}
_MAX_STRUCTURED_VALUE_LENGTH = 80
_MAX_STRUCTURED_VALUE_TOKENS = 8


@functools.lru_cache(maxsize=512)
def normalize_gazetteer_value(value: str) -> str:
    """Normalize one exact lookup value without splitting it into tokens.

    Unicode compatibility folding, diacritic removal, case folding, and
    punctuation/whitespace folding make catalogue spellings comparable while
    preserving phrase boundaries. Apostrophes are removed so ``men's`` maps
    to the conservative department alias ``mens``.

    Cached: a single extract_slots() call independently recomputes this on
    the same `message` string up to 4 times (_gazetteer_findings,
    _semantic_findings, _use_case_findings, _residual_scenario), each
    paying the full NFKD-fold-and-regex cost on the same input. The cache
    is pure/hashable-args-only (str -> str, no side effects), so this is
    safe; bounded size keeps memory flat over a long-running process
    without needing to thread a precomputed value through four call sites
    with independent signatures and independent test coverage.
    """
    text = unicodedata.normalize("NFKD", value)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"['’`]", "", text)
    return re.sub(r"[\W_]+", " ", text, flags=re.UNICODE).strip()


def _canonical_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _safe_structured_value(value: str) -> bool:
    normalized = normalize_gazetteer_value(value)
    return (
        len(normalized) >= 2
        and len(value) <= _MAX_STRUCTURED_VALUE_LENGTH
        and len(normalized.split()) <= _MAX_STRUCTURED_VALUE_TOKENS
    )


# Marketing/campaign taxonomy noise that survives _safe_structured_value's
# length check but is not a real department or category a user would ever
# state. Verified against data/catalog.jsonl: every one of the 72 distinct
# categories[] path entries containing a digit, "$", or "%" is a seasonal
# or promo bucket ("Under $50", "Prime Day: 30% off...", "Toddler Size 6",
# "Save up to X% on Burt's Bees") — none is a genuine department/category
# name. The keyword set is kept deliberately short and word-matched (not
# substring) because the category-path space also legitimately contains
# thousands of short, rare real category names ("Fedoras", "Cravats",
# "Tapers") that a broader denylist would risk rejecting.
_PROMO_NOISE_TOKENS: frozenset[str] = frozenset(
    {"clearance", "test", "outlet", "markdown", "markdowns"}
)


def _looks_like_promo_noise(canonical: str) -> bool:
    """True for department/category source strings that are taxonomy noise.

    Checked against `canonical` (pre-normalization) rather than the
    normalized form, since normalize_gazetteer_value() strips "$"/"%" as
    non-word characters before this would ever see them.

    Deliberately NOT applied to brand/store or style: 418 distinct real
    store names in this catalogue contain digits ("7 For All Mankind",
    "5.11", "Core 10"), so the same filter there would reject legitimate
    brands rather than noise.
    """
    if any(char.isdigit() for char in canonical) or "$" in canonical or "%" in canonical:
        return True
    tokens = set(normalize_gazetteer_value(canonical).split())
    return bool(tokens & _PROMO_NOISE_TOKENS)


def _string_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def _detail_values(row: dict, wanted_key: str) -> list[str]:
    details = row.get("details")
    if not isinstance(details, dict):
        return []
    for key, value in details.items():
        if str(key).casefold() == wanted_key.casefold():
            return _string_values(value)
    return []


def _choose_canonical(counts: Counter[str]) -> str:
    """Choose the most common original spelling with a stable tie-break."""
    return min(
        counts,
        key=lambda value: (-counts[value], value.casefold(), value),
    )


def build_attribute_gazetteer(catalog: list[dict]) -> AttributeGazetteer:
    """Build precision-first exact lookup maps from shared catalogue rows.

    Catalogue-backed values come only from labelled structured fields:
    ``categories[1]`` for department, every category-path entry for category,
    ``store`` for brand, and ``details.Style`` for style. Color, material, and
    use-case values are small evaluator-aligned fixed vocabularies because the
    corresponding metadata is sparse or because no structured field exists.

    The function reads but never mutates ``catalog`` or its rows. It
    deliberately does not call ``product_text()``: that shared representation
    is correct for retrieval, but tokenising it would turn arbitrary title and
    description words into dangerous explicit constraints.
    """
    counts: dict[SlotKey, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    def add(slot: SlotKey, raw_value: str, *, filter_promo_noise: bool = False) -> None:
        canonical = _canonical_text(raw_value)
        if not _safe_structured_value(canonical):
            return
        if filter_promo_noise and _looks_like_promo_noise(canonical):
            return
        normalized = normalize_gazetteer_value(canonical)
        counts[slot][normalized][canonical] += 1

    for row in catalog:
        categories = _string_values(row.get("categories"))
        if len(categories) > 1:
            add("department", categories[1], filter_promo_noise=True)
        for category in categories:
            add("category", category, filter_promo_noise=True)
        for store in _string_values(row.get("store")):
            add("brand", store)
        for style in _detail_values(row, "Style"):
            add("style", style)

    gazetteer: AttributeGazetteer = {
        slot: {
            normalized: _choose_canonical(canonical_counts)
            for normalized, canonical_counts in counts.get(slot, {}).items()
        }
        for slot in GAZETTEER_BACKED_SLOTS
    }

    department = gazetteer["department"]
    for normalized, canonical in tuple(department.items()):
        for alias in _DEPARTMENT_ALIASES.get(normalized, ()):
            department.setdefault(normalize_gazetteer_value(alias), canonical)

    gazetteer["color"].update(_FIXED_COLOR_LOOKUPS)
    gazetteer["material"].update(
        {normalize_gazetteer_value(value): value for value in _FIXED_MATERIALS}
    )
    gazetteer["use_case"].update(
        {normalize_gazetteer_value(value): value for value in _FIXED_USE_CASES}
    )
    return gazetteer


def lookup_gazetteer(
    gazetteer: AttributeGazetteer, slot: SlotKey, value: str
) -> str | None:
    """Return a canonical value for one exact normalized lookup."""
    return gazetteer.get(slot, {}).get(normalize_gazetteer_value(value))


def gazetteer_vocabulary_sizes(
    gazetteer: AttributeGazetteer,
) -> dict[SlotKey, int]:
    """Return normalized lookup count for each gazetteer-backed slot."""
    return {slot: len(gazetteer.get(slot, {})) for slot in GAZETTEER_BACKED_SLOTS}


# ---------------------------------------------------------------------------
# B3: deterministic extraction from one utterance
# ---------------------------------------------------------------------------
ExtractionSource: TypeAlias = Literal[
    "gazetteer",
    "budget_pattern",
    "size_pattern",
    "use_case_pattern",
    "requirement_phrase",
    "attribute_label",
    "clarification_context",
]


@dataclass(frozen=True)
class SlotObservation:
    """One explicit value found in the current user utterance.

    ``source`` is deliberately categorical rather than a synthetic numeric
    confidence. B3 emits only high-confidence evidence; uncertain text stays
    in ``residual_scenario`` or is omitted.
    """

    slot: SlotKey
    value: str
    source: ExtractionSource


@dataclass(frozen=True)
class ExtractionResult:
    """Pure B3 output: explicit slots plus unstructured scenario intent."""

    slots: ExplicitSlots
    residual_scenario: str | None
    observations: tuple[SlotObservation, ...]


@dataclass(frozen=True)
class _Finding:
    slot: SlotKey
    value: str
    source: ExtractionSource
    order: int


_FEATURE_PHRASES: tuple[str, ...] = (
    "water resistant",
    "machine washable",
    "waterproof",
    "lightweight",
    "breathable",
)
_STYLE_PHRASES: tuple[str, ...] = (
    "relaxed fit",
    "slim fit",
    "regular fit",
    "loose fit",
    "oversized fit",
    "athletic fit",
    "long sleeve",
    "short sleeve",
    "crew neck",
    "v neck",
)
_PRODUCT_NOUNS: frozenset[str] = frozenset(
    {
        "apparel",
        "clothing",
        "coat",
        "coats",
        "dress",
        "dresses",
        "jacket",
        "jackets",
        "product",
        "products",
        "shirt",
        "shirts",
        "shoe",
        "shoes",
    }
)
_SCENARIO_CUES: frozenset[str] = frozenset(
    {
        "beach",
        "commute",
        "exploring",
        "gift",
        "honeymoon",
        "occasion",
        "trip",
        "travel",
        "vacation",
        "wedding",
    }
)
_RESIDUAL_FILLER: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "boy",
        "boys",
        "children",
        "for",
        "girl",
        "girls",
        "kid",
        "kids",
        "men",
        "mens",
        "my",
        "our",
        "something",
        "the",
        "women",
        "womens",
    }
    | _PRODUCT_NOUNS
)
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_BETWEEN_BUDGET = re.compile(
    rf"\bbetween\s*(?:usd\s*)?\$?\s*({_NUMBER})\s*"
    rf"(?:and|to|-)\s*(?:usd\s*)?\$?\s*({_NUMBER})\b",
    re.IGNORECASE,
)
_MAX_BUDGET = re.compile(
    rf"\b(?:under|below|less\s+than|up\s+to|at\s+most|maximum(?:\s+of)?)"
    rf"\s*(?:usd\s*)?\$?\s*({_NUMBER})\b",
    re.IGNORECASE,
)
_MIN_BUDGET = re.compile(
    rf"\b(?:at\s+least|over|above|more\s+than|minimum(?:\s+of)?)"
    rf"\s*(?:usd\s*)?\$?\s*({_NUMBER})\b",
    re.IGNORECASE,
)
_APPROXIMATE_BUDGET = re.compile(
    rf"\bbudget\s+(?:is\s+)?(?:around|about|roughly|approximately)\s*"
    rf"(?:usd\s*)?\$?\s*({_NUMBER})\b",
    re.IGNORECASE,
)
_APPROXIMATE_CURRENCY = re.compile(
    rf"\b(?:around|about|roughly|approximately)\s*"
    rf"(?:usd\s*)?\$\s*({_NUMBER})\b",
    re.IGNORECASE,
)
_CURRENCY_RANGE = re.compile(
    rf"(?:usd\s*)?\$\s*({_NUMBER})\s*(?:to|-)\s*"
    rf"(?:usd\s*)?\$?\s*({_NUMBER})\b",
    re.IGNORECASE,
)
_BARE_MONEY = re.compile(
    rf"^(?:usd\s*)?\$?\s*({_NUMBER})$",
    re.IGNORECASE,
)
_SIZE_PATTERN = re.compile(
    r"\bsize\s*(?:is\s*)?"
    r"(one\s+size|(?:\d{1,2}(?:\.5)?|(?:[2-6]x|x{0,3})[sl]|m))\b",
    re.IGNORECASE,
)
_EVALUATOR_REPLY = re.compile(
    r"\bwhat\s+matters\s+is\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_EXPLICIT_REQUIREMENT = re.compile(
    r"\b(?:a\s+key\s+requirement\s+is|what\s+i\s+need\s+is|must\s+have)"
    r"\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_ATTRIBUTE_LABEL = re.compile(
    r"^(department|category|brand|color|material|style|size|feature|"
    r"use[ _-]?case|budget)\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_CLARIFICATION_SLOT_MAP: dict[ClarificationAttribute, SlotKey] = {
    "category": "category",
    "material": "material",
    "color": "color",
    "size": "size",
    "style": "style",
    "brand": "brand",
    "feature": "feature",
    "use_case": "use_case",
}


def _canonical_number(value: str) -> str:
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return value
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _budget_findings(message: str, order_offset: int = 0) -> list[_Finding]:
    findings: list[_Finding] = []
    range_matches = list(_BETWEEN_BUDGET.finditer(message))
    range_matches.extend(_CURRENCY_RANGE.finditer(message))
    for match in range_matches:
        # NOTE: a reversed range ("$100 to $50") is intentionally stored
        # verbatim, not swapped — test_owner_b_e2e_audit.py's
        # test_case_10_malformed_single_utterance_range_is_not_silently_repaired
        # locks this in explicitly (asserts price_min=100/price_max=50 and
        # the literal "between $100 and $50" canonical rendering survive
        # unchanged). An earlier version of this fix auto-swapped the
        # bounds; that broke this test, so don't reintroduce it without
        # confirming the repair is actually wanted first — silently
        # "fixing" the range also silently guesses at what the user meant.
        findings.extend(
            (
                _Finding(
                    "price_min",
                    _canonical_number(match.group(1)),
                    "budget_pattern",
                    order_offset + match.start(),
                ),
                _Finding(
                    "price_max",
                    _canonical_number(match.group(2)),
                    "budget_pattern",
                    order_offset + match.start() + 1,
                ),
            )
        )
    for pattern, slot in ((_MAX_BUDGET, "price_max"), (_MIN_BUDGET, "price_min")):
        for match in pattern.finditer(message):
            findings.append(
                _Finding(
                    slot,
                    _canonical_number(match.group(1)),
                    "budget_pattern",
                    order_offset + match.start(),
                )
            )
    approximate_matches = list(_APPROXIMATE_BUDGET.finditer(message))
    approximate_matches.extend(_APPROXIMATE_CURRENCY.finditer(message))
    for match in approximate_matches:
        findings.append(
            _Finding(
                "price_target",
                _canonical_number(match.group(1)),
                "budget_pattern",
                order_offset + match.start(),
            )
        )
    return findings


def _canonical_size(value: str) -> str:
    normalized = normalize_gazetteer_value(value)
    if normalized == "one size":
        return normalized
    if re.fullmatch(r"\d{1,2}(?:\.5)?|(?:[2-6]x|x{0,3})[sl]|m", normalized):
        return normalized.upper()
    return normalized


def _size_findings(message: str, order_offset: int = 0) -> list[_Finding]:
    return [
        _Finding(
            "size",
            _canonical_size(match.group(1)),
            "size_pattern",
            order_offset + match.start(),
        )
        for match in _SIZE_PATTERN.finditer(message)
    ]


def _phrase_positions(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    phrase_tokens = phrase.split()
    width = len(phrase_tokens)
    if not width:
        return []
    return [
        (start, start + width)
        for start in range(len(tokens) - width + 1)
        if tokens[start : start + width] == phrase_tokens
    ]


def _is_use_case_context(tokens: list[str], start: int, end: int) -> bool:
    previous = tokens[max(0, start - 2) : start]
    if previous and previous[-1] in {"for", "during", "at"}:
        return True
    if len(previous) == 2 and previous == ["for", "the"]:
        return True
    # Winter/outdoor used adjectivally before a product is still explicit
    # purpose context. Running is excluded here: "running shoes" is a
    # product/category phrase, not two independent constraints.
    return (
        tokens[start] in {"winter", "outdoor"}
        and end < len(tokens)
        and tokens[end] in _PRODUCT_NOUNS
    )


def _is_looking_for_category_context(tokens: list[str], start: int) -> bool:
    return tokens[max(0, start - 2) : start] == ["looking", "for"]


_GENERIC_BRAND_CONTROL_WORDS: frozenset[str] = frozenset(
    {"key", "not", "sole", "style"}
)
_BRAND_CONTEXT_WORDS: frozenset[str] = frozenset(
    {
        "actually",
        "brand",
        "by",
        "choose",
        "from",
        "like",
        "love",
        "mind",
        "prefer",
        "preferred",
        "want",
        "wanted",
        "wear",
    }
)


def _brand_is_ambiguous_without_syntax(
    normalized: str, gazetteer: AttributeGazetteer
) -> bool:
    """Whether a store label is also ordinary non-brand vocabulary."""
    return (
        normalized in _GENERIC_BRAND_CONTROL_WORDS
        or normalized in _FEATURE_PHRASES
        or normalized in _PRODUCT_NOUNS
        or any(
            normalized in gazetteer.get(slot, {})
            for slot in ("department", "category", "color", "material", "style", "use_case")
        )
    )


def _is_brand_context(
    tokens: list[str],
    start: int,
    end: int,
    normalized: str,
    gazetteer: AttributeGazetteer,
) -> bool:
    """Require local brand-like evidence for a generic store-name match."""
    previous = tokens[start - 1] if start else None
    if previous in {"brand", "by", "from"}:
        return True
    if _brand_is_ambiguous_without_syntax(normalized, gazetteer):
        return False
    preference_prefix = tokens[max(0, start - 3) : start]
    while preference_prefix and preference_prefix[-1] in {"a", "an", "the"}:
        preference_prefix.pop()
    if preference_prefix and preference_prefix[-1] in _BRAND_CONTEXT_WORDS:
        return True
    if tokens[end : end + 3] == ["would", "be", "better"]:
        return True
    if start == 0 and end == len(tokens):
        return True
    return any(token in _PRODUCT_NOUNS for token in tokens[end : end + 3])


def _select_scalar_candidate(
    slot: SlotKey,
    candidates: list[tuple[int, int, str]],
    tokens: list[str],
) -> tuple[int, int, str]:
    """Select one scalar using local intent cues before global label length."""
    if slot in {"department", "category"}:
        replacements = [
            candidate
            for candidate in candidates
            if candidate[1] < len(tokens) and tokens[candidate[1]] == "instead"
        ]
        if replacements:
            return max(replacements, key=lambda item: (item[0], item[1] - item[0]))

    if slot == "category":
        # The evaluator renders a broad path followed by its explicit leaf,
        # e.g. "Jackets & Vests Vests". A repeated local leaf is stronger
        # evidence than the globally longer taxonomy phrase.
        repeated = [
            candidate
            for candidate in candidates
            if candidate[0] >= candidate[1] - candidate[0]
            and tokens[candidate[0] - (candidate[1] - candidate[0]) : candidate[0]]
            == tokens[candidate[0] : candidate[1]]
        ]
        if repeated:
            return max(repeated, key=lambda item: (item[0], item[1] - item[0]))

    return max(
        candidates,
        key=lambda item: (
            item[1] - item[0],
            len(item[2]),
            -item[0],
        ),
    )


def _gazetteer_findings(
    message: str, gazetteer: AttributeGazetteer
) -> list[_Finding]:
    tokens = normalize_gazetteer_value(message).split()
    selected: list[_Finding] = []
    slot_order: tuple[SlotKey, ...] = (
        "department",
        "category",
        "brand",
        "color",
        "material",
        "style",
    )

    for slot in slot_order:
        candidates: list[tuple[int, int, str]] = []
        for normalized, canonical in gazetteer.get(slot, {}).items():
            for start, end in _phrase_positions(tokens, normalized):
                if slot == "brand" and not _is_brand_context(
                    tokens, start, end, normalized, gazetteer
                ):
                    continue
                if (
                    slot == "category"
                    and normalized in gazetteer.get("use_case", {})
                    and _is_use_case_context(tokens, start, end)
                    and not _is_looking_for_category_context(tokens, start)
                    and not (end < len(tokens) and tokens[end] in _PRODUCT_NOUNS)
                ):
                    continue
                if (
                    slot == "style"
                    and normalized in gazetteer.get("use_case", {})
                    and _is_use_case_context(tokens, start, end)
                    and not _is_looking_for_category_context(tokens, start)
                ):
                    continue
                candidates.append((start, end, canonical))

        if not candidates:
            continue
        if slot in {"department", "category", "brand"}:
            candidates = [_select_scalar_candidate(slot, candidates, tokens)]
        else:
            candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
            non_overlapping: list[tuple[int, int, str]] = []
            for candidate in candidates:
                start, end, _ = candidate
                overlaps = any(
                    start < kept_end and kept_start < end
                    for kept_start, kept_end, _ in non_overlapping
                )
                if overlaps:
                    continue
                non_overlapping.append(candidate)
            candidates = non_overlapping

        selected.extend(
            _Finding(slot, canonical, "gazetteer", start)
            for start, _, canonical in candidates
        )
    return selected


def _controlled_values(
    phrase: str, gazetteer: AttributeGazetteer, slot: SlotKey
) -> list[str]:
    tokens = normalize_gazetteer_value(phrase).split()
    values: list[tuple[int, int, str]] = []
    for normalized, canonical in gazetteer.get(slot, {}).items():
        for start, end in _phrase_positions(tokens, normalized):
            values.append((start, end, canonical))
    values.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    return list(dict.fromkeys(canonical for _, _, canonical in values))


def _style_value(phrase: str) -> str | None:
    normalized = normalize_gazetteer_value(phrase)
    for style in _STYLE_PHRASES:
        if re.search(rf"(?<!\w){re.escape(style)}(?!\w)", normalized):
            return style
    if re.search(r"\b(?:fit|sleeve|neck|style)\b", normalized):
        # Arbitrary style wording is accepted only when the evaluator has
        # explicitly framed this whole phrase as a requirement.
        return normalized
    return None


def _classify_requirement(
    phrase: str,
    gazetteer: AttributeGazetteer,
    order: int,
) -> list[_Finding]:
    cleaned = phrase.strip().strip(" .!?")
    if not cleaned:
        return []

    label_match = _ATTRIBUTE_LABEL.match(cleaned)
    if label_match:
        label = normalize_gazetteer_value(label_match.group(1)).replace(" ", "_")
        value = label_match.group(2).strip()
        if label == "budget":
            budget = _budget_findings(value, order)
            bare_money = _BARE_MONEY.fullmatch(value.strip())
            if not budget and bare_money is not None:
                budget = [
                    _Finding(
                        "price_target",
                        _canonical_number(bare_money.group(1)),
                        "attribute_label",
                        order,
                    )
                ]
            return budget
        if label == "size":
            normalized = normalize_gazetteer_value(value)
            if normalized:
                return [
                    _Finding(
                        "size", _canonical_size(normalized), "attribute_label", order
                    )
                ]
            return []
        if label in {"feature", "style", "use_case"}:
            normalized = normalize_gazetteer_value(value)
            if normalized:
                return [_Finding(label, normalized, "attribute_label", order)]
            return []
        slot = label
        canonical = lookup_gazetteer(gazetteer, slot, value)
        if canonical is None and slot in {"color", "material"}:
            # A literal attribute label supplies the disambiguation that the
            # deliberately small B2 controlled vocabulary otherwise lacks.
            # department/category/brand/style deliberately do NOT get this
            # fallback below — they are strictly catalog-backed (module
            # docstring), so a gazetteer miss there returns [] rather than
            # inventing a value. _classify_requested_requirement() applies
            # this identical color/material-only carve-out to the
            # clarification-answer path; keep the two in sync.
            canonical = normalize_gazetteer_value(value)
        return (
            [_Finding(slot, canonical, "attribute_label", order)]
            if canonical is not None
            else []
        )

    budget = _budget_findings(cleaned, order)
    if budget:
        return budget
    size = _size_findings(cleaned, order)
    if size:
        return size
    # Collect matches across BOTH slots before returning: a single phrase
    # can legitimately name a material and a color together ("cotton and
    # black"), and returning on the first non-empty slot silently dropped
    # whichever slot came second in this tuple's order.
    material_and_color: list[_Finding] = []
    next_order = order
    for slot in ("material", "color"):
        for value in _controlled_values(cleaned, gazetteer, slot):
            material_and_color.append(
                _Finding(slot, value, "requirement_phrase", next_order)
            )
            next_order += 1
    if material_and_color:
        return material_and_color
    style = _style_value(cleaned)
    if style is not None:
        return [_Finding("style", style, "requirement_phrase", order)]

    normalized = normalize_gazetteer_value(cleaned)
    use_cases = _controlled_values(normalized, gazetteer, "use_case")
    if use_cases and all(
        token in set(" ".join(use_cases).split()) | {"for", "the"}
        for token in normalized.split()
    ):
        return [
            _Finding("use_case", value, "requirement_phrase", order + index)
            for index, value in enumerate(use_cases)
        ]

    # A requirement payload is authoritative context. Exact catalogue
    # values can still name a category or a legitimate brand, but generic
    # feature/control vocabulary must not become a brand merely because a
    # noisy store label exists in the catalogue.
    category = lookup_gazetteer(gazetteer, "category", cleaned)
    if category is not None:
        return [_Finding("category", category, "requirement_phrase", order)]
    brand = lookup_gazetteer(gazetteer, "brand", cleaned)
    if brand is not None and not _brand_is_ambiguous_without_syntax(
        normalized, gazetteer
    ):
        return [_Finding("brand", brand, "requirement_phrase", order)]
    return (
        [_Finding("feature", normalized, "requirement_phrase", order)]
        if normalized
        else []
    )


def _classify_requested_requirement(
    phrase: str,
    gazetteer: AttributeGazetteer,
    requested_attribute: ClarificationAttribute,
    order: int,
) -> list[_Finding]:
    """Classify one evaluator reply value using the question that elicited it."""
    if requested_attribute == "other":
        return _classify_requirement(phrase, gazetteer, order)

    cleaned = phrase.strip().strip(" .!?")
    if not cleaned:
        return []
    label_match = _ATTRIBUTE_LABEL.match(cleaned)
    value = label_match.group(2).strip() if label_match is not None else cleaned

    if requested_attribute == "budget":
        budget = _budget_findings(value, order)
        bare_money = _BARE_MONEY.fullmatch(value.strip())
        if not budget and bare_money is not None:
            budget = [
                _Finding(
                    "price_target",
                    _canonical_number(bare_money.group(1)),
                    "clarification_context",
                    order,
                )
            ]
        return budget

    slot = _CLARIFICATION_SLOT_MAP[requested_attribute]
    normalized = normalize_gazetteer_value(value)
    if not normalized:
        return []
    if slot == "size":
        canonical = _canonical_size(normalized)
    elif slot in {"material", "color", "feature", "use_case", "style"}:
        # These slots tolerate free text beyond the fixed/gazetteer
        # vocabulary (module docstring: "Color, material, and use-case
        # values are small evaluator-aligned fixed vocabularies ... because
        # the corresponding metadata is sparse"). style is included too —
        # test_extraction.py's requested_attribute="style" case explicitly
        # expects a reply like "classic" (not in the catalogue's Style
        # vocabulary) to be accepted verbatim; only category/brand are
        # actually strict here. Matches _classify_requirement()'s
        # color/material carve-out (style there goes through _style_value()
        # separately, which is likewise never gazetteer-only).
        canonical = lookup_gazetteer(gazetteer, slot, value) or normalized
    else:
        # category/brand are strictly catalog-backed (module docstring:
        # "Catalogue-backed values come only from labelled structured
        # fields"). An answer the gazetteer doesn't recognise must not
        # become a fabricated constraint — that guarantees zero retrieval
        # matches for the rest of the session (verified: an unrecognised
        # brand answer previously got stored verbatim). Drop the finding
        # rather than inventing one.
        canonical = lookup_gazetteer(gazetteer, slot, value)
        if canonical is None:
            return []
    return [_Finding(slot, canonical, "clarification_context", order)]


def _evaluator_reply_findings(
    message: str,
    gazetteer: AttributeGazetteer,
    requested_attribute: ClarificationAttribute | None,
) -> list[_Finding] | None:
    match = _EVALUATOR_REPLY.search(message)
    if match is None:
        return None
    findings: list[_Finding] = []
    for index, value in enumerate(match.group(1).split(";")):
        order = index * 100
        if requested_attribute is None:
            classified = _classify_requirement(value, gazetteer, order)
        else:
            classified = _classify_requested_requirement(
                value, gazetteer, requested_attribute, order
            )
        findings.extend(classified)
    return findings


def _semantic_findings(
    message: str, gazetteer: AttributeGazetteer
) -> list[_Finding]:
    normalized = normalize_gazetteer_value(message)
    findings: list[_Finding] = []
    for phrase in _FEATURE_PHRASES:
        match = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized)
        if match is not None:
            findings.append(
                _Finding("feature", phrase, "requirement_phrase", match.start())
            )
    for phrase in _STYLE_PHRASES:
        match = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized)
        if match is not None:
            findings.append(
                _Finding("style", phrase, "requirement_phrase", match.start())
            )

    requirement = _EXPLICIT_REQUIREMENT.search(message)
    if requirement is not None:
        findings.extend(
            _classify_requirement(
                requirement.group(1), gazetteer, requirement.start(1)
            )
        )
    return findings


def _use_case_findings(
    message: str, gazetteer: AttributeGazetteer
) -> list[_Finding]:
    tokens = normalize_gazetteer_value(message).split()
    findings: list[_Finding] = []
    for normalized, canonical in gazetteer.get("use_case", {}).items():
        for start, end in _phrase_positions(tokens, normalized):
            if not _is_use_case_context(tokens, start, end):
                continue
            if _is_looking_for_category_context(tokens, start):
                continue
            if end < len(tokens) and tokens[end] in _PRODUCT_NOUNS:
                continue
            findings.append(
                _Finding("use_case", canonical, "use_case_pattern", start)
            )
    return findings


def _deduplicate_findings(findings: list[_Finding]) -> list[_Finding]:
    result: list[_Finding] = []
    seen: set[tuple[SlotKey, str]] = set()
    for finding in sorted(findings, key=lambda item: item.order):
        identity = (finding.slot, normalize_gazetteer_value(finding.value))
        if identity not in seen:
            seen.add(identity)
            result.append(finding)
    return result


def _build_result_slots(
    findings: list[_Finding],
) -> tuple[ExplicitSlots, list[_Finding]]:
    slots: ExplicitSlots = {}
    retained: list[_Finding] = []
    for slot in SLOT_KEYS:
        values = [finding for finding in findings if finding.slot == slot]
        if not values:
            continue
        if slot in MULTI_VALUE_SLOT_KEYS:
            slots[slot] = tuple(finding.value for finding in values)
            retained.extend(values)
        else:
            slots[slot] = values[0].value
            retained.append(values[0])
    retained.sort(key=lambda item: item.order)
    return slots, retained


def _residual_scenario(message: str, findings: list[_Finding]) -> str | None:
    normalized = normalize_gazetteer_value(message)
    tokens = normalized.split()
    if not findings:
        has_intent_for_clause = "for" in tokens and any(
            token in tokens
            for token in {"anything", "looking", "need", "something", "want"}
        )
        return (
            message.strip()
            if has_intent_for_clause or any(cue in tokens for cue in _SCENARIO_CUES)
            else None
        )

    for match in re.finditer(r"\bfor\s+([^.;!?]+)", message, re.IGNORECASE):
        clause = match.group(0).strip()
        remaining = normalize_gazetteer_value(clause)
        for finding in findings:
            value = normalize_gazetteer_value(finding.value)
            remaining = re.sub(
                rf"(?<!\w){re.escape(value)}(?!\w)", " ", remaining
            )
        remaining_tokens = [
            token
            for token in remaining.split()
            if token not in _RESIDUAL_FILLER
        ]
        if remaining_tokens:
            return clause
    return None


# B9: off by default (§8.6: "explicitly off the critical path"). When True,
# update_slots() tries extract_slots_llm() first and falls back to B3's
# extract_slots() on parse failure, timeout, or missing credentials —
# never on by default, so the deterministic B3 path stays what every
# other module is developed and tested against.
USE_LLM_EXTRACTION = False


def extract_slots_llm(
    message: str,
    gazetteer: AttributeGazetteer | None = None,
    *,
    requested_attribute: ClarificationAttribute | None = None,
) -> Optional[ExtractionResult]:
    """Constrained-JSON LLM extraction of the same flat attribute object
    extract_slots() (B3) returns, for update_slots() to try first when
    USE_LLM_EXTRACTION is on.

    Design doc §8.2 step B9: "Falls back to B3 on parse failure, timeout, or
    missing credentials." STUB: no LLM access is provided in this
    environment (§1.2, same constraint documented on rank.py's C7
    llm_rerank()) — this always returns None, reporting "missing
    credentials" so update_slots()'s fallback-to-B3 path is real and
    covered by tests even though no LLM call is ever actually made.

    Args:
        message: The raw user utterance for this turn.
        gazetteer: The B2 attribute gazetteer (unused by the stub; a real
            implementation would pass catalogue vocabulary into the
            constrained-JSON schema/prompt).
        requested_attribute: The clarification attribute this message may
            be answering (unused by the stub).

    Returns:
        None always (stub) — signals "unavailable", i.e. fall back to B3.
        A real implementation returns an ExtractionResult on a successful,
        schema-valid parse, or None on parse failure/timeout/missing
        credentials.
    """
    return None


def extract_slots(
    message: str,
    gazetteer: AttributeGazetteer | None = None,
    *,
    requested_attribute: ClarificationAttribute | None = None,
) -> ExtractionResult:
    """Extract explicit evidence from one message without touching state.

    Evaluator replies have the strongest structural signal, followed by
    explicit budget/size patterns, controlled vocabulary matches, contextual
    use cases, and narrowly defined feature/style phrases. No profile or
    catalogue-derived semantic inference is performed.

    ``update_slots`` consumes this pure result: B5 plans explicit-slot
    operations, B4 applies them, and B6 transitions the scenario buffer.

    The optional gazetteer preserves the existing one-message call shape. A
    caller that already owns the catalogue (such as ``Agent``) should pass its
    prebuilt B2 gazetteer; the fallback uses the repository's shared loader.
    """
    if gazetteer is None:
        from utils import load_catalog

        gazetteer = build_attribute_gazetteer(load_catalog())

    evaluator_findings = _evaluator_reply_findings(
        message, gazetteer, requested_attribute
    )
    if evaluator_findings is not None:
        findings = evaluator_findings
        residual = None
    else:
        findings = []
        findings.extend(_budget_findings(message))
        findings.extend(_size_findings(message))
        requirement = _EXPLICIT_REQUIREMENT.search(message)
        generic_scope = (
            message
            if requirement is None
            else message[: requirement.start(1)]
        )
        findings.extend(_gazetteer_findings(generic_scope, gazetteer))
        findings.extend(_use_case_findings(message, gazetteer))
        findings.extend(_semantic_findings(message, gazetteer))
        # _residual_scenario() only reads `findings` (membership of each
        # finding's value, order-independent) and never mutates it, so
        # deduplicating here as well as below was pure duplicate work on
        # every ordinary user turn — one _deduplicate_findings() call
        # covers both branches.
        residual = _residual_scenario(message, findings)

    findings = _deduplicate_findings(findings)
    slots, retained = _build_result_slots(findings)
    observations = tuple(
        SlotObservation(finding.slot, finding.value, finding.source)
        for finding in retained
    )
    return ExtractionResult(slots, residual, observations)


# ---------------------------------------------------------------------------
# B4: deterministic structured state transitions
# ---------------------------------------------------------------------------
SlotOperationKind: TypeAlias = Literal[
    "upsert",
    "replace",
    "delete_value",
    "delete_slot",
]


@dataclass(frozen=True)
class SlotOperation:
    """One structured state transition, produced without language inference.

    ``upsert`` is the positive B3 path. For scalar slots it writes or
    overwrites; for multi-value slots it adds. B5 may later emit ``replace``,
    ``delete_value``, or ``delete_slot`` after separately interpreting the
    utterance.
    """

    kind: SlotOperationKind
    slot: SlotKey
    values: tuple[str, ...] = ()


def _operation_values(operation: SlotOperation) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for raw_value in operation.values:
        value = _canonical_text(raw_value)
        identity = normalize_gazetteer_value(value)
        if not identity:
            raise ValueError("slot operation values must not be empty")
        if identity not in seen:
            seen.add(identity)
            values.append(value)
    return tuple(values)


def _current_values(state: SessionState, slot: SlotKey) -> tuple[str, ...]:
    current = state.slots.get(slot)
    if current is None:
        return ()
    return current if isinstance(current, tuple) else (current,)


def _value_identities(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(normalize_gazetteer_value(value) for value in values)


def apply_slot_operation(
    state: SessionState, operation: SlotOperation
) -> SessionState:
    """Apply one validated structured operation to current explicit state."""
    if operation.slot not in SLOT_KEYS:
        raise ValueError(f"unknown slot: {operation.slot}")

    is_multi = operation.slot in MULTI_VALUE_SLOT_KEYS
    values = _operation_values(operation)
    current = _current_values(state, operation.slot)

    if operation.kind == "delete_slot":
        if values:
            raise ValueError("delete_slot does not accept values")
        state.slots.pop(operation.slot, None)
        state.slot_override_flags.pop(operation.slot, None)
        return state

    if not values:
        raise ValueError(f"{operation.kind} requires at least one value")
    if not is_multi and len(values) != 1:
        raise ValueError("single-valued slot operations require exactly one value")
    if operation.kind == "delete_value" and not is_multi:
        raise ValueError("delete_value is only valid for multi-valued slots")

    if operation.kind == "delete_value":
        removed = set(_value_identities(values))
        remaining = tuple(
            value
            for value in current
            if normalize_gazetteer_value(value) not in removed
        )
        if remaining == current:
            return state
        if not remaining:
            state.slots.pop(operation.slot, None)
            state.slot_override_flags.pop(operation.slot, None)
        else:
            state.slots[operation.slot] = remaining
            state.slot_override_flags[operation.slot] = True
        return state

    if operation.kind == "replace":
        if not current:
            state.slots[operation.slot] = values if is_multi else values[0]
            state.slot_override_flags[operation.slot] = False
            return state
        current_ids = _value_identities(current)
        replacement_ids = _value_identities(values)
        equivalent = (
            set(current_ids) == set(replacement_ids)
            if is_multi
            else current_ids == replacement_ids
        )
        if equivalent:
            return state
        state.slots[operation.slot] = values if is_multi else values[0]
        state.slot_override_flags[operation.slot] = True
        return state

    if operation.kind != "upsert":
        raise ValueError(f"unknown slot operation: {operation.kind}")

    if not current:
        state.slots[operation.slot] = values if is_multi else values[0]
        state.slot_override_flags[operation.slot] = False
        return state
    if not is_multi:
        if _value_identities(current) != _value_identities(values):
            state.slots[operation.slot] = values[0]
            state.slot_override_flags[operation.slot] = True
        return state

    active_identities = set(_value_identities(current))
    additions = tuple(
        value
        for value in values
        if normalize_gazetteer_value(value) not in active_identities
    )
    if additions:
        state.slots[operation.slot] = current + additions
        state.slot_override_flags.setdefault(operation.slot, False)
    return state


def apply_slot_operations(
    state: SessionState, operations: Iterable[SlotOperation]
) -> SessionState:
    """Apply structured operations in their supplied deterministic order."""
    for operation in operations:
        apply_slot_operation(state, operation)
    return state


def operations_from_extraction(
    extraction: ExtractionResult,
) -> tuple[SlotOperation, ...]:
    """Convert positive B3 evidence into B4 upsert operations."""
    return tuple(
        SlotOperation(
            "upsert",
            slot,
            value if isinstance(value, tuple) else (value,),
        )
        for slot, value in extraction.slots.items()
    )


def apply_extraction_result(
    state: SessionState, extraction: ExtractionResult
) -> SessionState:
    """Apply only B3's positive slots; leave residual scenario untouched."""
    return apply_slot_operations(state, operations_from_extraction(extraction))


# ---------------------------------------------------------------------------
# B5 V1: conservative clause-local negation and replacement planning
# ---------------------------------------------------------------------------
_SLOT_LABELS: dict[SlotKey, tuple[str, ...]] = {
    "department": ("department",),
    "category": ("category",),
    "brand": ("brand",),
    "price_min": ("minimum price", "price minimum"),
    "price_max": ("maximum price", "price maximum"),
    "price_target": ("target price", "price target"),
    "color": ("color", "colour"),
    "material": ("material",),
    "style": ("style",),
    "size": ("size",),
    "feature": ("feature",),
    "use_case": ("use case",),
}
_FALSE_NEGATION_PREFIXES: tuple[str, ...] = (
    "not sure",
    "not really",
    "i dont know",
    "dont know",
    "nothing too",
)
_REJECTION_PREFIXES: tuple[tuple[str, bool], ...] = (
    ("not", False),
    ("dont want", False),
    ("do not want", False),
    ("dont need", False),
    ("do not need", False),
    ("no", False),
    ("without", False),
    ("forget", False),
    ("instead of", True),
    ("rather than", True),
)
_EVALUATOR_INITIAL_OVERRIDE = re.compile(
    r"^\s*i['’]m\s+looking\s+for\s+.+?\.\s*(?P<preference>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_EVALUATOR_OVERRIDE_FALLBACK = (
    "actually please ignore my earlier preference"
)
_EVALUATOR_OVERRIDE_WITH_VALUE = (
    "actually ignore my earlier preference what i need is "
)
_PRICE_SLOT_KEYS: tuple[SlotKey, ...] = (
    "price_min",
    "price_max",
    "price_target",
)
_PRICE_REVISION_CUE = re.compile(
    r"(?:^\s*actually\b|\binstead\b|\brather\s+than\b|"
    r"\bforget\b|\bchange\s+(?:(?:the|my)\s+)?budget\s+to\b|"
    r"^\s*no\s*[,;:])",
    re.IGNORECASE,
)
_PRICE_ADDITION_CUE = re.compile(
    r"\b(?:also|too|as\s+well|in\s+addition)\b",
    re.IGNORECASE,
)


def _message_clauses(message: str) -> list[str]:
    return [
        normalized
        for raw_clause in re.split(
            r"\s*(?:[,;.!?]+|\bbut\b)\s*", message, flags=re.IGNORECASE
        )
        if (normalized := normalize_gazetteer_value(raw_clause))
    ]


def _mention_forms(
    slot: SlotKey,
    value: str,
    gazetteer: AttributeGazetteer | None,
) -> tuple[str, ...]:
    canonical = normalize_gazetteer_value(value)
    forms = {canonical}
    if gazetteer is not None:
        for lookup, stored in gazetteer.get(slot, {}).items():
            if normalize_gazetteer_value(stored) == canonical:
                forms.add(lookup)
    return tuple(sorted(forms, key=lambda form: (-len(form.split()), form)))


def _contains_form(clause: str, form: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(form)}(?!\w)", clause) is not None


def _uncertain_negation_clause(clause: str) -> bool:
    return clause.startswith(_FALSE_NEGATION_PREFIXES)


def _rejection_kind(clause: str, form: str) -> tuple[bool, bool]:
    """Return (is_rejected, starts_explicit_replacement)."""
    if _uncertain_negation_clause(clause) or "dont mind" in clause:
        return False, False
    ending = rf"{re.escape(form)}(?: anymore)?$"
    for prefix, replacement in _REJECTION_PREFIXES:
        if re.search(rf"(?:^|\s){re.escape(prefix)}\s+{ending}", clause):
            return True, replacement
    if re.search(
        rf"(?:^|\s){re.escape(form)}\s+(?:doesnt|does not)\s+"
        r"matter\s+anymore$",
        clause,
    ):
        return True, False
    if re.search(
        rf"(?:^|\s){re.escape(form)}\s+(?:isnt|is not)\s+ideal$",
        clause,
    ):
        return True, False
    return False, False


def _has_ambiguous_negative_scope(clause: str, form: str) -> bool:
    if _uncertain_negation_clause(clause):
        return True
    if "dont mind" in clause:
        # Indifference is neither a rejection nor a positive constraint.
        # B3 may still observe a controlled value such as a brand, but B5
        # must not let that observation become an explicit-state upsert.
        return True
    return any(
        re.search(
            rf"(?:^|\s){re.escape(prefix)}\s+{re.escape(form)}(?:\s|$)",
            clause,
        )
        is not None
        for prefix, _ in _REJECTION_PREFIXES
    )


def _whole_slot_rejections(
    clauses: list[str], state: SessionState
) -> set[SlotKey]:
    rejected: set[SlotKey] = set()
    for slot, labels in _SLOT_LABELS.items():
        if slot not in state.slots:
            continue
        for clause in clauses:
            for label in labels:
                escaped = re.escape(label)
                dont_care = re.search(
                    rf"(?:^|\s)(?:i\s+)?(?:dont|do not)\s+care\s+about\s+"
                    rf"(?:the\s+)?{escaped}(?:\s+anymore)?$",
                    clause,
                )
                doesnt_matter = re.search(
                    rf"^{escaped}\s+(?:doesnt|does not)\s+matter\s+anymore$",
                    clause,
                )
                if dont_care is not None or doesnt_matter is not None:
                    rejected.add(slot)
    return rejected


def _extracted_slot_values(
    extraction: ExtractionResult,
) -> dict[SlotKey, tuple[str, ...]]:
    extracted = {
        slot: value if isinstance(value, tuple) else (value,)
        for slot, value in extraction.slots.items()
    }
    # The evaluator's full correction template can make B3 see the same word
    # once through a controlled slot and once through the generic requirement
    # fallback. Prefer the specific observation rather than writing a duplicate
    # feature (for example brand=Adidas plus feature=adidas).
    specific_ids = {
        normalize_gazetteer_value(value)
        for slot, values in extracted.items()
        if slot != "feature"
        for value in values
    }
    if "feature" in extracted:
        features = tuple(
            value
            for value in extracted["feature"]
            if normalize_gazetteer_value(value) not in specific_ids
        )
        if features:
            extracted["feature"] = features
        else:
            del extracted["feature"]
    return extracted


def _evaluator_override_message(message: str) -> bool:
    normalized = normalize_gazetteer_value(message)
    return normalized == _EVALUATOR_OVERRIDE_FALLBACK or normalized.startswith(
        _EVALUATOR_OVERRIDE_WITH_VALUE
    )


def _price_family_revision_cleanup(
    message: str,
    state: SessionState,
    extracted: dict[SlotKey, tuple[str, ...]],
) -> tuple[SlotOperation, ...]:
    """Plan removal of obsolete price-family members for a strong revision.

    Price fields may coexist, so a new price observation normally accumulates.
    A strong correction cue without additive language instead makes the price
    fields present in this utterance the new active price representation.
    Exact evaluator override scaffolds are excluded because their narrower
    value-level provenance is authoritative and must preserve later prices.
    """
    observed = {slot for slot in _PRICE_SLOT_KEYS if slot in extracted}
    if (
        not observed
        or _evaluator_override_message(message)
        or _PRICE_REVISION_CUE.search(message) is None
        or _PRICE_ADDITION_CUE.search(message) is not None
    ):
        return ()
    return tuple(
        SlotOperation("delete_slot", slot)
        for slot in _PRICE_SLOT_KEYS
        if slot in state.slots and slot not in observed
    )


def _initial_override_reference_values(
    message: str,
    gazetteer: AttributeGazetteer,
) -> OverrideReferenceValues | None:
    """Extract only the trailing preference from the evaluator's turn 1 form."""
    match = _EVALUATOR_INITIAL_OVERRIDE.fullmatch(message)
    if match is None:
        return None
    preference = match.group("preference")
    if normalize_gazetteer_value(preference).startswith(
        "a key requirement is"
    ):
        # This is the evaluator's buying template, not intent_override.
        return None
    extraction = extract_slots(preference, gazetteer)
    extracted = _extracted_slot_values(extraction)
    reference_values: OverrideReferenceValues = {
        slot: values
        for slot, values in extracted.items()
        if slot not in ("department", "category")
    }
    if reference_values:
        return reference_values

    # Some evaluator old preferences are arbitrary catalogue feature text
    # (for example "Pull On closure") that B3 correctly refuses to add as
    # a generic live constraint. Keep only a provenance reference so an
    # exact matching value learned later through a feature clarification can
    # still be invalidated by the evaluator's explicit override scaffold.
    fallback = _classify_requirement(preference, gazetteer, 0)
    fallback_slots, _ = _build_result_slots(_deduplicate_findings(fallback))
    return {
        slot: value if isinstance(value, tuple) else (value,)
        for slot, value in fallback_slots.items()
        if slot not in ("department", "category")
    }


def _override_reference_deletions(
    state: SessionState,
    operations: tuple[SlotOperation, ...],
) -> tuple[SlotOperation, ...]:
    """Invalidate only still-active values traced to the old preference."""
    fully_revised_slots = {
        operation.slot
        for operation in operations
        if operation.kind in ("replace", "delete_slot")
    }
    deletions: list[SlotOperation] = []
    for slot in SLOT_KEYS:
        if slot in fully_revised_slots:
            continue
        referenced = state.override_reference_values.get(slot, ())
        if not referenced:
            continue
        referenced_ids = set(_value_identities(referenced))
        matching_active = tuple(
            value
            for value in _current_values(state, slot)
            if normalize_gazetteer_value(value) in referenced_ids
        )
        if not matching_active:
            continue
        if slot in MULTI_VALUE_SLOT_KEYS:
            deletions.append(
                SlotOperation("delete_value", slot, matching_active)
            )
        else:
            deletions.append(SlotOperation("delete_slot", slot))
    return tuple(deletions)


def _replacement_positive(
    slot: SlotKey,
    clauses: list[str],
    clause_indexes: list[int],
    forms: tuple[str, ...],
    rejection_clauses: set[int],
    replacement_rejections: set[int],
) -> bool:
    for index in clause_indexes:
        clause = clauses[index]
        for form in forms:
            escaped = re.escape(form)
            if re.search(rf"(?<!\w){escaped}\s+instead(?:\s|$)", clause):
                return True
            if index - 1 in rejection_clauses and re.search(
                rf"(?:^|\s)just\s+{escaped}(?!\w)", clause
            ):
                return True
            if (
                slot not in MULTI_VALUE_SLOT_KEYS
                or index - 1 in rejection_clauses
            ) and re.search(
                rf"(?<!\w){escaped}\s+would\s+be\s+better(?:\s|$)", clause
            ):
                return True
            if slot not in MULTI_VALUE_SLOT_KEYS and re.search(
                rf"(?:^|\s)actually\s+{escaped}(?!\w)", clause
            ):
                return True
        if index - 1 in replacement_rejections:
            return True
        if clause.startswith("what i need is") and any(
            prior == "actually" or prior.startswith("ignore my earlier preference")
            for prior in clauses[max(0, index - 2) : index]
        ):
            return True
    return False


def detect_slot_operations(
    message: str,
    state: SessionState,
    extraction: ExtractionResult,
    gazetteer: AttributeGazetteer | None = None,
) -> tuple[SlotOperation, ...]:
    """Plan conservative B5 operations without mutating session state.

    The current state is an allow-list for deletion: an explicitly rejected
    value is deleted only when it is active. Rejected inactive observations
    are suppressed so the positive-only state never stores ``not X``.
    """
    clauses = _message_clauses(message)
    extracted = _extracted_slot_values(extraction)
    whole_slot_rejections = _whole_slot_rejections(clauses, state)
    operations: list[SlotOperation] = []

    for slot in SLOT_KEYS:
        if slot in whole_slot_rejections:
            operations.append(SlotOperation("delete_slot", slot))
            continue

        active_values = _current_values(state, slot)
        observed_values = extracted.get(slot, ())
        candidate_values = tuple(dict.fromkeys(active_values + observed_values))
        rejected_ids: set[str] = set()
        rejection_clauses: set[int] = set()
        replacement_rejections: set[int] = set()

        for value in candidate_values:
            identity = normalize_gazetteer_value(value)
            forms = _mention_forms(slot, value, gazetteer)
            for index, clause in enumerate(clauses):
                for form in forms:
                    rejected, replacement = _rejection_kind(clause, form)
                    if rejected:
                        rejected_ids.add(identity)
                        rejection_clauses.add(index)
                        if replacement:
                            replacement_rejections.add(index)

        positive_values: list[str] = []
        replacement_values: list[str] = []
        for value in observed_values:
            identity = normalize_gazetteer_value(value)
            if identity in rejected_ids:
                continue
            forms = _mention_forms(slot, value, gazetteer)
            clause_indexes = [
                index
                for index, clause in enumerate(clauses)
                if any(_contains_form(clause, form) for form in forms)
            ]
            has_positive_scope = any(
                not any(
                    _has_ambiguous_negative_scope(clauses[index], form)
                    for form in forms
                    if _contains_form(clauses[index], form)
                )
                for index in clause_indexes
            )
            if clause_indexes and not has_positive_scope:
                continue
            positive_values.append(value)
            if active_values and _replacement_positive(
                slot,
                clauses,
                clause_indexes,
                forms,
                rejection_clauses,
                replacement_rejections,
            ):
                replacement_values.append(value)

        if replacement_values:
            operations.append(SlotOperation("replace", slot, tuple(replacement_values)))
            # A replacement classification for one value doesn't make every
            # other positively-mentioned value in this utterance a
            # replacement too ("red instead, blue too" replaces with red
            # but also adds blue). apply_slot_operations() applies
            # operations in order and each one re-reads live state, so a
            # trailing upsert here correctly adds on top of the replace
            # above rather than being dropped.
            leftover_additions = tuple(
                value for value in positive_values if value not in replacement_values
            )
            if leftover_additions:
                operations.append(SlotOperation("upsert", slot, leftover_additions))
            continue

        active_ids = set(_value_identities(active_values))
        rejected_active = tuple(
            value
            for value in active_values
            if normalize_gazetteer_value(value) in rejected_ids
        )
        if rejected_active:
            if slot in MULTI_VALUE_SLOT_KEYS:
                operations.append(
                    SlotOperation("delete_value", slot, rejected_active)
                )
            else:
                operations.append(SlotOperation("delete_slot", slot))

        additions = tuple(
            value
            for value in positive_values
            if normalize_gazetteer_value(value) not in rejected_ids
            and (
                normalize_gazetteer_value(value) not in active_ids
                or not rejected_active
            )
        )
        if additions:
            operations.append(SlotOperation("upsert", slot, additions))
        elif positive_values and not rejected_active:
            operations.append(SlotOperation("upsert", slot, tuple(positive_values)))

    planned = (
        _price_family_revision_cleanup(message, state, extracted)
        + tuple(operations)
    )
    if _evaluator_override_message(message):
        return _override_reference_deletions(state, planned) + planned
    return planned


# ---------------------------------------------------------------------------
# B6: deterministic current-scenario transitions
# ---------------------------------------------------------------------------
_SCENARIO_DETAIL_SEPARATOR = " — "
_ADDITIVE_SCENARIO_DETAIL = re.compile(
    r"^\s*(?:with|while)\b", re.IGNORECASE
)
_SCENARIO_REPLACEMENT_PURPOSE = re.compile(
    r"\b(?P<lead>(?:actually\s+)?(?:this|it)\s+is(?:\s+actually)?)"
    r"\s+for\s+(?P<context>[^.;!?]+)",
    re.IGNORECASE,
)
_SCENARIO_NOT_FOR = re.compile(
    r"^(?:its|it is|this is)\s+not\s+for\s+(?:the\s+)?"
    r"(?P<reference>.+?)\s+anymore$"
)
_SCENARIO_FORGET = re.compile(
    r"^forget\s+(?:the\s+)?(?P<reference>.+)$"
)
_SCENARIO_REFERENCE_FILLER = frozenset(
    {"a", "an", "the", "part", "scenario", "context", "purpose"}
)
_STRUCTURED_SCENARIO_FILLER = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "during",
        "for",
        "in",
        "instead",
        "or",
        "the",
        "trip",
        "with",
    }
)


def _scenario_rejection_reference(message: str) -> str | None:
    normalized = normalize_gazetteer_value(message)
    match = _SCENARIO_NOT_FOR.fullmatch(normalized)
    if match is not None:
        return match.group("reference")
    match = _SCENARIO_FORGET.fullmatch(normalized)
    if match is None:
        return None
    return re.sub(r"\s+part$", "", match.group("reference")).strip()


def _reference_matches_scenario(reference: str, current_scenario: str) -> bool:
    reference_tokens = {
        token
        for token in normalize_gazetteer_value(reference).split()
        if token not in _SCENARIO_REFERENCE_FILLER
    }
    current_tokens = set(normalize_gazetteer_value(current_scenario).split())
    return bool(reference_tokens) and reference_tokens <= current_tokens


def _replacement_scenario_candidate(message: str) -> str | None:
    for match in _SCENARIO_REPLACEMENT_PURPOSE.finditer(message):
        lead = normalize_gazetteer_value(match.group("lead"))
        context = match.group("context").strip()
        normalized_context = normalize_gazetteer_value(context)
        if "actually" not in lead and not normalized_context.endswith(" instead"):
            continue
        context = re.sub(
            r"\s+instead\s*$", "", context, flags=re.IGNORECASE
        ).strip(" -—,\t\n")
        if context:
            return f"for {context}"
    return None


def _scenario_beyond_structured_use_case(
    text: str,
    extraction: ExtractionResult,
) -> str | None:
    use_cases = extraction.slots.get("use_case")
    if use_cases is None:
        return text.strip() or None
    values = use_cases if isinstance(use_cases, tuple) else (use_cases,)
    remaining = normalize_gazetteer_value(text)
    for value in values:
        normalized_value = normalize_gazetteer_value(value)
        remaining = re.sub(
            rf"(?<!\w){re.escape(normalized_value)}(?!\w)", " ", remaining
        )
    tokens = remaining.split()
    while tokens and tokens[0] in {"a", "an", "for", "the"}:
        tokens.pop(0)
    while tokens and tokens[-1] == "instead":
        tokens.pop()
    meaningful = [
        token for token in tokens if token not in _STRUCTURED_SCENARIO_FILLER
    ]
    return " ".join(tokens) if meaningful else None


def _combine_scenario_detail(current_scenario: str, detail: str) -> str:
    base = current_scenario.split(_SCENARIO_DETAIL_SEPARATOR, 1)[0]
    base = base.rstrip().rstrip(".!?")
    detail = detail.strip()
    return f"{base}{_SCENARIO_DETAIL_SEPARATOR}{detail}"


def update_scenario_buffer(
    current_scenario: str,
    message: str,
    extraction: ExtractionResult,
) -> str:
    """Return the deterministic next value of the current scenario buffer.

    ``None`` residual evidence means preserve, not clear. Strong evaluator
    control scaffolds are ignored. An explicit purpose revision replaces the
    current scenario, including clearing it when the new purpose is completely
    represented by ``use_case``. Matching anchored rejection clears. Leading
    ``with``/``while`` detail retains one base plus only the latest detail.
    """
    current = current_scenario.strip()
    if _evaluator_override_message(message):
        return current

    rejected_reference = _scenario_rejection_reference(message)
    if rejected_reference is not None:
        return (
            ""
            if _reference_matches_scenario(rejected_reference, current)
            else current
        )

    replacement = _replacement_scenario_candidate(message)
    if replacement is not None:
        return _scenario_beyond_structured_use_case(replacement, extraction) or ""

    residual = extraction.residual_scenario
    if residual is not None:
        candidate = _scenario_beyond_structured_use_case(residual, extraction)
        if candidate is None:
            return current
        if _ADDITIVE_SCENARIO_DETAIL.match(message) and current:
            return _combine_scenario_detail(current, candidate)
        return candidate

    if _ADDITIVE_SCENARIO_DETAIL.match(message) and current:
        candidate = _scenario_beyond_structured_use_case(message, extraction)
        if candidate is not None:
            return _combine_scenario_detail(current, candidate)
    return current


def update_slots(
    state: SessionState,
    message: str,
    gazetteer: AttributeGazetteer | None = None,
    *,
    use_llm_extraction: bool = USE_LLM_EXTRACTION,
) -> SessionState:
    """Merge one utterance's extracted attributes into the slot dictionary.

    Consumes one pending clarification, performs B3 (or, when
    `use_llm_extraction` is on, B9-then-B3) extraction, asks B5 to plan
    structured operations from the pre-transition state, delegates slot
    mutation to B4, then applies B6's independent scenario transition.

    Args:
        state: The session state to update, mutated in place.
        message: The raw user utterance for this turn.
        use_llm_extraction: When True, tries extract_slots_llm() (B9)
            first and falls back to extract_slots() (B3) only if it
            returns None (parse failure/timeout/missing credentials).
            Defaults to USE_LLM_EXTRACTION (off) — B9 is off the critical
            path and, in this environment, always falls back (no LLM
            access provided; see extract_slots_llm()'s docstring).

    Returns:
        The same SessionState with explicit slots updated.
    """
    if gazetteer is None:
        from utils import load_catalog

        gazetteer = build_attribute_gazetteer(load_catalog())

    had_no_slots = not state.slots
    requested_attribute = consume_pending_clarification(state)
    extraction = None
    if use_llm_extraction:
        extraction = extract_slots_llm(
            message, gazetteer, requested_attribute=requested_attribute
        )
    if extraction is None:
        extraction = extract_slots(
            message,
            gazetteer,
            requested_attribute=requested_attribute,
        )
    if had_no_slots:
        reference_values = _initial_override_reference_values(message, gazetteer)
        if reference_values is not None:
            state.override_reference_values = reference_values
    consumes_override_reference = _evaluator_override_message(message)
    operations = detect_slot_operations(message, state, extraction, gazetteer)
    apply_slot_operations(state, operations)
    if consumes_override_reference:
        state.override_reference_values.clear()
    set_scenario(
        state,
        update_scenario_buffer(state.scenario_buffer, message, extraction),
    )
    return state
