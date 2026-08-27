"""The ten ranking features from §3.4 Step 6.

Design doc §3.4 Step 6 (Ranking) feature table.

Owner: Emerson (Ranking). §8.3, step C2 ("Implement the ten feature functions.
Each independently unit-testable. price_fit returns a neutral value for
nulls — never excludes."), step C4 ("Handle missing stream scores. Default
to zero.").

Nine of the ten feature functions are implemented for real below (issues
#20-#23 / C1-C4). price_fit is deliberately left as it was: its null-safety
branch (0.5 when price is None, §2.2) is real and must not regress per
issue #21 step 3, but its priced-item branch remains a fixture_score()
stub — issue #21 scoped only the other seven functions (plus pop/rating,
already real) and does not define an actual budget-fit formula.

rating_style_fit reads a new `SessionState.rating_style` field that is not
part of Qikun's original B1 schema (§3.3 lists only slots/scenario_buffer/
profile_terms/telemetry) — added in state.py because this feature has no
other way to reach it. See that field's docstring in state.py for why.
"""

from __future__ import annotations

from indexes import Indexes
from state import SessionState
from utils import Candidate, fixture_score

# Order matters: this is the feature vector's column order everywhere
# (telemetry rows, the fitted regression's coefficients, rank.py's
# HANDSET_WEIGHTS). §3.4 Step 6: "Ten features."
FEATURE_NAMES: tuple[str, ...] = (
    "bm25_norm",
    "cos_sim",
    "pop",
    "rating",
    "price_fit",
    "category_match",
    "brand_match",
    "slot_coverage",
    "rare_tag_match",
    "rating_style_fit",
)


def bm25_norm(candidate: Candidate, pool: list[Candidate]) -> float:
    """Sign-corrected, max-normalised FTS5 score. §3.4 Step 6: lexical relevance.

    Divides `candidate.bm25_raw` by the maximum `bm25_raw` across `pool`,
    so the most keyword-relevant candidate in this turn's pool scores 1.0
    and everything else is relative to it. A candidate contributed by no
    keyword stream carries `bm25_raw == 0.0` (§8.3 step C4) and scores 0.0
    here without any special-casing.

    Args:
        candidate: The candidate to score.
        pool: The full candidate pool, for max-normalisation.

    Returns:
        A float in [0, 1]. 0.0 if no candidate in the pool has a keyword
        score (empty pool, or every candidate is popularity/semantic-only).
    """
    max_raw = max((c.bm25_raw for c in pool), default=0.0)
    if max_raw <= 0.0:
        return 0.0
    return candidate.bm25_raw / max_raw


def cos_sim(candidate: Candidate) -> float:
    """Dot product with the query vector. §3.4 Step 6: semantic relevance.

    Both the catalogue embeddings and the canonical-intent embedding are
    L2-normalised (§3.2 Index 2), so `candidate.cos_raw` already *is* the
    cosine similarity — there is nothing left to compute here. A candidate
    contributed by no semantic stream carries `cos_raw == 0.0` (§8.3 step
    C4) by construction, not a special case in this function.

    Args:
        candidate: The candidate to score.

    Returns:
        `candidate.cos_raw` directly, intended range [-1, 1].
    """
    return candidate.cos_raw


def pop(candidate: Candidate, facts: dict[str, dict]) -> float:
    """log1p(rating_number)/log1p(1e5). §3.4 Step 6, §2.1: purchase-frequency prior.

    STUB: looks up the real precomputed `pop` value from `facts` when
    present (indexes.build_facts_dict() already computes this for real),
    falling back to a fixture pseudo-score otherwise.

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts (indexes.Indexes.facts).

    Returns:
        A float, intended range [0, 1].
    """
    record = facts.get(candidate.asin)
    if record is not None and "pop" in record:
        return record["pop"]
    return fixture_score("pop:" + candidate.asin)


def rating(candidate: Candidate, facts: dict[str, dict]) -> float:
    """average_rating / 5. §3.4 Step 6: quality signal.

    STUB: divides the real `rating` field from `facts` by 5 when present
    (a one-line normalisation, not a design decision), else a fixture
    pseudo-score.

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts.

    Returns:
        A float, intended range [0, 1].
    """
    record = facts.get(candidate.asin)
    if record is not None and record.get("rating") is not None:
        return record["rating"] / 5.0
    return fixture_score("rating:" + candidate.asin)


# Strictly interior — never 0.0 or 1.0 — so a missing price (78.9% of the
# catalogue, §2.2) reads as genuinely uninformative rather than as an
# implicit worst-possible-fit penalty. Named rather than inlined so the
# invariant is checkable by name (tests/test_features.py) instead of by
# grep for a bare "0.5" that could later be confused with a real value.
PRICE_FIT_NEUTRAL = 0.5


def price_fit(candidate: Candidate, facts: dict[str, dict], state: SessionState) -> float:
    """PRICE_FIT_NEUTRAL when price is null, else fit to stated budget. §2.2, §3.4 Step 6.

    Design doc §2.2: "78.9% of catalogue rows have price: null ... Price is
    therefore a scoring feature with a neutral value for nulls, and never a
    constraint." The null-safety itself is implemented for real below,
    since leaving it as a fixture would misrepresent the one invariant this
    feature exists to guarantee; the actual budget-fit formula is not.

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts.
        state: Current session state (would supply price_min/price_max).

    Returns:
        PRICE_FIT_NEUTRAL if price is null; otherwise a fixture pseudo-score
        in [0, 1] (the real budget-fit formula is unimplemented — issue #21
        step 3 scoped only the null-safety branch).
    """
    record = facts.get(candidate.asin)
    if record is None or record.get("price") is None:
        return PRICE_FIT_NEUTRAL
    return fixture_score("price_fit:" + candidate.asin)


def category_match(candidate: Candidate, facts: dict[str, dict], state: SessionState) -> float:
    """Whether the candidate matches the slot category. §3.4 Step 6: constraint satisfaction.

    Case-insensitive substring match of `state.slots["category"]` against
    `facts[asin]["cat3"]`, not equality — `cat3` is a full category path
    ("... > Men > Clothing > Jackets & Coats") while the slot holds a leaf
    noun ("jacket"). §3.4 Step 3: category leans, it never excludes; this
    function only ever contributes a score, so a mismatch still returns a
    value (0.0) rather than raising or signalling "drop this candidate" —
    any actual filtering happens upstream in retrieval.py, not here.

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts.
        state: Current session state.

    Returns:
        1.0 on a match, 0.0 otherwise (including no category slot set, or
        the candidate missing from facts).
    """
    category = state.slots.get("category")
    if not isinstance(category, str) or not category.strip():
        return 0.0
    record = facts.get(candidate.asin)
    if record is None:
        return 0.0
    cat3 = (record.get("cat3") or "").lower()
    return 1.0 if category.strip().lower() in cat3 else 0.0


def brand_match(candidate: Candidate, facts: dict[str, dict], state: SessionState) -> float:
    """Whether `store` matches the slot brand. §3.4 Step 6: constraint satisfaction.

    Case-insensitive substring match of `state.slots["brand"]` against
    `facts[asin]["store"]` (§2.3: store is the 99.4%-populated brand
    proxy). Substring rather than equality for the same reason as
    category_match — stated brand and the catalogue's store string are not
    always typed identically ("Nike" vs "Nike USA").

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts.
        state: Current session state.

    Returns:
        1.0 on a match, 0.0 otherwise.
    """
    brand = state.slots.get("brand")
    if not isinstance(brand, str) or not brand.strip():
        return 0.0
    record = facts.get(candidate.asin)
    if record is None or not record.get("store"):
        return 0.0
    store = record["store"].lower()
    return 1.0 if brand.strip().lower() in store else 0.0


def slot_coverage(candidate: Candidate, facts: dict[str, dict], state: SessionState) -> float:
    """Fraction of slot terms present in the text blob. §2.3, §3.4 Step 6.

    Design doc §2.3: "Attribute matching must therefore operate over text,
    not structured lookup — which makes slot matching a scoring signal
    rather than a filter." Every value in `state.slots` is checked as a
    lowercase substring of `facts[asin]["blob"]` (multi-value slots are
    flattened to their individual terms first); the return is the fraction
    found.

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts.
        state: Current session state.

    Returns:
        A float in [0, 1]. 0.0 if no slots are set yet, rather than
        dividing by zero.
    """
    terms: list[str] = []
    for value in state.slots.values():
        values = value if isinstance(value, tuple) else (value,)
        terms.extend(v.strip().lower() for v in values if isinstance(v, str) and v.strip())
    if not terms:
        return 0.0
    record = facts.get(candidate.asin)
    blob = (record.get("blob") or "").lower() if record is not None else ""
    hits = sum(1 for term in terms if term in blob)
    return hits / len(terms)


def rare_tag_match(candidate: Candidate, facts: dict[str, dict], state: SessionState) -> float:
    """1 if any rare-tag term is present, else 0. §2.4, §3.4 Step 6.

    Design doc §2.4: of eight preference tags, only performance/warmth/
    weather showed real lift against a popularity-matched baseline; the
    other five sit within +/-2 of chance and are never carried into
    `state.profile_terms` in the first place (state.py's
    derive_profile_terms() already filters to the three RARE_TAGS). This
    function therefore only needs to check literal presence of each
    surviving tag word in the text blob, exactly as §2.4 measured lift in
    the first place ("targets tagged comfort mentioning comfort X% of the
    time"); a session carrying more than one rare tag unions the check
    across all of them.

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts.
        state: Current session state.

    Returns:
        1.0 if any of `state.profile_terms` appears in the blob, else 0.0.
    """
    if not state.profile_terms:
        return 0.0
    record = facts.get(candidate.asin)
    blob = (record.get("blob") or "").lower() if record is not None else ""
    return 1.0 if any(tag in blob for tag in state.profile_terms) else 0.0


# The rating_style value the §2.4.1 interaction treats as "expects high
# ratings". Measured target means put "usually positive" (4.413) clearly
# apart from "mixed" (4.305) and "critical" (4.282), which sit close to
# each other — a binary split tracks the measured data better than a
# three-way linear skew would. Module-level so it is one line to revisit.
RATING_STYLE_EXPECTS_HIGH = "usually positive"


def rating_style_fit(candidate: Candidate, facts: dict[str, dict], state: SessionState) -> float:
    """average_rating x profile rating skew. §2.4.1, §3.4 Step 6.

    Design doc §2.4.1: "retained as an interaction feature (rating x
    profile_expects_high) rather than a standalone signal." Reads
    `state.rating_style` only — `average_prior_rating` is deliberately
    never read here, or anywhere in this module, since §2.4.1 states the
    two fields are the same signal and including both would introduce
    perfect collinearity on a 200-session fit.

    `state.rating_style` is not part of Qikun's original B1 session-state
    schema (§3.3 lists only slots/scenario_buffer/profile_terms/telemetry);
    it was added to SessionState as a minimal, read-only, profile-derived
    field — mirroring profile_terms — because this feature has no other
    way to reach it. See that field's docstring in state.py.

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts.
        state: Current session state.

    Returns:
        The candidate's rating/5 when the profile expects high ratings
        (RATING_STYLE_EXPECTS_HIGH), else 0.0. Never raises when
        rating_style is missing, None, or unrecognised.
    """
    record = facts.get(candidate.asin)
    if record is None or record.get("rating") is None:
        return 0.0
    rating_norm = record["rating"] / 5.0
    profile_expects_high = getattr(state, "rating_style", None) == RATING_STYLE_EXPECTS_HIGH
    return rating_norm if profile_expects_high else 0.0


def extract_features(
    candidate: Candidate, pool: list[Candidate], indexes: Indexes, state: SessionState
) -> dict[str, float]:
    """Compute all ten features for one candidate.

    Design doc §3.4 Step 6 feature table; §3.4 Step 7 ("the ten feature
    values per candidate" logged to telemetry).

    Args:
        candidate: The candidate to score.
        pool: The full candidate pool (for bm25_norm's max-normalisation).
        indexes: Offline indexes bundle, supplying `facts`.
        state: Current session state.

    Returns:
        A dict keyed by FEATURE_NAMES, in that order.
    """
    facts = indexes.facts
    return {
        "bm25_norm": bm25_norm(candidate, pool),
        "cos_sim": cos_sim(candidate),
        "pop": pop(candidate, facts),
        "rating": rating(candidate, facts),
        "price_fit": price_fit(candidate, facts, state),
        "category_match": category_match(candidate, facts, state),
        "brand_match": brand_match(candidate, facts, state),
        "slot_coverage": slot_coverage(candidate, facts, state),
        "rare_tag_match": rare_tag_match(candidate, facts, state),
        "rating_style_fit": rating_style_fit(candidate, facts, state),
    }


def feature_vector(
    candidate: Candidate, pool: list[Candidate], indexes: Indexes, state: SessionState
) -> tuple[float, ...]:
    """Compute all ten features for one candidate, as a fixed-order tuple.

    Design doc §3.4 Step 6; §8.3 step C2. Same inputs, same per-feature
    values as extract_features() — just shaped as a positional tuple in
    FEATURE_NAMES order rather than a dict, which is what rank.py's
    weighted sum and Owner D's telemetry corpus actually need positionally.

    Never raises for an asin absent from `indexes.facts`: every feature
    function above treats a missing facts record as "no information" and
    returns its own null-safe default rather than raising KeyError.

    Args:
        candidate: The candidate to score.
        pool: The full candidate pool (for bm25_norm's max-normalisation).
        indexes: Offline indexes bundle, supplying `facts`.
        state: Current session state.

    Returns:
        A tuple of ten floats, in FEATURE_NAMES order.
    """
    features = extract_features(candidate, pool, indexes, state)
    return tuple(features[name] for name in FEATURE_NAMES)
