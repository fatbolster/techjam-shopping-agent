"""The ten ranking features from §3.4 Step 6.

Design doc §3.4 Step 6 (Ranking) feature table.

Owner: Emerson (Ranking). §8.3, step C2 ("Implement the ten feature functions.
Each independently unit-testable. price_fit returns a neutral value for
nulls — never excludes."), step C4 ("Handle missing stream scores. Default
to zero.").

Everything below is a stub. Function bodies return fixture values only,
via utils.fixture_score(), except price_fit's null-safety (§2.2) and
rating's /5 normalisation, which are simple enough to implement for real
without pre-empting any design decision.
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

    STUB: returns a fixture pseudo-score keyed by the candidate's ASIN,
    not the real `candidate.bm25_raw / max(bm25_raw over pool)`.

    Args:
        candidate: The candidate to score.
        pool: The full candidate pool, for max-normalisation.

    Returns:
        A float, intended range [0, 1].
    """
    return fixture_score("bm25_norm:" + candidate.asin)


def cos_sim(candidate: Candidate) -> float:
    """Dot product with the query vector. §3.4 Step 6: semantic relevance.

    STUB: returns a fixture pseudo-score rather than `candidate.cos_raw`.

    Args:
        candidate: The candidate to score.

    Returns:
        A float, intended range [-1, 1].
    """
    return fixture_score("cos_sim:" + candidate.asin)


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


def price_fit(candidate: Candidate, facts: dict[str, dict], state: SessionState) -> float:
    """0.5 when price is null, else fit to stated budget. §2.2, §3.4 Step 6.

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
        0.5 if price is null; otherwise a fixture pseudo-score in [0, 1].
    """
    record = facts.get(candidate.asin)
    if record is None or record.get("price") is None:
        return 0.5
    return fixture_score("price_fit:" + candidate.asin)


def category_match(candidate: Candidate, facts: dict[str, dict], state: SessionState) -> float:
    """Whether the candidate matches the slot category. §3.4 Step 6: constraint satisfaction.

    STUB: returns a fixture pseudo-score rather than checking
    `state.slots.get("category")` against `facts[asin]["cat3"]`.

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts.
        state: Current session state.

    Returns:
        A float, intended range {0, 1} (or a partial-match score).
    """
    return fixture_score("category_match:" + candidate.asin)


def brand_match(candidate: Candidate, facts: dict[str, dict], state: SessionState) -> float:
    """Whether `store` matches the slot brand. §3.4 Step 6: constraint satisfaction.

    STUB: returns a fixture pseudo-score rather than checking
    `state.slots.get("brand")` against `facts[asin]["store"]`.

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts.
        state: Current session state.

    Returns:
        A float, intended range {0, 1}.
    """
    return fixture_score("brand_match:" + candidate.asin)


def slot_coverage(candidate: Candidate, facts: dict[str, dict], state: SessionState) -> float:
    """Fraction of slot terms present in the text blob. §2.3, §3.4 Step 6.

    Design doc §2.3: "Attribute matching must therefore operate over text,
    not structured lookup — which makes slot matching a scoring signal
    rather than a filter."

    STUB: returns a fixture pseudo-score rather than counting
    `state.slots.values()` substrings found in `facts[asin]["blob"]`.

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts.
        state: Current session state.

    Returns:
        A float, intended range [0, 1].
    """
    return fixture_score("slot_coverage:" + candidate.asin)


def rare_tag_match(candidate: Candidate, facts: dict[str, dict], state: SessionState) -> float:
    """1 if any rare-tag term is present, else 0. §2.4, §3.4 Step 6.

    Design doc §2.4: three rare tags (performance, warmth, weather) show
    real lift; carried in `state.profile_terms`, read-only (§3.3 invariant).

    STUB: returns a fixture pseudo-score rather than checking
    `state.profile_terms` against `facts[asin]["blob"]`.

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts.
        state: Current session state.

    Returns:
        A float, intended range {0, 1}.
    """
    return fixture_score("rare_tag_match:" + candidate.asin)


def rating_style_fit(candidate: Candidate, facts: dict[str, dict], state: SessionState) -> float:
    """average_rating x profile rating skew. §2.4.1, §3.4 Step 6.

    Design doc §2.4.1: "It is retained as an interaction feature (rating x
    profile_expects_high) rather than a standalone signal ... measured
    at 0.131 stars, p=0.003, across 100% of sessions."

    STUB: returns a fixture pseudo-score rather than the real interaction
    of `facts[asin]["rating"]` with the session's rating_style.

    Args:
        candidate: The candidate to score.
        facts: Per-ASIN structured facts.
        state: Current session state.

    Returns:
        A float, intended range [0, 1].
    """
    return fixture_score("rating_style_fit:" + candidate.asin)


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
