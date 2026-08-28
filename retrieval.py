"""Three retrieval streams, union + dedupe, and the floor check.

Design doc §3.4 Step 4 (Multi-stream retrieval) and §4 (System diagram,
block ④ "THREE STREAMS").

Owner: Haojun (Indexes and retrieval). §8.1, step A2 (stub retrieve() — BLOCKING,
"C and D can develop against a stable signature within the first hour"),
step A7 (three streams with per-track quotas), step A8 (union, dedupe,
floor check), step A9 (browsing-track category diversity).

union_dedupe() was already real (§8.1: "simple enough not to warrant
deferring"). keyword_stream/semantic_stream/popularity_stream (A7),
floor_check (A8), and browse-track diversity (A9) are now real too — see
each function's docstring for what "real" means and where a design
judgment call had to be made in the absence of a fully-specified formula
(dept-match semantics for the keyword filter, the diversity cap's share,
what "relax the least-confident constraint" concretely does).
"""

from __future__ import annotations

from collections import Counter

from indexes import Indexes, keyword_search, knn_search
from state import SessionState
from utils import Candidate

# Per-track quotas (§3.4 Step 4 table).
STREAM_QUOTAS: dict[str, dict[str, int]] = {
    "buy": {"keyword": 120, "semantic": 40, "popularity": 20},
    "browse": {"keyword": 60, "semantic": 150, "popularity": 20},
}

# Keyword over-fetches this multiple before filtering (§3.4 Step 4
# technology note: "Keyword: FTS5 MATCH with over-fetch (3x limit) before
# filtering, so post-filter truncation cannot leave an undersized pool.").
KEYWORD_OVERFETCH = 3

# Semantic over-fetches by the same multiple on the browsing track only, so
# there is a pool to draw from after the diversity cap below excludes
# over-represented departments (§3.4 Step 4 technology note: "MMR or
# per-category caps applied on the browsing track for diversity" — no
# over-fetch is needed on the buying track, which applies no cap).
SEMANTIC_BROWSE_OVERFETCH = 3

# A9: "No single category exceeds a set share of the browsing pool."
# The doc leaves the share unset ("a set share") — 30% is this
# implementation's choice: loose enough that a genuinely dominant category
# (the shopper's actual intent) isn't starved, tight enough that three-plus
# departments must appear in any semantic-stream quota of gte ~4.
CATEGORY_DIVERSITY_MAX_SHARE = 0.3

# Floor check target: if the unioned pool falls below this, relax the
# least-confident constraint and re-retrieve (§3.4 Step 4).
POOL_FLOOR = 50


def _dept_of(asin: str, indexes: Indexes) -> str:
    """facts[asin]['dept'], or a stable placeholder when unknown/absent.

    Centralises the "unknown dept" bucket key so category-diversity and
    modal-category logic never let a None dept silently collapse into
    Python's hash-of-None grouping — every unknown-dept candidate is
    treated as sharing one visible bucket instead.
    """
    dept = indexes.facts.get(asin, {}).get("dept")
    return dept if dept else "[unknown department]"


def keyword_stream(state: SessionState, indexes: Indexes, quota: int) -> list[Candidate]:
    """Run the keyword (FTS5) stream, filtering only on the buying track.

    Design doc §3.4 Step 4: "Keyword (FTS5) ... 120/60 ... filters
    department + category, buying only." Owner Haojun, §8.1 step A7.

    On `state.track == "buy"`, over-fetches `quota * KEYWORD_OVERFETCH`
    hits, then filters to those whose facts entry matches the stated
    `department`/`category` slots before truncating to `quota` — matching
    the design doc's "over-fetch before filtering, so post-filter
    truncation cannot leave an undersized pool" (§3.4 Step 4 technology
    note). "Matches" is a design judgment call the doc doesn't fully spec:
    `department` is compared case-insensitively against `facts[asin]['dept']`
    (categories[1], §3.2 Index 3 — a controlled-ish vocabulary like "Men"/
    "Women"); `category` is far freer text from the user ("jackets", "a
    zip-front golf jacket"), so it is matched as a case-insensitive
    substring against `facts[asin]['blob']` (the lowercased product_text())
    rather than exact-matched against `cat3` — an exact match would reject
    almost everything, defeating the point of a *filter* that still needs
    the floor check's relaxation path to be the true safety net, not the
    common case. On `state.track == "browse"`, or when the slot in
    question is unset, no filter is applied on that slot (§3.4 Step 4:
    "never filters" for browse; an unset slot has nothing to filter by).

    Args:
        state: Current session state (canonical intent, slots, track).
        indexes: Offline indexes bundle.
        quota: Max candidates this stream should contribute.

    Returns:
        Candidates with `bm25_raw` set and "keyword" in `sources`.
    """
    hits = keyword_search(indexes.fts_conn, state.canonical_intent, quota * KEYWORD_OVERFETCH)

    if state.track == "buy":
        department = state.slots.get("department")
        category = state.slots.get("category")
        filtered = []
        for asin, score in hits:
            facts = indexes.facts.get(asin, {})
            if department and (facts.get("dept") or "").casefold() != department.casefold():
                continue
            if category and category.casefold() not in (facts.get("blob") or ""):
                continue
            filtered.append((asin, score))
        hits = filtered

    return [Candidate(asin=asin, bm25_raw=score, sources={"keyword"}) for asin, score in hits[:quota]]


def semantic_stream(state: SessionState, indexes: Indexes, quota: int) -> list[Candidate]:
    """Run the semantic (kNN) stream. Never filters, on either track.

    Design doc §3.4 Step 4: "Semantic (kNN) ... 40/150 ... never filters
    ... MMR or per-category caps applied on the browsing track for
    diversity" (A9). "Never filters" and "applies a diversity cap" sound
    contradictory but are not: a filter permanently removes a candidate a
    hard slot ruled out; the cap only reorders/limits *representation*
    within this stream's own contribution — every excluded-by-cap
    candidate had a real cosine score and none is excluded for
    query-relevance reasons, so recall is not what the cap trades away
    (over-representation of one department at the other departments'
    expense within *this stream's* quota is what it trades away).

    On `state.track == "browse"`, over-fetches `quota *
    SEMANTIC_BROWSE_OVERFETCH` hits and greedily fills `quota` slots
    highest-cosine-first, skipping a hit once its department has already
    reached `floor(quota * CATEGORY_DIVERSITY_MAX_SHARE)` selections from
    this stream — so a hit skipped for its own department can still be
    followed by a lower-scoring hit from an under-represented department.
    Falls back to filling any remaining slots uncapped (highest score
    first among what's left) if the cap leaves the quota unfilled, so
    diversity never costs recall for a genuinely narrow catalogue segment.
    On `state.track == "buy"`, no cap or over-fetch — behavior matches the
    original stub exactly (top `quota` hits, unmodified).

    Args:
        state: Current session state; uses `canonical_vector`.
        indexes: Offline indexes bundle.
        quota: Max candidates this stream should contribute.

    Returns:
        Candidates with `cos_raw` set and "semantic" in `sources`.
    """
    if state.canonical_vector is None:
        return []

    if state.track != "browse":
        hits = knn_search(indexes.embedding_matrix, indexes.embedding_asins, state.canonical_vector, quota)
        return [Candidate(asin=asin, cos_raw=score, sources={"semantic"}) for asin, score in hits]

    hits = knn_search(
        indexes.embedding_matrix,
        indexes.embedding_asins,
        state.canonical_vector,
        quota * SEMANTIC_BROWSE_OVERFETCH,
    )
    cap = max(1, int(quota * CATEGORY_DIVERSITY_MAX_SHARE))
    dept_counts: Counter[str] = Counter()
    selected: list[tuple[str, float]] = []
    leftover: list[tuple[str, float]] = []
    for asin, score in hits:
        if len(selected) >= quota:
            break
        dept = _dept_of(asin, indexes)
        if dept_counts[dept] < cap:
            dept_counts[dept] += 1
            selected.append((asin, score))
        else:
            leftover.append((asin, score))
    if len(selected) < quota:
        selected.extend(leftover[: quota - len(selected)])
    return [Candidate(asin=asin, cos_raw=score, sources={"semantic"}) for asin, score in selected]


def popularity_stream(pool: list[Candidate], indexes: Indexes, quota: int) -> list[Candidate]:
    """Run the popularity stream, ignoring the query entirely.

    Design doc §3.4 Step 4: "Popularity ... 20/20 ... ignores query
    entirely ... reads pool's modal categories, slice the pre-sorted
    per-category lists." Owner Haojun, §8.1 step A7.

    Finds `pool`'s modal department (via `_dept_of()`, i.e. `facts[asin]
    ['dept']`, ties broken by department name so results are deterministic)
    and slices `indexes.category_lists[modal_dept]` — already pre-sorted by
    rating_number descending (§8.1 step A3) — for up to `quota` ASINs not
    already in `pool`. If that slice runs short (a thin department, or an
    empty `pool` with no modal department to find), tops up from the
    remaining departments' lists, largest first, so popularity_stream()
    itself never returns short of `quota` purely for lack of a query-
    derived category to slice (§3.4 Step 4: "ignores query entirely" — a
    thin modal department is not query-relevance information worth
    respecting here).

    Args:
        pool: Candidates already gathered by the other two streams, used
            to find the modal department.
        indexes: Offline indexes bundle.
        quota: Max candidates this stream should contribute.

    Returns:
        Candidates with no bm25/cos score and "popularity" in `sources`.
    """
    already_present = {c.asin for c in pool}
    dept_counts = Counter(_dept_of(c.asin, indexes) for c in pool)
    ranked_depts = [dept for dept, _ in dept_counts.most_common()]
    # Departments absent from the pool entirely (e.g. an empty pool) are
    # still eligible top-up sources, largest list first.
    ranked_depts += sorted(
        (d for d in indexes.category_lists if d not in dept_counts),
        key=lambda d: len(indexes.category_lists[d]),
        reverse=True,
    )

    asins: list[str] = []
    for dept in ranked_depts:
        if len(asins) >= quota:
            break
        for asin in indexes.category_lists.get(dept, []):
            if len(asins) >= quota:
                break
            if asin in already_present or asin in asins:
                continue
            asins.append(asin)
    return [Candidate(asin=asin, sources={"popularity"}) for asin in asins]


def union_dedupe(streams: list[list[Candidate]]) -> list[Candidate]:
    """Union candidates from all streams, merging scores for duplicates.

    Design doc §3.4 Step 4: "Three independent streams, unioned and
    deduplicated into one candidate pool ... Union is monotonic — it can
    only add candidates." Owner Haojun, §8.1 step A8.

    STUB: performs the real merge-by-asin (this is simple enough not to
    warrant deferring): first occurrence of a duplicate keeps growing its
    `sources` set and score fields from later occurrences.

    Args:
        streams: One list of Candidates per stream.

    Returns:
        Deduplicated candidates, one per ASIN, insertion order preserved.
    """
    merged: dict[str, Candidate] = {}
    for stream in streams:
        for cand in stream:
            if cand.asin not in merged:
                merged[cand.asin] = cand
            else:
                existing = merged[cand.asin]
                existing.bm25_raw = existing.bm25_raw or cand.bm25_raw
                existing.cos_raw = existing.cos_raw or cand.cos_raw
                existing.sources |= cand.sources
    return list(merged.values())


def floor_check(pool: list[Candidate], state: SessionState, indexes: Indexes) -> list[Candidate]:
    """If the pool is undersized, relax the least-confident constraint and retry.

    Design doc §3.4 Step 4: "Floor check after all filtering — if the pool
    is undersized, relax the least-confident constraint and re-retrieve."
    Owner Haojun, §8.1 step A8: "Pool is never empty."

    The "least-confident constraint" available to relax at this point,
    without re-deriving state or re-running the keyword/semantic streams,
    is popularity_stream()'s own department targeting: it already ignores
    the query and never filters, so re-invoking it for `POOL_FLOOR -
    len(pool)` more candidates (it excludes anything already in `pool`,
    §8.1 step A7) tops the pool up from whichever departments have supply
    left — the correctness-preserving move, since popularity contributing
    more never risks removing a candidate a hard slot legitimately ruled
    out (§3.4 Step 4's "why union" rationale: "an extra stream carries no
    recall risk"). If even that leaves the pool empty (no catalogue data
    at all — the last-resort case, not a real deployment path), a single
    placeholder candidate still guarantees the "never empty" contract so
    every downstream consumer (ranking, clarification) has at least one
    row to operate on.

    Args:
        pool: The unioned, deduplicated candidate pool.
        state: Current session state (unused directly here; the relax
            path works through popularity_stream() rather than re-deriving
            slot state, but is accepted to keep this function's signature
            stable for existing callers — §8.1 step A8).
        indexes: Offline indexes bundle, re-queried on relax.

    Returns:
        A pool guaranteed non-empty.
    """
    if len(pool) >= POOL_FLOOR:
        return pool

    topped_up = popularity_stream(pool, indexes, POOL_FLOOR - len(pool))
    pool = union_dedupe([pool, topped_up])

    if pool:
        return pool
    return [Candidate(asin="[STUB floor-check fallback]", sources={"floor_check"})]


def retrieve(state: SessionState, track: str, indexes: Indexes) -> list[Candidate]:
    """Run all three streams for this turn and return the unioned pool.

    Design doc §7.2 interface contract: `retrieve(state, track) ->
    [candidate]`. Owner Haojun, §8.1 step A2 (BLOCKING stub) through A8.

    Args:
        state: Current session state.
        track: "buy" or "browse", from state.pick_track().
        indexes: Offline indexes bundle.

    Returns:
        The floor-checked, deduplicated candidate pool for this turn.
    """
    quotas = STREAM_QUOTAS.get(track, STREAM_QUOTAS["browse"])
    keyword = keyword_stream(state, indexes, quotas["keyword"])
    semantic = semantic_stream(state, indexes, quotas["semantic"])
    popularity = popularity_stream(keyword + semantic, indexes, quotas["popularity"])
    pool = union_dedupe([keyword, semantic, popularity])
    return floor_check(pool, state, indexes)
