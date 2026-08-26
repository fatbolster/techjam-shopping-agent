"""Three retrieval streams, union + dedupe, and the floor check.

Design doc §3.4 Step 4 (Multi-stream retrieval) and §4 (System diagram,
block ④ "THREE STREAMS").

Owner: Haojun (Indexes and retrieval). §8.1, step A2 (stub retrieve() — BLOCKING,
"C and D can develop against a stable signature within the first hour"),
step A7 (three streams with per-track quotas), step A8 (union, dedupe,
floor check), step A9 (browsing-track category diversity).

Everything below is a stub. Function bodies return fixture values only.
"""

from __future__ import annotations

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

# Floor check target: if the unioned pool falls below this, relax the
# least-confident constraint and re-retrieve (§3.4 Step 4).
POOL_FLOOR = 50


def keyword_stream(state: SessionState, indexes: Indexes, quota: int) -> list[Candidate]:
    """Run the keyword (FTS5) stream, filtering only on the buying track.

    Design doc §3.4 Step 4: "Keyword (FTS5) ... 120/60 ... filters
    department + category, buying only." Owner Haojun, §8.1 step A7.

    STUB: calls the (stub) keyword_search() with a 3x over-fetch and
    truncates to `quota`; no department/category filter is actually
    applied regardless of `state.track`.

    Args:
        state: Current session state (canonical intent, slots, track).
        indexes: Offline indexes bundle.
        quota: Max candidates this stream should contribute.

    Returns:
        Candidates with `bm25_raw` set and "keyword" in `sources`.
    """
    hits = keyword_search(indexes.fts_conn, state.canonical_intent, quota * KEYWORD_OVERFETCH)
    return [Candidate(asin=asin, bm25_raw=score, sources={"keyword"}) for asin, score in hits[:quota]]


def semantic_stream(state: SessionState, indexes: Indexes, quota: int) -> list[Candidate]:
    """Run the semantic (kNN) stream. Never filters, on either track.

    Design doc §3.4 Step 4: "Semantic (kNN) ... 40/150 ... never filters."
    Owner Haojun, §8.1 step A7; diversity (MMR / per-category caps) on the
    browsing track is step A9, not implemented here.

    STUB: calls the (stub) knn_search() over the (stub) embedding matrix.

    Args:
        state: Current session state; uses `canonical_vector`.
        indexes: Offline indexes bundle.
        quota: Max candidates this stream should contribute.

    Returns:
        Candidates with `cos_raw` set and "semantic" in `sources`.
    """
    if state.canonical_vector is None:
        return []
    hits = knn_search(indexes.embedding_matrix, indexes.embedding_asins, state.canonical_vector, quota)
    return [Candidate(asin=asin, cos_raw=score, sources={"semantic"}) for asin, score in hits]


def popularity_stream(pool: list[Candidate], indexes: Indexes, quota: int) -> list[Candidate]:
    """Run the popularity stream, ignoring the query entirely.

    Design doc §3.4 Step 4: "Popularity ... 20/20 ... ignores query
    entirely ... reads pool's modal categories, slice the pre-sorted
    per-category lists." Owner Haojun, §8.1 step A7.

    STUB: ignores `pool`'s modal categories and simply slices the single
    fixture bucket in indexes.category_lists.

    Args:
        pool: Candidates already gathered by the other two streams, used
            (in the real implementation) to find modal categories.
        indexes: Offline indexes bundle.
        quota: Max candidates this stream should contribute.

    Returns:
        Candidates with no bm25/cos score and "popularity" in `sources`.
    """
    asins: list[str] = []
    for bucket in indexes.category_lists.values():
        asins.extend(bucket)
    return [Candidate(asin=asin, sources={"popularity"}) for asin in asins[:quota]]


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

    STUB: returns `pool` unchanged; no constraint relaxation or re-retrieve
    loop is implemented. Included so callers already depend on the final
    signature.

    Args:
        pool: The unioned, deduplicated candidate pool.
        state: Current session state (would identify which slot to relax).
        indexes: Offline indexes bundle (would be re-queried on relax).

    Returns:
        A pool guaranteed non-empty (fixture: whatever was passed in, or
        one fixture candidate if `pool` was empty).
    """
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
