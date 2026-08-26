"""Offline indexes: FTS5 keyword index, embedding matrix, facts dict, category lists.

Design doc §3.2 (Offline stage) and §4 (System diagram, "OFFLINE" block).

Owner: Haojun (Indexes and retrieval). §8.1, step A2 (stub retrieve — see
retrieval.py), step A3 (facts dict + per-category lists), step A4 (FTS5),
step A5 (embedding matrix), step A6 (brute-force kNN).

All four indexes are derived from utils.product_text() so they cannot
silently diverge (§3.2, §7.2). Everything below is a stub: no model is
downloaded, no SQLite table is populated, no matrix is built. Function
bodies return fixture values only.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np

from utils import fixture_score


@dataclass
class Indexes:
    """Bundle of the three offline views of the catalogue, plus raw rows.

    Design doc §3.2 and §4 (four boxes under "OFFLINE": FTS5, MiniLM matrix,
    facts dict, per-category lists).

    Attributes:
        catalog: The raw catalogue rows product_text() was derived from.
        fts_conn: SQLite connection holding the FTS5 virtual table
            (§3.2 Index 1).
        embedding_matrix: (n_products, 384) float32, L2-normalised
            (§3.2 Index 2).
        embedding_asins: Row-aligned ASINs for `embedding_matrix`.
        facts: Per-ASIN structured facts — dept, cat3, store, price,
            rating_number, pop, rating, text blob (§3.2 Index 3).
        category_lists: category path -> ASINs, pre-sorted by
            rating_number descending (§3.2 Index 3, §8.1 step A3).
    """

    catalog: list[dict]
    fts_conn: sqlite3.Connection
    embedding_matrix: np.ndarray
    embedding_asins: list[str]
    facts: dict[str, dict]
    category_lists: dict[str, list[str]]


def build_fts5_index(catalog: list[dict]) -> sqlite3.Connection:
    """Build the SQLite FTS5 keyword index over the catalogue.

    Design doc §3.2 Index 1: "SQLite FTS5, unicode61 tokenizer with
    diacritic removal ... Column weighting is already tuned in the
    baseline: title 6.0, categories 4.0, features 2.5, details 2.5, store
    1.5, description 1.0." Owner Haojun, §8.1 step A4.

    STUB: returns an empty in-memory SQLite connection with no virtual
    table populated. Real implementation creates an FTS5 virtual table over
    product_text()'s constituent columns with the weights above.

    Args:
        catalog: Raw catalogue rows.

    Returns:
        A SQLite connection (fixture: empty, in-memory).
    """
    return sqlite3.connect(":memory:")


def keyword_search(
    conn: sqlite3.Connection, query: str, limit: int
) -> list[tuple[str, float]]:
    """Run one FTS5 MATCH query and return sign-corrected scores.

    Design doc §3.2 Index 1: "Note that SQLite's bm25() returns negative
    scores, more negative being better; we normalise sign once at the
    retrieval boundary so all downstream code treats higher as better."
    Owner Haojun, §8.1 step A4: "Returns (asin, score) pairs."

    STUB: ignores `conn` and `query`; returns up to `limit` fixture ASINs
    from utils.FIXTURE_CATALOG with deterministic pseudo-scores.

    Args:
        conn: An FTS5-backed SQLite connection from build_fts5_index().
        query: The canonical intent string (or a slot-derived query).
        limit: Max rows to return (over-fetched 3x by the caller per
            §3.4 Step 4 before filtering).

    Returns:
        (asin, score) pairs, higher score is better, length <= limit.
    """
    from utils import FIXTURE_CATALOG

    rows = FIXTURE_CATALOG[:limit]
    return [(row["parent_asin"], fixture_score(row["parent_asin"] + query)) for row in rows]


def embed_text(text: str) -> np.ndarray:
    """Encode one string into the shared 384-d MiniLM space.

    Design doc §3.2 Index 2: "sentence-transformers/all-MiniLM-L6-v2, 384
    dimensions, used frozen." Also used for canonical-intent embedding
    (§3.4 Step 2), so query and products occupy one space. Owner Haojun, step A5.

    STUB: returns a deterministic 384-d vector derived from a hash of
    `text`, L2-normalised, without loading any model. Keeps the pipeline
    runnable without a network fetch or the sentence-transformers
    dependency installed.

    Args:
        text: Any string to embed (a product's product_text() or a
            canonical intent string).

    Returns:
        A (384,) float32 unit vector.
    """
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


def build_embedding_matrix(catalog: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Encode the whole catalogue into one L2-normalised matrix.

    Design doc §3.2 Index 2: "All 50,000 product texts are encoded once
    into a (50000, 384) float32 matrix, L2-normalised so cosine similarity
    reduces to a dot product ... persisted as .npy so subsequent runs load
    in under a second." Owner Haojun, §8.1 step A5.

    STUB: encodes utils.product_text() (itself a stub) for every row via
    embed_text(); no .npy persistence is implemented.

    Args:
        catalog: Raw catalogue rows.

    Returns:
        (matrix, asins) where matrix is (len(catalog), 384) float32 and
        asins[i] is the ASIN for matrix[i].
    """
    from utils import product_text

    asins = [row["parent_asin"] for row in catalog]
    matrix = np.stack([embed_text(product_text(row)) for row in catalog]).astype(np.float32)
    return matrix, asins


def knn_search(
    matrix: np.ndarray, asins: list[str], query_vec: np.ndarray, k: int
) -> list[tuple[str, float]]:
    """Brute-force exact top-k cosine search over the embedding matrix.

    Design doc §3.2 Index 2: "Search is brute-force exact kNN: scores =
    catalog_matrix @ q, then np.argpartition for top-k. Approximately
    5 ms." Owner Haojun, §8.1 step A6. See also §3.2 "Why not FAISS, HNSW or a
    vector database" — 50k rows makes exact search both compliant (no
    external vector DB clusters) and fast enough.

    STUB: performs the real matmul + argpartition over the (fixture)
    matrix passed in, so the *mechanism* is exercised even though the
    matrix itself comes from stub embeddings.

    Args:
        matrix: (n, 384) float32, L2-normalised.
        asins: Row-aligned ASINs for `matrix`.
        query_vec: (384,) float32, L2-normalised.
        k: Number of results to return.

    Returns:
        (asin, cosine_similarity) pairs, highest similarity first.
    """
    if matrix.shape[0] == 0:
        return []
    scores = matrix @ query_vec
    k = min(k, scores.shape[0])
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(asins[i], float(scores[i])) for i in top_idx]


def build_facts_dict(catalog: list[dict]) -> dict[str, dict]:
    """Build the per-ASIN structured facts dictionary.

    Design doc §3.2 Index 3: "One record per ASIN holding dept, cat3,
    store, price, rating_number, precomputed pop = log1p(rn)/log1p(100000),
    rating, and a lowercased text blob for substring matching ... Department
    is taken from categories[1] (100% coverage) rather than
    details.Department (87%)." Owner Haojun, §8.1 step A3.

    STUB: returns one fixture record per row in `catalog`, with `pop`
    computed for real (it is a one-line, non-controversial formula given in
    the doc) but `dept`/`cat3`/`blob` left as fixture placeholders.

    Args:
        catalog: Raw catalogue rows.

    Returns:
        asin -> {dept, cat3, store, price, rating_number, pop, rating, blob}
    """
    facts: dict[str, dict] = {}
    for row in catalog:
        asin = row["parent_asin"]
        rn = row.get("rating_number", 0) or 0
        facts[asin] = {
            "dept": "[STUB dept]",
            "cat3": "[STUB cat3]",
            "store": row.get("store"),
            "price": row.get("price"),
            "rating_number": rn,
            "pop": float(np.log1p(rn) / np.log1p(100_000)),
            "rating": row.get("average_rating"),
            "blob": "[STUB blob]",
        }
    return facts


def build_category_lists(catalog: list[dict], facts: dict[str, dict]) -> dict[str, list[str]]:
    """Group ASINs by category path, each list pre-sorted by review count.

    Design doc §3.2 Index 3: "A companion structure groups ASINs by
    category path with each list pre-sorted by review count, making
    'bestsellers in this category' a list slice rather than a scan."
    Haojun, §8.1 step A3.

    STUB: returns a single fixture bucket "[STUB category]" containing all
    ASINs in `catalog`, sorted by rating_number descending (the sort is
    real; the category grouping is not).

    Args:
        catalog: Raw catalogue rows.
        facts: Output of build_facts_dict(), for the sort key.

    Returns:
        category path -> ASINs, descending by rating_number.
    """
    asins = [row["parent_asin"] for row in catalog]
    asins.sort(key=lambda a: facts.get(a, {}).get("rating_number", 0), reverse=True)
    return {"[STUB category]": asins}


def build_indexes(catalog: list[dict] | None = None) -> Indexes:
    """Run the full offline stage and bundle the result.

    Design doc §3.2: "Runs once at startup, in roughly 5 seconds plus
    encoding time." §4's "OFFLINE" block. Composition point for A3-A6.

    Args:
        catalog: Raw catalogue rows, or None to load via
            utils.load_catalog() (itself a stub returning fixture rows).

    Returns:
        An Indexes bundle ready for retrieval.py to query.
    """
    from utils import load_catalog

    catalog = catalog if catalog is not None else load_catalog()
    fts_conn = build_fts5_index(catalog)
    matrix, asins = build_embedding_matrix(catalog)
    facts = build_facts_dict(catalog)
    category_lists = build_category_lists(catalog, facts)
    return Indexes(
        catalog=catalog,
        fts_conn=fts_conn,
        embedding_matrix=matrix,
        embedding_asins=asins,
        facts=facts,
        category_lists=category_lists,
    )
