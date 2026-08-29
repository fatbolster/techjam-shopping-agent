"""Offline indexes: FTS5 keyword index, embedding matrix, facts dict, category lists.

Design doc §3.2 (Offline stage) and §4 (System diagram, "OFFLINE" block).

Owner: Haojun (Indexes and retrieval). §8.1, step A3 (facts dict +
per-category lists), step A4 (FTS5), step A5 (embedding matrix), step A6
(brute-force kNN).

All four indexes are derived from utils.product_text() so they cannot
silently diverge (§3.2, §7.2). A3-A6 are real; knn_search() (A6) was
already real even before A3-A5 landed (matmul + argpartition needs no
catalogue-specific logic, only real inputs to run over).
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


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


# Column weights for bm25(), in FTS5_COLUMNS order after the leading
# UNINDEXED asin column (§3.2 Index 1: "title 6.0, categories 4.0,
# features 2.5, details 2.5, store 1.5, description 1.0"). asin's own
# weight (0.0) is never scored — UNINDEXED columns take no part in MATCH —
# but bm25() still expects one positional argument per table column.
FTS5_COLUMNS: list[str] = ["title", "categories", "features", "details", "store", "description"]
FTS5_WEIGHTS: list[float] = [6.0, 4.0, 2.5, 2.5, 1.5, 1.0]

_FTS5_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def build_fts5_index(catalog: list[dict]) -> sqlite3.Connection:
    """Build the SQLite FTS5 keyword index over the catalogue.

    Design doc §3.2 Index 1: "SQLite FTS5, unicode61 tokenizer with
    diacritic removal ... Column weighting is already tuned in the
    baseline: title 6.0, categories 4.0, features 2.5, details 2.5, store
    1.5, description 1.0." Owner Haojun, §8.1 step A4.

    Six columns hold the raw fields directly (not product_text()'s single
    blob), so each can carry its own bm25() weight via FTS5_WEIGHTS —
    product_text() feeds the *embedding* matrix and the facts blob, not
    this table (see utils.product_text()'s docstring). `asin` is stored
    UNINDEXED (present for retrieval, excluded from MATCH/scoring).

    Args:
        catalog: Raw catalogue rows.

    Returns:
        An in-memory SQLite connection with the populated `products` FTS5
        virtual table.
    """
    conn = sqlite3.connect(":memory:")
    columns_sql = ", ".join(FTS5_COLUMNS)
    conn.execute(
        f"CREATE VIRTUAL TABLE products USING fts5("
        f"asin UNINDEXED, {columns_sql}, "
        f"tokenize='unicode61 remove_diacritics 2')"
    )
    rows = []
    for row in catalog:
        details = row.get("details") or {}
        rows.append(
            (
                row.get("parent_asin", ""),
                str(row.get("title") or ""),
                " ".join(str(c) for c in (row.get("categories") or [])),
                " ".join(str(f) for f in (row.get("features") or [])),
                " ".join(str(v) for v in details.values()),
                str(row.get("store") or ""),
                " ".join(str(d) for d in (row.get("description") or [])),
            )
        )
    placeholders = ", ".join(["?"] * (1 + len(FTS5_COLUMNS)))
    conn.executemany(f"INSERT INTO products VALUES ({placeholders})", rows)
    conn.commit()
    return conn


def _fts5_match_query(query: str) -> Optional[str]:
    """Turn a free-text query into a forgiving FTS5 MATCH expression.

    OR-joins double-quoted tokens (rather than FTS5's implicit AND between
    bareword terms) so a query with one uncatalogued word still matches
    products sharing any of the others — the keyword stream is meant to
    over-fetch and get filtered/truncated downstream, not to be a strict
    boolean filter itself. Tokens are alphanumeric only (`\\w` minus `_`,
    via _FTS5_TOKEN_RE) and individually double-quoted so punctuation in
    `query` (":", "$", multi-value list separators) can't break FTS5's
    MATCH syntax or be misread as a column filter / operator.

    Args:
        query: Free text, e.g. a rendered canonical_intent string.

    Returns:
        An FTS5 MATCH expression, or None if `query` has no word tokens.
    """
    tokens = _FTS5_TOKEN_RE.findall(query or "")
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def keyword_search(
    conn: sqlite3.Connection, query: str, limit: int
) -> list[tuple[str, float]]:
    """Run one FTS5 MATCH query and return sign-corrected scores.

    Design doc §3.2 Index 1: "Note that SQLite's bm25() returns negative
    scores, more negative being better; we normalise sign once at the
    retrieval boundary so all downstream code treats higher as better."
    Owner Haojun, §8.1 step A4: "Returns (asin, score) pairs."

    Args:
        conn: An FTS5-backed SQLite connection from build_fts5_index().
        query: The canonical intent string (or a slot-derived query).
        limit: Max rows to return (over-fetched 3x by the caller per
            §3.4 Step 4 before filtering).

    Returns:
        (asin, score) pairs, higher score is better, length <= limit.
        Empty if `query` has no matchable tokens or nothing matches.
    """
    match_expr = _fts5_match_query(query)
    if match_expr is None or limit <= 0:
        return []
    weights_sql = ", ".join(str(w) for w in FTS5_WEIGHTS)
    cursor = conn.execute(
        f"SELECT asin, bm25(products, 0.0, {weights_sql}) AS score "
        f"FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
        (match_expr, limit),
    )
    # bm25(): more negative is better; sign-correct once at this boundary.
    return [(asin, -score) for asin, score in cursor.fetchall()]


# sentence-transformers/torch are imported lazily (inside _get_model(), not
# at module level) so indexes.py stays importable without them — same
# reasoning as rank.py's lazy sklearn import (§9: "Pipeline is fully
# functional with zero LLM calls" extends to "no model download required
# just to import a module").
_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
_model_singleton: list = []  # 0 or 1 elements; avoids a `global` rebind


def _get_embedding_model():
    """Lazily load and cache the one shared SentenceTransformer instance.

    A module-level singleton (rather than reloading per call) since
    encode() is what's expensive; embed_text() is called once per turn
    (canonical-intent embedding) and must stay fast after the first call.
    """
    if not _model_singleton:
        from sentence_transformers import SentenceTransformer

        _model_singleton.append(SentenceTransformer(_EMBEDDING_MODEL_NAME))
    return _model_singleton[0]


def embed_text(text: str) -> np.ndarray:
    """Encode one string into the shared 384-d MiniLM space.

    Design doc §3.2 Index 2: "sentence-transformers/all-MiniLM-L6-v2, 384
    dimensions, used frozen." Also used for canonical-intent embedding
    (§3.4 Step 2), so query and products occupy one space. Owner Haojun, step A5.

    Args:
        text: Any string to embed (a product's product_text() or a
            canonical intent string). An empty string still returns a
            valid (if not meaningful) unit vector — MiniLM has no
            empty-input special case.

    Returns:
        A (384,) float32 unit vector, L2-normalised.
    """
    model = _get_embedding_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)


def build_embedding_matrix(
    catalog: list[dict], cache_path: Optional[str] = "data/embeddings.npy"
) -> tuple[np.ndarray, list[str]]:
    """Encode the whole catalogue into one L2-normalised matrix.

    Design doc §3.2 Index 2: "All 50,000 product texts are encoded once
    into a (50000, 384) float32 matrix, L2-normalised so cosine similarity
    reduces to a dot product ... persisted as .npy so subsequent runs load
    in under a second." Owner Haojun, §8.1 step A5.

    Batch-encodes utils.product_text() for every row in one model.encode()
    call (far faster than one embed_text() call per row). When `cache_path`
    is given and both it and its sibling `<cache_path>.asins.json` exist
    with an ASIN list matching `catalog` exactly (order and content — the
    cheap correctness check the design doc's "loads in under a second"
    claim implicitly assumes holds), the cached matrix is loaded instead of
    re-encoding. Both paths are gitignored (`*.npy`, `data/`), matching the
    embedding matrix's "large derived artifact, never committed" status.

    Args:
        catalog: Raw catalogue rows.
        cache_path: Where to persist/load the matrix, or None to always
            encode fresh and skip persistence (e.g. in tests).

    Returns:
        (matrix, asins) where matrix is (len(catalog), 384) float32 and
        asins[i] is the ASIN for matrix[i].
    """
    from utils import product_text

    asins = [row["parent_asin"] for row in catalog]

    if cache_path is not None:
        asins_path = Path(f"{cache_path}.asins.json")
        matrix_path = Path(cache_path)
        if matrix_path.exists() and asins_path.exists():
            with open(asins_path, encoding="utf-8") as f:
                cached_asins = json.load(f)
            if cached_asins == asins:
                return np.load(matrix_path), asins

    if not catalog:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), asins

    model = _get_embedding_model()
    texts = [product_text(row) for row in catalog]
    matrix = np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32
    )

    if cache_path is not None:
        matrix_path = Path(cache_path)
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(matrix_path, matrix)
        with open(f"{cache_path}.asins.json", "w", encoding="utf-8") as f:
            json.dump(asins, f)

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

    `cat3` is the next category level down, `categories[2]` — the third
    entry overall (`categories[0]` is the top-level "Clothing, Shoes &
    Jewelry"-style root, `categories[1]` is dept). `blob` is
    `utils.product_text(row)` lowercased, reusing the one shared text view
    rather than re-deriving a second join (§3.2/§7.2: "so the indexes
    cannot silently diverge").

    Also carries `brand`/`color`/`material`/`style`/`size` — one field per
    `utils.PRODUCT_TEXT_DETAIL_KEYS` entry (`row["details"]`, or None when
    that row's `details` doesn't have the key), plus `size` (not one of
    product_text()'s five, since product_text() feeds embeddings/FTS5
    where a size like "M" or "10.5" is low-value token noise, but useful
    here structured). This is a superset of §3.2 Index 3's literal field
    list — not spec, but what clarify.py's score_attribute() needs for a
    real per-candidate value distribution (Marcus, §8.5 step E4) without
    reaching back into raw catalogue rows for a second, divergence-prone
    join.

    Args:
        catalog: Raw catalogue rows.

    Returns:
        asin -> {dept, cat3, store, price, rating_number, pop, rating, blob,
        brand, color, material, style, size}. `dept`/`cat3` are None when
        `categories` is too short to hold them; the five detail fields are
        None when absent from that row's `details`.
    """
    from utils import product_text

    facts: dict[str, dict] = {}
    for row in catalog:
        asin = row["parent_asin"]
        rn = row.get("rating_number", 0) or 0
        categories = row.get("categories") or []
        details = row.get("details") or {}
        facts[asin] = {
            "dept": categories[1] if len(categories) > 1 else None,
            "cat3": categories[2] if len(categories) > 2 else None,
            # The full category path, lowercased, for token-level category
            # matching. dept/cat3 alone stop at categories[2] ("Clothing",
            # "Shoes") — but the category a shopper states ("women
            # dresses") names the *deeper* levels, which previously
            # existed nowhere in facts except buried inside `blob` where
            # matching also hits descriptions (features.category_match()'s
            # near-zero fitted weight was measured to come exactly from
            # this gap: only 4 of 629 target rows matched their own
            # stated category).
            "cat_path": " ".join(str(c) for c in categories).lower(),
            "store": row.get("store"),
            "price": row.get("price"),
            "rating_number": rn,
            "brand": details.get("Brand"),
            "color": details.get("Color"),
            "material": details.get("Material"),
            "style": details.get("Style"),
            "size": details.get("Size"),
            "pop": float(np.log1p(rn) / np.log1p(100_000)),
            "rating": row.get("average_rating"),
            "blob": product_text(row).lower(),
        }
    return facts


# Bucket key for candidates whose category is unknown (categories too short
# to hold a dept) — keeps every ASIN groupable without dropping any from
# the popularity stream's source lists.
UNKNOWN_CATEGORY = "[unknown department]"


def build_category_lists(catalog: list[dict], facts: dict[str, dict]) -> dict[str, list[str]]:
    """Group ASINs by department, each list pre-sorted by review count.

    Design doc §3.2 Index 3: "A companion structure groups ASINs by
    category path with each list pre-sorted by review count, making
    'bestsellers in this category' a list slice rather than a scan."
    Haojun, §8.1 step A3.

    Grouped by `dept` (`facts[asin]["dept"]`, i.e. `categories[1]`) rather
    than the full category path: dept is the one level with 100% coverage
    (§3.2's justification for using it over `details.Department` in the
    first place carries over here), and it is what popularity_stream()
    reads via "the pool's modal categories" (§3.4 Step 4) — a finer path
    would fragment lists past the point a 20-candidate popularity quota
    could usefully slice.

    Args:
        catalog: Raw catalogue rows.
        facts: Output of build_facts_dict(), for dept/rating_number.

    Returns:
        dept -> ASINs (or UNKNOWN_CATEGORY for rows with no dept),
        each list descending by rating_number.
    """
    buckets: dict[str, list[str]] = {}
    for row in catalog:
        asin = row["parent_asin"]
        dept = facts.get(asin, {}).get("dept") or UNKNOWN_CATEGORY
        buckets.setdefault(dept, []).append(asin)
    for asins in buckets.values():
        asins.sort(key=lambda a: facts.get(a, {}).get("rating_number", 0), reverse=True)
    return buckets


def build_indexes(
    catalog: list[dict] | None = None, embedding_cache_path: Optional[str] = "data/embeddings.npy"
) -> Indexes:
    """Run the full offline stage and bundle the result.

    Design doc §3.2: "Runs once at startup, in roughly 5 seconds plus
    encoding time." §4's "OFFLINE" block. Composition point for A3-A6.

    Args:
        catalog: Raw catalogue rows, or None to load via
            utils.load_catalog() (falls back to the 3-row FIXTURE_CATALOG
            when data/catalog.jsonl is absent).
        embedding_cache_path: Passed through to build_embedding_matrix().
            Callers passing a small/fixture `catalog` (tests, Agent's
            fixture path) should pass None here — otherwise a fixture-sized
            matrix can overwrite the real cached embeddings.npy on disk the
            next time the catalog's ASIN set doesn't match (see
            build_embedding_matrix()'s cache-invalidation check).

    Returns:
        An Indexes bundle ready for retrieval.py to query.
    """
    from utils import load_catalog

    catalog = catalog if catalog is not None else load_catalog()
    fts_conn = build_fts5_index(catalog)
    matrix, asins = build_embedding_matrix(catalog, cache_path=embedding_cache_path)
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
