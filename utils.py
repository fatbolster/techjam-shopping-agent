"""Shared utilities: `product_text()`, catalogue loading, the `Candidate` shape.

Design doc §3.2 ("Produces three views of one catalogue, all derived from a
single shared product_text() function so that the indexes cannot silently
diverge") and §7.2 ("The single highest-risk shared object").

Owner: Haojun (Indexes and retrieval). §8.1, step A1 — BLOCKING.
This module has no internal dependencies; every other module may import it.

Everything below is a stub. Function bodies return fixture values only —
see the module docstring in each file for what "real" means here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Shared candidate shape (§7.2 C1: "Define the candidate object shape").
# Placed here rather than in features.py/rank.py because utils.py is the one
# module every retrieval- and ranking-adjacent module already imports.
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """One pooled candidate product, carrying each stream's raw contribution.

    Design doc §7.2 (Owner Emerson, step C1) and §3.4 Step 4 / Step 6.

    Attributes:
        asin: `parent_asin` of the candidate product.
        bm25_raw: Sign-corrected FTS5 score (higher is better), 0.0 if this
            candidate was not surfaced by the keyword stream (§3.4 Step 6,
            Owner Emerson step C4: "Handle missing stream scores. Default to zero").
        cos_raw: Cosine similarity against the canonical query vector, 0.0 if
            not surfaced by the semantic stream.
        sources: Which stream(s) contributed this candidate, e.g.
            {"keyword"}, {"semantic", "popularity"}. Diagnostic only, feeds
            §6.2's per-stream recall reporting (Owner Marcus, step D7).
    """

    asin: str
    bm25_raw: float = 0.0
    cos_raw: float = 0.0
    sources: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Fixture catalogue: three rows, shaped like the Amazon Reviews 2023 dataset
# (§1, "Dataset"), used so every module can be exercised without the real
# 50,000-row catalog.jsonl on disk.
# ---------------------------------------------------------------------------
FIXTURE_CATALOG: list[dict] = [
    {
        "parent_asin": "B00FIXTURE1",
        "title": "London Fog Men's Auburn Zip-Front Golf Jacket",
        "features": ["Full zip front", "Water resistant shell", "Machine washable"],
        "description": ["A lightweight golf jacket built for cool-weather rounds."],
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Jackets & Coats"],
        "store": "London Fog",
        "details": {
            "Department": "Mens",
            "Material": "Polyester",
            "Color": "Auburn",
            "Brand": "London Fog",
            "Style": "Golf Jacket",
        },
        "price": 64.99,
        "rating_number": 8421,
        "average_rating": 4.5,
    },
    {
        "parent_asin": "B00FIXTURE2",
        "title": "Saucony Cohesion 10 Running Shoe",
        "features": ["Breathable mesh upper", "EVA midsole", "Rubber outsole"],
        "description": ["A dependable everyday trainer for road running."],
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Running"],
        "store": "Saucony",
        "details": {
            "Department": "Mens",
            "Material": "Mesh",
            "Color": "Black",
            "Brand": "Saucony",
            "Style": "Running Shoe",
        },
        "price": None,
        "rating_number": 15230,
        "average_rating": 4.6,
    },
    {
        "parent_asin": "B00FIXTURE3",
        "title": "One Size Fits Most Beach Cover-Up",
        "features": ["100% Polyester", "One Size Fits Most (1X to 3X)"],
        "description": ["A breezy cover-up for beach trips and pool days."],
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Swimsuit Cover Ups"],
        "store": "SunDaze",
        "details": {},
        "price": None,
        "rating_number": 12,
        "average_rating": 4.1,
    },
]


def load_catalog(path: str = "data/catalog.jsonl") -> list[dict]:
    """Load the catalogue.

    Design doc §3.2 (offline stage input) and §8.0 ("Never commit large
    binaries to the repo directly; ship a `make data` download script").

    STUB: ignores `path` and returns `FIXTURE_CATALOG` (3 rows) instead of
    the real 50,000-row catalog.jsonl.

    Args:
        path: Path to catalog.jsonl on disk.

    Returns:
        A list of raw product record dicts.
    """
    return FIXTURE_CATALOG


def product_text(row: dict) -> str:
    """Render the single text view of a product shared by all three indexes.

    Design doc §3.2: "product_text(r) = title + features + description +
    categories + store + details[Department, Material, Color, Brand,
    Style]". Owner Haojun, §7.2/§8.1 step A1 (BLOCKING) — "All three indexes
    import it. No local copies exist anywhere in the repo."

    STUB: does not concatenate the real fields; returns a fixture string
    tagged with the row's `parent_asin` so downstream stubs stay traceable
    per-product without implementing the real join/weighting logic.

    Args:
        row: A raw catalogue record (see FIXTURE_CATALOG for shape).

    Returns:
        The product's text blob, one string per ASIN.
    """
    asin = row.get("parent_asin", "UNKNOWN")
    return f"[STUB product_text for {asin}]"


def fixture_score(seed: str) -> float:
    """Deterministic pseudo-score in [0, 1], keyed by `seed`.

    Not part of the design document. A shared helper so every stubbed
    scoring function (features.py, rank.py, clarify.py) returns varied,
    reproducible numbers instead of a single repeated constant, without any
    of them implementing real scoring logic ahead of §3.4 Step 6.

    Args:
        seed: Any string to key the pseudo-score off of, e.g. an ASIN.

    Returns:
        A float in [0, 1], stable across runs for the same seed.
    """
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF
