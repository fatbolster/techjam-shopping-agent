"""
Contract tests for indexes.py's four offline indexes (design doc §3.2,
§8.1 steps A3-A6) and utils.py's product_text()/load_catalog() (A1).

knn_search() (A6) was already real before this file existed (matmul +
argpartition needs no catalogue-specific logic) and is covered lightly
here for completeness; the rest of this file is what actually changed —
build_fts5_index()/keyword_search() (A4), embed_text()/
build_embedding_matrix() (A5), build_facts_dict()/build_category_lists()
(A3). Embedding tests are marked to skip without sentence-transformers
installed, same convention as rank.py's sklearn-dependent tests.
"""

import json

import numpy as np
import pytest

from indexes import (
    EMBEDDING_DIM,
    UNKNOWN_CATEGORY,
    Indexes,
    build_category_lists,
    build_embedding_matrix,
    build_facts_dict,
    build_fts5_index,
    build_indexes,
    embed_text,
    keyword_search,
    knn_search,
)
from utils import FIXTURE_CATALOG, load_catalog, product_text

sentence_transformers = pytest.importorskip("sentence_transformers")


# --------------------------------------------------------------------------
# utils.product_text() / load_catalog() — A1
# --------------------------------------------------------------------------

def test_product_text_includes_title_features_description_categories_store():
    row = FIXTURE_CATALOG[0]  # the London Fog jacket
    text = product_text(row)
    assert row["title"] in text
    for feature in row["features"]:
        assert feature in text
    for desc in row["description"]:
        assert desc in text
    for category in row["categories"]:
        assert category in text
    assert row["store"] in text


def test_product_text_includes_the_five_detail_keys_in_order():
    """Detail values are appended last, in PRODUCT_TEXT_DETAIL_KEYS order.
    Uses rfind rather than find: this fixture's title happens to already
    contain "Auburn" and "Golf Jacket" (the color/style detail values), so
    their *first* occurrence is mid-title — the *last* occurrence is what
    the appended-last details section actually controls."""
    row = FIXTURE_CATALOG[0]
    text = product_text(row)
    detail_values = [row["details"][k] for k in ["Department", "Material", "Color", "Brand", "Style"]]
    positions = [text.rfind(v) for v in detail_values]
    assert all(p != -1 for p in positions)
    assert positions == sorted(positions)  # appear in the documented order


def test_product_text_survives_empty_details():
    row = FIXTURE_CATALOG[2]  # details: {}
    text = product_text(row)
    assert row["title"] in text  # doesn't crash, still has the other fields


def test_product_text_survives_missing_optional_fields():
    text = product_text({"parent_asin": "X"})  # no title/features/etc. at all
    assert text == ""


def test_load_catalog_falls_back_to_fixture_when_path_absent():
    assert load_catalog("/nonexistent/path/catalog.jsonl") == FIXTURE_CATALOG


def test_load_catalog_reads_real_jsonl(tmp_path):
    path = tmp_path / "catalog.jsonl"
    rows = [{"parent_asin": "A"}, {"parent_asin": "B"}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert load_catalog(str(path)) == rows


def test_load_catalog_skips_blank_lines(tmp_path):
    path = tmp_path / "catalog.jsonl"
    path.write_text('{"parent_asin": "A"}\n\n{"parent_asin": "B"}\n')
    assert load_catalog(str(path)) == [{"parent_asin": "A"}, {"parent_asin": "B"}]


# --------------------------------------------------------------------------
# build_fts5_index() / keyword_search() — A4
# --------------------------------------------------------------------------

@pytest.fixture
def fts_conn():
    return build_fts5_index(FIXTURE_CATALOG)


def test_keyword_search_finds_by_title_token(fts_conn):
    results = keyword_search(fts_conn, "Saucony", limit=10)
    assert "B00FIXTURE2" in {asin for asin, _ in results}


def test_keyword_search_scores_are_sign_corrected_higher_is_better(fts_conn):
    """A query matching only the golf jacket on two columns (title + a
    detail value) must score higher than one matching it on one column
    only — bm25's raw more-negative-is-better must have been flipped."""
    two_column_hit = dict(keyword_search(fts_conn, "London Auburn", limit=10))
    one_column_hit = dict(keyword_search(fts_conn, "London", limit=10))
    assert two_column_hit["B00FIXTURE1"] > one_column_hit["B00FIXTURE1"]
    assert all(score > 0 or score < 0 for score in two_column_hit.values())  # real floats, not placeholders


def test_keyword_search_or_matches_when_only_some_tokens_present(fts_conn):
    """Design intent: over-fetch and let downstream filter/truncate, not a
    strict AND filter — a query with one uncatalogued word must still
    match products sharing the other words."""
    results = keyword_search(fts_conn, "Saucony xyzzynonexistentword", limit=10)
    assert "B00FIXTURE2" in {asin for asin, _ in results}


def test_keyword_search_respects_limit(fts_conn):
    results = keyword_search(fts_conn, "Clothing", limit=1)
    assert len(results) <= 1


def test_keyword_search_empty_query_returns_empty(fts_conn):
    assert keyword_search(fts_conn, "", limit=10) == []


def test_keyword_search_query_with_only_punctuation_returns_empty(fts_conn):
    """Must not raise an FTS5 syntax error on a query with no word tokens."""
    assert keyword_search(fts_conn, ": $ - ,", limit=10) == []


def test_keyword_search_query_with_colons_does_not_raise(fts_conn):
    """Canonical intent strings look like 'department: Men category: Jackets'
    — the raw colon must not be interpreted as an FTS5 column filter."""
    results = keyword_search(fts_conn, "department: Men category: Jackets", limit=10)
    assert isinstance(results, list)  # doesn't raise sqlite3.OperationalError


def test_keyword_search_no_match_returns_empty(fts_conn):
    assert keyword_search(fts_conn, "zzzznonexistenttoken", limit=10) == []


# --------------------------------------------------------------------------
# embed_text() / build_embedding_matrix() — A5
# --------------------------------------------------------------------------

def test_embed_text_returns_unit_vector_of_expected_dimension():
    vec = embed_text("a running shoe")
    assert vec.shape == (EMBEDDING_DIM,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


def test_embed_text_is_deterministic():
    a = embed_text("a running shoe")
    b = embed_text("a running shoe")
    assert np.allclose(a, b)


def test_embed_text_different_text_gives_different_vector():
    a = embed_text("a running shoe")
    b = embed_text("a beach cover-up")
    assert not np.allclose(a, b)


def test_build_embedding_matrix_shape_and_row_alignment():
    matrix, asins = build_embedding_matrix(FIXTURE_CATALOG, cache_path=None)
    assert matrix.shape == (len(FIXTURE_CATALOG), EMBEDDING_DIM)
    assert asins == [row["parent_asin"] for row in FIXTURE_CATALOG]
    assert matrix.dtype == np.float32


def test_build_embedding_matrix_rows_match_embed_text_of_product_text():
    """Batch encoding (build_embedding_matrix) vs single-item encoding
    (embed_text) can differ at the ULP level from batching/padding
    internals — atol=1e-4 tolerates that noise while still catching a
    real divergence (e.g. wrong row, wrong text)."""
    matrix, asins = build_embedding_matrix(FIXTURE_CATALOG, cache_path=None)
    expected = embed_text(product_text(FIXTURE_CATALOG[0]))
    assert np.allclose(matrix[asins.index(FIXTURE_CATALOG[0]["parent_asin"])], expected, atol=1e-4)


def test_build_embedding_matrix_empty_catalog():
    matrix, asins = build_embedding_matrix([], cache_path=None)
    assert matrix.shape == (0, EMBEDDING_DIM)
    assert asins == []


def test_build_embedding_matrix_no_cache_path_never_touches_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build_embedding_matrix(FIXTURE_CATALOG, cache_path=None)
    assert list(tmp_path.iterdir()) == []


def test_build_embedding_matrix_persists_and_reloads_from_cache(tmp_path):
    cache_path = str(tmp_path / "embeddings.npy")
    matrix1, asins1 = build_embedding_matrix(FIXTURE_CATALOG, cache_path=cache_path)
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "embeddings.npy.asins.json").exists()

    # Corrupt-proof the "loaded from cache, not recomputed" check: mutate
    # the cache file directly, then confirm the second call reads it back
    # rather than re-encoding (which would restore the original values).
    tampered = matrix1.copy()
    tampered[0, 0] = 999.0
    np.save(cache_path, tampered)

    matrix2, asins2 = build_embedding_matrix(FIXTURE_CATALOG, cache_path=cache_path)
    assert asins2 == asins1
    assert matrix2[0, 0] == 999.0  # came from the tampered cache, not a fresh encode


def test_build_embedding_matrix_cache_invalidated_on_asin_mismatch(tmp_path):
    cache_path = str(tmp_path / "embeddings.npy")
    build_embedding_matrix(FIXTURE_CATALOG[:1], cache_path=cache_path)
    # A differently-sized catalog must not silently reuse the stale cache.
    matrix, asins = build_embedding_matrix(FIXTURE_CATALOG, cache_path=cache_path)
    assert len(asins) == len(FIXTURE_CATALOG)
    assert matrix.shape[0] == len(FIXTURE_CATALOG)


# --------------------------------------------------------------------------
# knn_search() — A6 (already real; light coverage for completeness)
# --------------------------------------------------------------------------

def test_knn_search_returns_query_itself_as_top_hit():
    matrix, asins = build_embedding_matrix(FIXTURE_CATALOG, cache_path=None)
    query_vec = matrix[0]
    results = knn_search(matrix, asins, query_vec, k=3)
    assert results[0][0] == asins[0]
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)  # cosine sim of a vector with itself


def test_knn_search_empty_matrix_returns_empty():
    assert knn_search(np.zeros((0, EMBEDDING_DIM), dtype=np.float32), [], np.zeros(EMBEDDING_DIM), k=5) == []


# --------------------------------------------------------------------------
# build_facts_dict() / build_category_lists() — A3
# --------------------------------------------------------------------------

@pytest.fixture
def facts():
    return build_facts_dict(FIXTURE_CATALOG)


def test_build_facts_dict_dept_is_categories_index_1(facts):
    assert facts["B00FIXTURE1"]["dept"] == "Men"
    assert facts["B00FIXTURE3"]["dept"] == "Women"


def test_build_facts_dict_cat3_is_categories_index_2(facts):
    assert facts["B00FIXTURE1"]["cat3"] == "Clothing"


def test_build_facts_dict_dept_and_cat3_none_when_categories_too_short():
    facts = build_facts_dict([{"parent_asin": "X", "categories": ["Only one level"]}])
    assert facts["X"]["dept"] is None
    assert facts["X"]["cat3"] is None


def test_build_facts_dict_pop_formula():
    facts = build_facts_dict([{"parent_asin": "X", "rating_number": 8421}])
    assert facts["X"]["pop"] == pytest.approx(np.log1p(8421) / np.log1p(100_000))


def test_build_facts_dict_pop_zero_when_rating_number_missing():
    facts = build_facts_dict([{"parent_asin": "X"}])
    assert facts["X"]["pop"] == 0.0


def test_build_facts_dict_blob_is_lowercased_product_text(facts):
    assert facts["B00FIXTURE1"]["blob"] == product_text(FIXTURE_CATALOG[0]).lower()
    assert "AUBURN" not in facts["B00FIXTURE1"]["blob"]  # confirms it's actually lowercased


def test_build_facts_dict_carries_store_price_rating(facts):
    assert facts["B00FIXTURE1"]["store"] == "London Fog"
    assert facts["B00FIXTURE1"]["price"] == 64.99
    assert facts["B00FIXTURE1"]["rating"] == 4.5


def test_build_category_lists_groups_by_dept(facts):
    lists = build_category_lists(FIXTURE_CATALOG, facts)
    assert set(lists["Men"]) == {"B00FIXTURE1", "B00FIXTURE2"}
    assert lists["Women"] == ["B00FIXTURE3"]


def test_build_category_lists_sorted_by_rating_number_descending(facts):
    lists = build_category_lists(FIXTURE_CATALOG, facts)
    # B00FIXTURE2 (15230 ratings) must outrank B00FIXTURE1 (8421) within Men.
    assert lists["Men"] == ["B00FIXTURE2", "B00FIXTURE1"]


def test_build_category_lists_unknown_bucket_for_missing_dept():
    catalog = [{"parent_asin": "X", "categories": []}]
    facts = build_facts_dict(catalog)
    lists = build_category_lists(catalog, facts)
    assert lists[UNKNOWN_CATEGORY] == ["X"]


# --------------------------------------------------------------------------
# build_indexes() — composition (A2's stable signature, now real innards)
# --------------------------------------------------------------------------

def test_build_indexes_returns_populated_bundle():
    indexes = build_indexes(FIXTURE_CATALOG, embedding_cache_path=None)
    assert isinstance(indexes, Indexes)
    assert indexes.catalog == FIXTURE_CATALOG
    assert indexes.embedding_matrix.shape == (len(FIXTURE_CATALOG), EMBEDDING_DIM)
    assert set(indexes.facts) == {row["parent_asin"] for row in FIXTURE_CATALOG}
    assert set(indexes.category_lists) == {"Men", "Women"}
    # fts_conn is queryable end to end through the same composition path.
    assert keyword_search(indexes.fts_conn, "Saucony", limit=10)


def test_build_indexes_defaults_to_load_catalog_when_none_given(tmp_path, monkeypatch):
    """catalog=None must fall through to load_catalog()'s own default path
    resolution. Runs from an empty tmp_path (not the repo root, which does
    have a real data/catalog.jsonl) so this exercises the FIXTURE_CATALOG
    fallback rather than encoding the real 50,000-row catalogue."""
    monkeypatch.chdir(tmp_path)
    indexes = build_indexes(embedding_cache_path=None)
    assert indexes.catalog == FIXTURE_CATALOG
