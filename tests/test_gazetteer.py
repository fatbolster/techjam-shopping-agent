"""Focused tests for Owner B's B2 attribute gazetteer."""

from copy import deepcopy

from extract import (
    build_attribute_gazetteer,
    gazetteer_vocabulary_sizes,
    lookup_gazetteer,
)
from utils import FIXTURE_CATALOG


def test_catalogue_backed_values_use_only_labelled_fields() -> None:
    catalog = deepcopy(FIXTURE_CATALOG)
    original = deepcopy(catalog)

    gazetteer = build_attribute_gazetteer(catalog)

    assert lookup_gazetteer(gazetteer, "department", "men's") == "Men"
    assert (
        lookup_gazetteer(gazetteer, "category", "Jackets and Coats")
        == "Jackets & Coats"
    )
    assert lookup_gazetteer(gazetteer, "brand", "LONDON FOG") == "London Fog"
    assert lookup_gazetteer(gazetteer, "style", "golf-jacket") == "Golf Jacket"
    assert catalog == original


def test_fixed_vocabularies_are_deliberately_conservative() -> None:
    gazetteer = build_attribute_gazetteer(FIXTURE_CATALOG)

    assert lookup_gazetteer(gazetteer, "color", "grey") == "gray"
    assert lookup_gazetteer(gazetteer, "color", "auburn") is None
    assert lookup_gazetteer(gazetteer, "material", "POLYESTER") == "polyester"
    assert lookup_gazetteer(gazetteer, "material", "mesh") is None
    assert lookup_gazetteer(gazetteer, "use_case", "winter") == "winter"


def test_arbitrary_product_text_does_not_become_a_slot_value() -> None:
    gazetteer = build_attribute_gazetteer(FIXTURE_CATALOG)

    assert lookup_gazetteer(gazetteer, "feature", "water resistant shell") is None
    assert lookup_gazetteer(gazetteer, "use_case", "cool-weather rounds") is None
    assert lookup_gazetteer(gazetteer, "brand", "cohesion") is None
    assert lookup_gazetteer(gazetteer, "category", "lightweight") is None


def test_pattern_backed_attributes_have_no_gazetteer() -> None:
    gazetteer = build_attribute_gazetteer(FIXTURE_CATALOG)

    assert lookup_gazetteer(gazetteer, "size", "1X") is None
    assert lookup_gazetteer(gazetteer, "price_min", "50") is None
    assert lookup_gazetteer(gazetteer, "price_max", "100") is None


def test_normalized_collisions_choose_the_most_frequent_catalogue_spelling() -> None:
    catalog = [
        {"categories": ["Root", "Men"], "store": "Acme", "details": {}},
        {"categories": ["Root", "Men"], "store": "ACME", "details": {}},
        {"categories": ["Root", "Men"], "store": "ACME", "details": {}},
    ]

    gazetteer = build_attribute_gazetteer(catalog)

    assert lookup_gazetteer(gazetteer, "brand", "acme") == "ACME"


def test_fixture_vocabulary_sizes_are_stable() -> None:
    gazetteer = build_attribute_gazetteer(FIXTURE_CATALOG)

    assert gazetteer_vocabulary_sizes(gazetteer) == {
        "department": 4,
        "category": 8,
        "brand": 3,
        "color": 12,
        "material": 9,
        "style": 2,
        "use_case": 6,
    }
