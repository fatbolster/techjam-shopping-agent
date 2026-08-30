"""Regression tests for high-precision extraction and state repair."""

from copy import deepcopy

import pytest

from extract import (
    apply_slot_operations,
    build_attribute_gazetteer,
    detect_slot_operations,
    extract_slots,
    update_slots,
)
from state import init_state, set_pending_clarification


CATALOG = [
    {
        "categories": ["Root", "Women", "Jackets & Vests", "Vests"],
        "store": "Jackets",
        "details": {"Style": "Outdoor"},
    },
    {
        "categories": ["Root", "Men", "Shoes", "Running"],
        "store": "Nike",
        "details": {"Style": "Polyester"},
    },
    {
        "categories": ["Root", "Women", "Boots", "Outdoor", "Cotton"],
        "store": "waterproof",
        "details": {},
    },
    {"categories": ["Root", "Women"], "store": "NOT", "details": {}},
    {"categories": ["Root", "Women"], "store": "Key", "details": {}},
    {"categories": ["Root", "Women"], "store": "Style", "details": {}},
    {"categories": ["Root", "Women"], "store": "SOLE", "details": {}},
]
GAZETTEER = build_attribute_gazetteer(CATALOG)


def test_scalar_replacement_preserves_positive_evidence_for_b5() -> None:
    state = init_state("scalar-replacement")
    state.slots = {"department": "Women", "category": "Women"}
    state.slot_override_flags = {"department": False, "category": False}
    message = "not women, men instead"

    extraction = extract_slots(message, GAZETTEER)
    operations = detect_slot_operations(message, state, extraction, GAZETTEER)
    apply_slot_operations(state, operations)

    assert any(
        observation.value == "Men" for observation in extraction.observations
    )
    assert state.slots["department"] == "Men"
    assert state.slots["category"] == "Men"
    assert "Women" not in state.slots.values()
    assert state.slots.get("brand") != "NOT"


def test_requested_material_reply_with_cotton_is_material_only() -> None:
    result = extract_slots(
        "For that, what matters is: Cotton.",
        GAZETTEER,
        requested_attribute="material",
    )

    assert result.slots == {"material": ("cotton",)}


def test_feature_requirement_with_waterproof_has_no_brand_collision() -> None:
    result = extract_slots(
        "Actually, ignore my earlier preference. What I need is: "
        "100% waterproof women's boots are suitable for any season and "
        "any outdoor activity.",
        GAZETTEER,
    )

    assert "waterproof" in result.slots["feature"]
    assert "brand" not in result.slots
    assert "category" not in result.slots
    assert "style" not in result.slots


def test_locally_explicit_vests_category_wins_over_jackets() -> None:
    result = extract_slots(
        "I'm looking for Jackets & Vests Vests. Hand Wash Only",
        GAZETTEER,
    )

    assert result.slots["category"] == "Vests"
    assert "brand" not in result.slots


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "A key requirement is: cotton.",
            {"material": ("cotton",)},
        ),
        (
            "For that, what matters is: cotton.",
            {"material": ("cotton",)},
        ),
        (
            "Actually, ignore my earlier preference. What I need is: cotton.",
            {"material": ("cotton",)},
        ),
    ],
)
def test_evaluator_scaffolding_never_becomes_a_constraint(
    message: str, expected: dict
) -> None:
    assert extract_slots(message, GAZETTEER).slots == expected


@pytest.mark.parametrize(
    "message",
    [
        "Those options are not quite right yet.",
        "A key requirement is still missing.",
        "I don't have an additional preference for style.",
        "The sole issue is comfort.",
    ],
)
def test_catalog_control_words_are_not_inferred_as_brands(message: str) -> None:
    assert "brand" not in extract_slots(message, GAZETTEER).slots


@pytest.mark.parametrize(
    ("message", "expected_slots"),
    [
        ("Nike", {"brand": "Nike"}),
        ("Nike running shoes", {"category": "Running", "brand": "Nike"}),
        ("I prefer Nike", {"brand": "Nike"}),
    ],
)
def test_legitimate_brand_syntax_survives(
    message: str, expected_slots: dict
) -> None:
    assert extract_slots(message, GAZETTEER).slots == expected_slots


def test_shared_catalog_vocabulary_uses_requirement_context() -> None:
    material = extract_slots("What I need is: cotton.", GAZETTEER)
    use_case = extract_slots("Something for outdoor use.", GAZETTEER)

    assert material.slots == {"material": ("cotton",)}
    assert use_case.slots.get("use_case") == ("outdoor",)
    assert "category" not in use_case.slots
    assert "style" not in use_case.slots


def test_override_removes_repeated_unstructured_initial_preference() -> None:
    state = init_state("override-provenance")
    update_slots(
        state,
        "I'm looking for Running. Pull On closure",
        GAZETTEER,
    )
    set_pending_clarification(state, "feature")
    update_slots(
        state,
        "For that, what matters is: Imported; Pull On closure.",
        GAZETTEER,
    )

    update_slots(
        state,
        "Actually, ignore my earlier preference. What I need is: cotton.",
        GAZETTEER,
    )

    assert state.slots["category"] == "Running"
    assert state.slots["material"] == ("cotton",)
    assert state.slots["feature"] == ("imported",)
