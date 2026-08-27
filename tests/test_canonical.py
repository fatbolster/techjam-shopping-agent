"""Contract tests for Owner B's deterministic B7 current-intent renderer."""

from copy import deepcopy

import numpy as np
import pytest

from extract import (
    SlotOperation,
    apply_slot_operation,
    build_attribute_gazetteer,
    update_slots,
)
from state import init_state, reconstruct_canonical, render
from utils import FIXTURE_CATALOG


CATALOG = deepcopy(FIXTURE_CATALOG) + [
    {
        "categories": ["Root", "Men", "Shoes", "Running"],
        "store": "Nike",
        "details": {},
    },
    {
        "categories": ["Root", "Men", "Shoes", "Running"],
        "store": "Adidas",
        "details": {},
    },
]
GAZETTEER = build_attribute_gazetteer(CATALOG)


@pytest.mark.parametrize(
    ("slots", "scenario", "expected"),
    [
        (
            {"category": "Running", "color": ("black",)},
            "",
            "category: Running; color: black",
        ),
        (
            {
                "brand": "Nike",
                "category": "Running",
                "feature": ("waterproof",),
            },
            "",
            "category: Running; brand: Nike; features: waterproof",
        ),
        (
            {"feature": ("waterproof", "lightweight")},
            "",
            "features: waterproof, lightweight",
        ),
        ({"price_max": "100"}, "", "under $100"),
        ({"price_min": "50"}, "", "at least $50"),
        (
            {"price_max": "90", "price_min": "40"},
            "",
            "between $40 and $90",
        ),
        ({"price_target": "100"}, "", "around $100"),
        ({}, "for a beach holiday", "for a beach holiday"),
        (
            {"color": ("blue",)},
            "for a beach holiday — with lots of walking",
            "color: blue; for a beach holiday — with lots of walking",
        ),
        ({}, "", ""),
    ],
)
def test_exact_canonical_rendering(
    slots: dict,
    scenario: str,
    expected: str,
) -> None:
    state = init_state("render")
    state.slots = slots
    state.scenario_buffer = scenario

    assert render(state) == expected


def test_all_non_price_slots_use_stable_readable_order() -> None:
    state = init_state("order")
    state.slots = {
        "use_case": ("hiking", "work"),
        "feature": ("waterproof", "lightweight"),
        "size": ("large",),
        "material": ("cotton",),
        "color": ("blue", "black"),
        "style": ("classic",),
        "brand": "Nike",
        "category": "Running",
        "department": "Men",
    }

    assert render(state) == (
        "department: Men; category: Running; brand: Nike; style: classic; "
        "color: blue, black; material: cotton; size: large; "
        "features: waterproof, lightweight; use case: hiking, work"
    )


def test_all_active_price_semantics_are_retained() -> None:
    state = init_state("prices")
    state.slots = {
        "price_min": "40",
        "price_max": "90",
        "price_target": "65",
    }

    assert render(state) == "between $40 and $90; around $65"


def test_render_ignores_internal_metadata_and_previous_semantic_state() -> None:
    first = init_state("first", {"preference_tags": ["weather"]})
    second = init_state("second")
    for state in (first, second):
        state.slots = {"category": "Running", "color": ("blue",)}
        state.scenario_buffer = "for a beach holiday"

    first.slot_override_flags = {"color": True}
    first.override_reference_values = {"color": ("black",)}
    first.asked_attributes = {"brand", "style"}
    first.pending_clarification = "material"
    first.canonical_intent = "stale black query"
    first.canonical_vector = np.ones(3)
    first.track = "buy"

    assert render(first) == render(second)
    assert render(first) == (
        "category: Running; color: blue; for a beach holiday"
    )


def test_black_to_blue_rebuild_contains_no_stale_color() -> None:
    state = init_state("color-override")
    update_slots(state, "I want black running shoes.", GAZETTEER)
    assert render(state) == "category: Running; color: black"

    update_slots(state, "Actually not black, blue instead.", GAZETTEER)

    assert render(state) == "category: Running; color: blue"
    assert "black" not in render(state).casefold()


def test_nike_to_adidas_rebuild_contains_no_stale_brand() -> None:
    state = init_state("brand-override")
    update_slots(state, "Nike", GAZETTEER)
    assert render(state) == "brand: Nike"

    update_slots(state, "Actually Adidas", GAZETTEER)

    assert render(state) == "brand: Adidas"
    assert "nike" not in render(state).casefold()


def test_partial_multi_value_deletion_disappears_from_rebuild() -> None:
    state = init_state("partial-delete")
    state.slots = {"material": ("leather", "cotton")}
    assert render(state) == "material: leather, cotton"

    apply_slot_operation(
        state, SlotOperation("delete_value", "material", ("leather",))
    )

    assert render(state) == "material: cotton"
    assert "leather" not in render(state).casefold()


def test_scenario_replacement_by_structured_use_case_removes_old_terms() -> None:
    state = init_state("scenario-replace")
    update_slots(state, "Something for a beach holiday.", GAZETTEER)
    assert "beach" in render(state).casefold()

    update_slots(state, "Actually this is for hiking instead.", GAZETTEER)

    assert render(state) == "use case: hiking"
    assert "beach" not in render(state).casefold()


def test_scenario_clear_produces_empty_current_intent() -> None:
    state = init_state("scenario-clear")
    update_slots(state, "Something for my honeymoon.", GAZETTEER)
    assert render(state) == "Something for my honeymoon."

    update_slots(state, "Forget the honeymoon part.", GAZETTEER)

    assert render(state) == ""


def test_reconstruct_embeds_each_nonempty_render_fresh() -> None:
    state = init_state("embedding")
    calls: list[str] = []

    def encoder(text: str) -> np.ndarray:
        calls.append(text)
        return np.array([len(calls), len(text)], dtype=np.float32)

    state.slots = {"color": ("black",)}
    reconstruct_canonical(state, encoder)
    first_vector = state.canonical_vector

    state.slots = {"color": ("blue",)}
    reconstruct_canonical(state, encoder)

    assert calls == ["color: black", "color: blue"]
    assert state.canonical_intent == "color: blue"
    assert state.canonical_vector is not first_vector
    np.testing.assert_array_equal(
        state.canonical_vector, np.array([2, 11], dtype=np.float32)
    )


def test_empty_reconstruction_clears_stale_vector_without_encoding() -> None:
    state = init_state("empty-embedding")
    state.canonical_intent = "stale query"
    state.canonical_vector = np.ones(3)

    def encoder(_: str) -> np.ndarray:
        raise AssertionError("empty canonical intent must not be embedded")

    reconstruct_canonical(state, encoder)

    assert state.canonical_intent == ""
    assert state.canonical_vector is None
