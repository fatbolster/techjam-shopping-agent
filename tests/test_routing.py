"""Contract tests for Owner B's deterministic B8 retrieval-track router."""

from copy import deepcopy

import pytest

from extract import (
    SlotOperation,
    apply_slot_operation,
    build_attribute_gazetteer,
    update_slots,
)
from state import ExplicitSlots, SessionState, Track, init_state, pick_track
from utils import FIXTURE_CATALOG


GAZETTEER = build_attribute_gazetteer(deepcopy(FIXTURE_CATALOG))


def route(slots: ExplicitSlots, scenario: str = "") -> Track:
    state = init_state("route")
    state.slots = slots
    state.scenario_buffer = scenario
    return pick_track(state)


def test_empty_state_browses() -> None:
    assert route({}) == "browse"


def test_scenario_only_browses() -> None:
    assert route({}, "Something for a beach holiday.") == "browse"


def test_department_only_browses() -> None:
    assert route({"department": "Men"}) == "browse"


@pytest.mark.parametrize(
    "category",
    [
        "Clothing, Shoes & Jewelry",
        "Men",
        "Women",
        "Clothing",
        "Shoes",
    ],
)
def test_known_broad_catalogue_categories_browse(category: str) -> None:
    assert route({"category": category}) == "browse"


@pytest.mark.parametrize(
    "category",
    ["Running", "Jackets & Coats", "Swimsuit Cover Ups"],
)
def test_specific_catalogue_categories_buy(category: str) -> None:
    assert route({"category": category}) == "buy"


def test_unknown_nonempty_category_uses_documented_specific_default() -> None:
    assert route({"category": "Specialty Footwear"}) == "buy"


def test_brand_only_browses() -> None:
    assert route({"brand": "Nike"}) == "browse"


def test_feature_and_use_case_do_not_authorize_buying() -> None:
    assert route(
        {"feature": ("waterproof",), "use_case": ("hiking",)}
    ) == "browse"


def test_price_and_feature_do_not_authorize_buying() -> None:
    assert route(
        {"price_max": "100", "feature": ("waterproof",)}
    ) == "browse"


def test_specific_category_with_brand_buys() -> None:
    assert route({"category": "Running", "brand": "Nike"}) == "buy"


def test_specific_category_with_descriptive_attributes_buys() -> None:
    assert route(
        {
            "category": "Running",
            "color": ("blue",),
            "material": ("cotton",),
        }
    ) == "buy"


def test_broad_category_with_two_descriptive_attributes_still_browses() -> None:
    assert route(
        {
            "category": "Clothing",
            "color": ("blue",),
            "material": ("cotton",),
        }
    ) == "browse"


def test_department_plus_broad_category_still_browses() -> None:
    assert route({"department": "Men", "category": "Clothing"}) == "browse"


def test_route_can_transition_from_browse_to_buy() -> None:
    state = init_state("browse-to-buy")
    update_slots(state, "Something for hiking.", GAZETTEER)

    assert pick_track(state) == "browse"
    assert state.track == "browse"

    update_slots(state, "Men's waterproof running shoes.", GAZETTEER)

    assert state.slots["department"] == "Men"
    assert state.slots["category"] == "Running"
    assert pick_track(state) == "buy"
    assert state.track == "buy"


def test_route_can_transition_from_buy_to_browse_after_deletion() -> None:
    state = init_state("buy-to-browse")
    state.slots = {"category": "Running", "color": ("black",)}
    assert pick_track(state) == "buy"

    apply_slot_operation(state, SlotOperation("delete_slot", "category"))

    assert state.slots == {"color": ("black",)}
    assert pick_track(state) == "browse"
    assert state.track == "browse"


def test_previous_route_is_not_sticky() -> None:
    state = init_state("not-sticky")
    state.track = "buy"
    state.slots = {"feature": ("comfortable",), "use_case": ("travel",)}

    assert pick_track(state) == "browse"

    state.track = "browse"
    state.slots = {"category": "Running"}

    assert pick_track(state) == "buy"


def test_profile_differences_do_not_change_route() -> None:
    first = init_state("profile-a", {"preference_tags": ["weather"]})
    second = init_state("profile-b", {"preference_tags": []})
    first.slots = {"feature": ("comfortable",)}
    second.slots = deepcopy(first.slots)

    assert pick_track(first) == pick_track(second) == "browse"


def test_internal_metadata_differences_do_not_change_route() -> None:
    first = init_state("metadata-a")
    second = init_state("metadata-b")
    for state in (first, second):
        state.slots = {"feature": ("comfortable",), "use_case": ("travel",)}

    first.slot_override_flags = {"feature": True}
    first.override_reference_values = {"feature": ("waterproof",)}
    first.asked_attributes = {"category", "brand"}
    first.pending_clarification = "style"
    first.canonical_intent = "category: Running"
    first.track = "buy"

    assert pick_track(first) == pick_track(second) == "browse"


def test_same_state_always_produces_same_route() -> None:
    state = SessionState(
        session_id="deterministic",
        slots={"category": "Running", "brand": "Nike"},
    )

    assert [pick_track(state) for _ in range(5)] == ["buy"] * 5
