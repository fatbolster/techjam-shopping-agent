"""Table-driven tests for Owner B's B4 state transition engine."""

from copy import deepcopy

import pytest

from extract import (
    ExtractionResult,
    SlotOperation,
    apply_extraction_result,
    apply_slot_operation,
    apply_slot_operations,
    build_attribute_gazetteer,
    update_slots,
)
from state import (
    ExplicitSlots,
    SlotOverrideFlags,
    clear_pending_clarification,
    consume_pending_clarification,
    init_state,
    set_pending_clarification,
)
from utils import FIXTURE_CATALOG


@pytest.mark.parametrize(
    (
        "initial_slots",
        "initial_flags",
        "operation",
        "expected_slots",
        "expected_flags",
    ),
    [
        (
            {},
            {},
            SlotOperation("upsert", "brand", ("Nike",)),
            {"brand": "Nike"},
            {"brand": False},
        ),
        (
            {"brand": "Nike"},
            {"brand": False},
            SlotOperation("upsert", "brand", ("nike",)),
            {"brand": "Nike"},
            {"brand": False},
        ),
        (
            {"brand": "Nike"},
            {"brand": False},
            SlotOperation("upsert", "brand", ("Adidas",)),
            {"brand": "Adidas"},
            {"brand": True},
        ),
        (
            {},
            {},
            SlotOperation("upsert", "feature", ("waterproof",)),
            {"feature": ("waterproof",)},
            {"feature": False},
        ),
        (
            {"feature": ("waterproof",)},
            {"feature": False},
            SlotOperation("upsert", "feature", ("lightweight",)),
            {"feature": ("waterproof", "lightweight")},
            {"feature": False},
        ),
        (
            {"feature": ("waterproof",)},
            {"feature": False},
            SlotOperation("upsert", "feature", ("Waterproof",)),
            {"feature": ("waterproof",)},
            {"feature": False},
        ),
        (
            {"feature": ("waterproof", "lightweight")},
            {"feature": False},
            SlotOperation("replace", "feature", ("lightweight",)),
            {"feature": ("lightweight",)},
            {"feature": True},
        ),
        (
            {"feature": ("waterproof", "lightweight")},
            {"feature": False},
            SlotOperation("delete_value", "feature", ("waterproof",)),
            {"feature": ("lightweight",)},
            {"feature": True},
        ),
        (
            {"feature": ("waterproof",)},
            {"feature": True},
            SlotOperation("delete_value", "feature", ("Waterproof",)),
            {},
            {},
        ),
        (
            {"feature": ("waterproof",)},
            {"feature": True},
            SlotOperation("delete_slot", "feature"),
            {},
            {},
        ),
    ],
)
def test_structured_transition_table(
    initial_slots: ExplicitSlots,
    initial_flags: SlotOverrideFlags,
    operation: SlotOperation,
    expected_slots: ExplicitSlots,
    expected_flags: SlotOverrideFlags,
) -> None:
    state = init_state("transition")
    state.slots = deepcopy(initial_slots)
    state.slot_override_flags = deepcopy(initial_flags)

    returned = apply_slot_operation(state, operation)

    assert returned is state
    assert state.slots == expected_slots
    assert state.slot_override_flags == expected_flags


def test_additive_update_preserves_existing_override_flag() -> None:
    state = init_state("sticky-override")
    state.slots["feature"] = ("waterproof",)
    state.slot_override_flags["feature"] = True

    apply_slot_operation(
        state, SlotOperation("upsert", "feature", ("lightweight",))
    )

    assert state.slots["feature"] == ("waterproof", "lightweight")
    assert state.slot_override_flags["feature"] is True


def test_one_slot_update_does_not_change_unrelated_slots() -> None:
    state = init_state("independent-slots")
    state.slots = {
        "brand": "Nike",
        "color": ("black",),
        "material": ("cotton",),
    }
    state.slot_override_flags = {
        "brand": False,
        "color": False,
        "material": False,
    }

    apply_slot_operation(state, SlotOperation("upsert", "brand", ("Adidas",)))

    assert state.slots == {
        "brand": "Adidas",
        "color": ("black",),
        "material": ("cotton",),
    }
    assert state.slot_override_flags == {
        "brand": True,
        "color": False,
        "material": False,
    }


def test_price_fields_do_not_implicitly_delete_each_other() -> None:
    state = init_state("independent-price")

    apply_slot_operations(
        state,
        (
            SlotOperation("upsert", "price_target", ("100",)),
            SlotOperation("upsert", "price_max", ("120",)),
        ),
    )

    assert state.slots == {"price_max": "120", "price_target": "100"}
    assert state.slot_override_flags == {
        "price_max": False,
        "price_target": False,
    }


def test_extraction_result_applies_positive_slots_but_not_scenario() -> None:
    state = init_state("extraction")
    state.scenario_buffer = "existing scenario"
    extraction = ExtractionResult(
        slots={"brand": "Nike", "feature": ("waterproof", "lightweight")},
        residual_scenario="new scenario",
        observations=(),
    )

    apply_extraction_result(state, extraction)

    assert state.slots == {
        "brand": "Nike",
        "feature": ("waterproof", "lightweight"),
    }
    assert state.scenario_buffer == "existing scenario"


def test_pending_clarification_is_consumed_once_and_history_remains() -> None:
    state = init_state("pending")
    state.asked_attributes.add("style")
    set_pending_clarification(state, "style")

    assert consume_pending_clarification(state) == "style"
    assert state.pending_clarification is None
    assert consume_pending_clarification(state) is None
    assert state.asked_attributes == {"style"}

    set_pending_clarification(state, "material")
    clear_pending_clarification(state)
    assert state.pending_clarification is None


def test_update_slots_consumes_clarification_and_uses_b3_result() -> None:
    state = init_state("clarification-integration")
    state.scenario_buffer = "for a beach holiday"
    state.asked_attributes.add("style")
    set_pending_clarification(state, "style")
    gazetteer = build_attribute_gazetteer(FIXTURE_CATALOG)

    update_slots(
        state,
        "For that, what matters is: classic.",
        gazetteer,
    )

    assert state.slots == {"style": ("classic",)}
    assert state.slot_override_flags == {"style": False}
    assert state.pending_clarification is None
    assert state.asked_attributes == {"style"}
    assert state.scenario_buffer == "for a beach holiday"
