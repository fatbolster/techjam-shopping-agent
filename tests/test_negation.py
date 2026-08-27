"""Regression tests for Owner B's conservative B5 V1 operation planner."""

from copy import deepcopy

import pytest

from extract import (
    SlotOperation,
    apply_slot_operations,
    build_attribute_gazetteer,
    detect_slot_operations,
    extract_slots,
    update_slots,
)
from state import ExplicitSlots, SlotKey, init_state
from utils import FIXTURE_CATALOG


CATALOG = deepcopy(FIXTURE_CATALOG) + [
    {
        "categories": ["Root", "Men", "Shoes"],
        "store": "Nike",
        "details": {},
    },
    {
        "categories": ["Root", "Men", "Shoes"],
        "store": "Adidas",
        "details": {},
    },
]
GAZETTEER = build_attribute_gazetteer(CATALOG)


def plan_and_apply(
    initial_slots: ExplicitSlots,
    message: str,
    scenario: str = "",
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[SlotOperation, ...],
    ExplicitSlots,
    str | None,
    str,
]:
    state = init_state("b5")
    state.slots = deepcopy(initial_slots)
    state.slot_override_flags = {slot: False for slot in initial_slots}
    state.scenario_buffer = scenario
    extraction = extract_slots(message, GAZETTEER)
    before_slots = deepcopy(state.slots)
    before_flags = deepcopy(state.slot_override_flags)

    operations = detect_slot_operations(message, state, extraction, GAZETTEER)

    assert state.slots == before_slots
    assert state.slot_override_flags == before_flags
    apply_slot_operations(state, operations)
    observations = tuple(
        (observation.slot, observation.value)
        for observation in extraction.observations
    )
    return (
        observations,
        operations,
        state.slots,
        extraction.residual_scenario,
        state.scenario_buffer,
    )


@pytest.mark.parametrize(
    (
        "initial_slots",
        "message",
        "expected_observations",
        "expected_operations",
        "expected_slots",
    ),
    [
        (
            {"color": ("black",)},
            "not black",
            (("color", "black"),),
            (SlotOperation("delete_value", "color", ("black",)),),
            {},
        ),
        (
            {"color": ("black",)},
            "not black, blue instead",
            (("color", "black"), ("color", "blue")),
            (SlotOperation("replace", "color", ("blue",)),),
            {"color": ("blue",)},
        ),
        (
            {"brand": "Nike"},
            "actually Adidas",
            (("brand", "Adidas"),),
            (SlotOperation("replace", "brand", ("Adidas",)),),
            {"brand": "Adidas"},
        ),
        (
            {"feature": ("waterproof", "lightweight")},
            "I don't need waterproof anymore",
            (("feature", "waterproof"),),
            (SlotOperation("delete_value", "feature", ("waterproof",)),),
            {"feature": ("lightweight",)},
        ),
        (
            {"material": ("leather", "cotton")},
            "forget leather, just cotton",
            (("material", "leather"), ("material", "cotton")),
            (SlotOperation("replace", "material", ("cotton",)),),
            {"material": ("cotton",)},
        ),
        (
            {"brand": "Nike"},
            "I don't care about brand anymore",
            (),
            (SlotOperation("delete_slot", "brand"),),
            {},
        ),
        (
            {"color": ("blue",)},
            "not black",
            (("color", "black"),),
            (),
            {"color": ("blue",)},
        ),
        (
            {},
            "not red",
            (("color", "red"),),
            (),
            {},
        ),
        (
            {"color": ("blue",)},
            "not sure",
            (),
            (),
            {"color": ("blue",)},
        ),
        (
            {"brand": "Nike"},
            "I don't mind Nike",
            (("brand", "Nike"),),
            (),
            {"brand": "Nike"},
        ),
        (
            {"brand": "Nike", "color": ("black",)},
            "actually Adidas",
            (("brand", "Adidas"),),
            (SlotOperation("replace", "brand", ("Adidas",)),),
            {"brand": "Adidas", "color": ("black",)},
        ),
    ],
)
def test_b5_required_transition_table(
    initial_slots: ExplicitSlots,
    message: str,
    expected_observations: tuple[tuple[str, str], ...],
    expected_operations: tuple[SlotOperation, ...],
    expected_slots: ExplicitSlots,
) -> None:
    observations, operations, slots, _, _ = plan_and_apply(initial_slots, message)

    assert observations == expected_observations
    assert operations == expected_operations
    assert slots == expected_slots


def test_structured_override_does_not_apply_residual_scenario() -> None:
    observations, operations, slots, residual, scenario = plan_and_apply(
        {"color": ("black",)},
        "not black, blue instead for my honeymoon",
        scenario="for a beach trip",
    )

    assert observations == (("color", "black"), ("color", "blue"))
    assert operations == (SlotOperation("replace", "color", ("blue",)),)
    assert slots == {"color": ("blue",)}
    assert residual == "for my honeymoon"
    assert scenario == "for a beach trip"


@pytest.mark.parametrize(
    "message",
    [
        "not sure",
        "not really waterproof",
        "I don't know",
        "nothing too fancy",
    ],
)
def test_uncertain_or_vague_negation_does_not_change_state(message: str) -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"feature": ("waterproof",)}, message
    )

    assert operations == ()
    assert slots == {"feature": ("waterproof",)}


def test_rejected_inactive_value_does_not_delete_active_value() -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"brand": "Nike"}, "Nike is fine, but I don't want Adidas"
    )

    assert operations == ()
    assert slots == {"brand": "Nike"}


def test_product_local_negation_does_not_delete_color_constraint() -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"color": ("black",)}, "I want black shoes, not black socks"
    )

    assert SlotOperation("delete_value", "color", ("black",)) not in operations
    assert slots["color"] == ("black",)


def test_separate_clauses_can_delete_and_add_without_forcing_replacement() -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"feature": ("waterproof",)},
        "not waterproof, but lightweight is important",
    )

    assert operations == (
        SlotOperation("delete_value", "feature", ("waterproof",)),
        SlotOperation("upsert", "feature", ("lightweight",)),
    )
    assert slots == {"feature": ("lightweight",)}


@pytest.mark.parametrize(
    "message",
    [
        "instead of leather, cotton",
        "rather than leather, cotton",
        "forget leather, just cotton",
        "I don't want leather, cotton would be better",
    ],
)
def test_explicit_replacement_constructions(message: str) -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"material": ("leather",)}, message
    )

    assert operations == (SlotOperation("replace", "material", ("cotton",)),)
    assert slots == {"material": ("cotton",)}


@pytest.mark.parametrize(
    "message",
    [
        "no leather",
        "without leather",
        "leather doesn't matter anymore",
    ],
)
def test_explicit_value_deletion_constructions(message: str) -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"material": ("leather", "cotton")}, message
    )

    assert operations == (
        SlotOperation("delete_value", "material", ("leather",)),
    )
    assert slots == {"material": ("cotton",)}


def test_scalar_value_rejection_removes_only_matching_active_constraint() -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"brand": "Nike", "color": ("black",)}, "I don't want Nike"
    )

    assert operations == (SlotOperation("delete_slot", "brand"),)
    assert slots == {"color": ("black",)}


def test_gazetteer_alias_matches_active_canonical_value() -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"color": ("gray",)}, "not grey"
    )

    assert operations == (SlotOperation("delete_value", "color", ("gray",)),)
    assert slots == {}


def test_full_evaluator_override_template_prefers_specific_slot() -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"brand": "Nike"},
        "Actually, ignore my earlier preference. What I need is: Adidas.",
    )

    assert operations == (SlotOperation("replace", "brand", ("Adidas",)),)
    assert slots == {"brand": "Adidas"}


def test_evaluator_override_initial_template_is_an_ordinary_upsert() -> None:
    observations, operations, slots, _, _ = plan_and_apply(
        {}, "I'm looking for Running. lightweight"
    )

    assert observations == (("category", "Running"), ("feature", "lightweight"))
    assert operations == (
        SlotOperation("upsert", "category", ("Running",)),
        SlotOperation("upsert", "feature", ("lightweight",)),
    )
    assert slots == {"category": "Running", "feature": ("lightweight",)}


def test_full_evaluator_override_template_replaces_multi_value_slot() -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"feature": ("lightweight",)},
        "Actually, ignore my earlier preference. What I need is: waterproof.",
    )

    assert operations == (SlotOperation("replace", "feature", ("waterproof",)),)
    assert slots == {"feature": ("waterproof",)}


@pytest.mark.parametrize(
    ("initial_slots", "message", "expected_operations", "expected_slots"),
    [
        (
            {"price_target": "100"},
            "Actually under $80.",
            (
                SlotOperation("delete_slot", "price_target"),
                SlotOperation("upsert", "price_max", ("80",)),
            ),
            {"price_max": "80"},
        ),
        (
            {"price_max": "100"},
            "Actually around $150.",
            (
                SlotOperation("delete_slot", "price_max"),
                SlotOperation("upsert", "price_target", ("150",)),
            ),
            {"price_target": "150"},
        ),
        (
            {"price_min": "50", "price_max": "100"},
            "Actually at least $120.",
            (
                SlotOperation("delete_slot", "price_max"),
                SlotOperation("upsert", "price_min", ("120",)),
            ),
            {"price_min": "120"},
        ),
    ],
)
def test_price_family_revision_plans_cleanup_through_b4(
    initial_slots: ExplicitSlots,
    message: str,
    expected_operations: tuple[SlotOperation, ...],
    expected_slots: ExplicitSlots,
) -> None:
    _, operations, slots, _, _ = plan_and_apply(initial_slots, message)

    assert operations == expected_operations
    assert slots == expected_slots


@pytest.mark.parametrize(
    ("old_preference", "old_slot", "old_value"),
    [
        ("color: black", "color", ("black",)),
        ("leather", "material", ("leather",)),
        ("lightweight", "feature", ("lightweight",)),
    ],
)
def test_evaluator_override_fallback_removes_only_initial_old_preference(
    old_preference: str,
    old_slot: SlotKey,
    old_value: tuple[str, ...],
) -> None:
    state = init_state("fallback")
    update_slots(
        state,
        f"I'm looking for Running. {old_preference}",
        GAZETTEER,
    )

    assert state.slots["category"] == "Running"
    assert state.slots[old_slot] == old_value
    assert state.override_reference_values == {old_slot: old_value}

    update_slots(
        state,
        "Actually, please ignore my earlier preference.",
        GAZETTEER,
    )

    assert state.slots == {"category": "Running"}
    assert state.override_reference_values == {}


def test_evaluator_full_override_keeps_category_and_replaces_old_preference() -> None:
    state = init_state("full-override")
    update_slots(state, "I'm looking for Running. color: black", GAZETTEER)

    update_slots(
        state,
        "Actually, ignore my earlier preference. What I need is: blue.",
        GAZETTEER,
    )

    assert state.slots == {"category": "Running", "color": ("blue",)}


def test_evaluator_full_override_removes_cross_slot_old_preference() -> None:
    state = init_state("cross-slot-full-override")
    update_slots(state, "I'm looking for Running. lightweight", GAZETTEER)

    update_slots(
        state,
        "Actually, ignore my earlier preference. What I need is: leather.",
        GAZETTEER,
    )

    assert state.slots == {
        "category": "Running",
        "material": ("leather",),
    }


def test_evaluator_fallback_preserves_unreferenced_later_values() -> None:
    state = init_state("value-level-fallback")
    update_slots(state, "I'm looking for Running. color: black", GAZETTEER)
    update_slots(state, "I also want blue and leather", GAZETTEER)

    update_slots(
        state,
        "Actually, please ignore my earlier preference.",
        GAZETTEER,
    )

    assert state.slots == {
        "category": "Running",
        "color": ("blue",),
        "material": ("leather",),
    }
    assert state.override_reference_values == {}


def test_evaluator_fallback_without_provenance_does_not_guess_or_reset() -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"category": "Running", "color": ("black",)},
        "Actually, please ignore my earlier preference.",
    )

    assert operations == ()
    assert slots == {"category": "Running", "color": ("black",)}


def test_update_slots_routes_b5_operations_through_b4() -> None:
    state = init_state("b5-integration")
    state.slots = {"color": ("black",)}
    state.slot_override_flags = {"color": False}
    state.scenario_buffer = "for a beach holiday"

    update_slots(state, "not black, blue instead", GAZETTEER)

    assert state.slots == {"color": ("blue",)}
    assert state.slot_override_flags == {"color": True}
    assert state.scenario_buffer == "for a beach holiday"


@pytest.mark.parametrize(
    ("initial_slots", "message", "expected_operations", "expected_slots"),
    [
        (
            {"feature": ("lightweight",)},
            "Actually I also need it waterproof.",
            (SlotOperation("upsert", "feature", ("waterproof",)),),
            {"feature": ("lightweight", "waterproof")},
        ),
        (
            {"brand": "Nike"},
            "Actually Adidas.",
            (SlotOperation("replace", "brand", ("Adidas",)),),
            {"brand": "Adidas"},
        ),
        (
            {"color": ("black",)},
            "Actually I want blue instead.",
            (SlotOperation("replace", "color", ("blue",)),),
            {"color": ("blue",)},
        ),
        (
            {"feature": ("waterproof", "lightweight")},
            "Actually breathable would be useful too.",
            (SlotOperation("upsert", "feature", ("breathable",)),),
            {"feature": ("waterproof", "lightweight", "breathable")},
        ),
        (
            {"category": "Running"},
            "Actually I need something waterproof.",
            (SlotOperation("upsert", "feature", ("waterproof",)),),
            {"category": "Running", "feature": ("waterproof",)},
        ),
    ],
)
def test_actually_only_replaces_with_slot_compatible_revision_evidence(
    initial_slots: ExplicitSlots,
    message: str,
    expected_operations: tuple[SlotOperation, ...],
    expected_slots: ExplicitSlots,
) -> None:
    _, operations, slots, _, _ = plan_and_apply(initial_slots, message)

    assert operations == expected_operations
    assert slots == expected_slots


def test_bare_actually_does_not_replace_a_multi_value_slot() -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"feature": ("lightweight",)}, "Actually waterproof."
    )

    assert operations == (SlotOperation("upsert", "feature", ("waterproof",)),)
    assert slots == {"feature": ("lightweight", "waterproof")}


@pytest.mark.parametrize(
    ("initial_slots", "message", "expected_operations", "expected_slots"),
    [
        (
            {"material": ("leather",)},
            "Leather isn't ideal; cotton would be better.",
            (SlotOperation("replace", "material", ("cotton",)),),
            {"material": ("cotton",)},
        ),
        (
            {"material": ("leather",), "color": ("black",)},
            "Black is fine, but cotton would be better.",
            (
                SlotOperation("upsert", "color", ("black",)),
                SlotOperation("upsert", "material", ("cotton",)),
            ),
            {"material": ("leather", "cotton"), "color": ("black",)},
        ),
        (
            {"feature": ("waterproof",)},
            "Waterproof is useful, but lightweight would be better too.",
            (
                SlotOperation(
                    "upsert", "feature", ("waterproof", "lightweight")
                ),
            ),
            {"feature": ("waterproof", "lightweight")},
        ),
        (
            {"brand": "Nike"},
            "Nike is fine, but Adidas would be better.",
            (SlotOperation("replace", "brand", ("Adidas",)),),
            {"brand": "Adidas"},
        ),
    ],
)
def test_better_only_replaces_with_same_slot_comparison_evidence(
    initial_slots: ExplicitSlots,
    message: str,
    expected_operations: tuple[SlotOperation, ...],
    expected_slots: ExplicitSlots,
) -> None:
    _, operations, slots, _, _ = plan_and_apply(initial_slots, message)

    assert operations == expected_operations
    assert slots == expected_slots


@pytest.mark.parametrize(
    ("message", "expected_operations", "expected_material"),
    [
        (
            "Forget leather, just cotton.",
            (SlotOperation("replace", "material", ("cotton",)),),
            ("cotton",),
        ),
        (
            "I just also want cotton.",
            (SlotOperation("upsert", "material", ("cotton",)),),
            ("leather", "cotton"),
        ),
        (
            "Just cotton.",
            (SlotOperation("upsert", "material", ("cotton",)),),
            ("leather", "cotton"),
        ),
    ],
)
def test_just_requires_same_slot_rejection_to_signal_replacement(
    message: str,
    expected_operations: tuple[SlotOperation, ...],
    expected_material: tuple[str, ...],
) -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"material": ("leather",)}, message
    )

    assert operations == expected_operations
    assert slots["material"] == expected_material


def test_just_without_an_extracted_slot_is_not_a_state_operation() -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"material": ("leather",)}, "Just something comfortable."
    )

    assert operations == ()
    assert slots == {"material": ("leather",)}


def test_replacement_pairing_does_not_cross_slot_boundaries() -> None:
    _, operations, slots, _, _ = plan_and_apply(
        {"material": ("leather",), "color": ("black",)},
        "Not leather, blue would be nice.",
    )

    assert operations == (
        SlotOperation("upsert", "color", ("blue",)),
        SlotOperation("delete_value", "material", ("leather",)),
    )
    assert slots == {"color": ("black", "blue")}
