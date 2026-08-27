"""Contract tests for Owner B's deterministic B6 scenario transitions."""

from copy import deepcopy

import pytest

from extract import (
    build_attribute_gazetteer,
    extract_slots,
    update_scenario_buffer,
    update_slots,
)
from state import init_state
from utils import FIXTURE_CATALOG


CATALOG = deepcopy(FIXTURE_CATALOG) + [
    {
        "categories": ["Root", "Men", "Shoes", "Running"],
        "store": "Nike",
        "details": {},
    }
]
GAZETTEER = build_attribute_gazetteer(CATALOG)


def next_scenario(current: str, message: str) -> str:
    extraction = extract_slots(message, GAZETTEER)
    return update_scenario_buffer(current, message, extraction)


def test_new_unstructured_scenario_is_set() -> None:
    assert next_scenario("", "Something for a beach holiday.") == (
        "Something for a beach holiday."
    )


def test_update_slots_wires_set_preserve_and_replace_transitions() -> None:
    state = init_state("scenario-integration")

    update_slots(state, "Something for a beach holiday.", GAZETTEER)
    assert state.slots == {}
    assert state.scenario_buffer == "Something for a beach holiday."

    update_slots(state, "Preferably blue.", GAZETTEER)
    assert state.slots == {"color": ("blue",)}
    assert state.scenario_buffer == "Something for a beach holiday."

    update_slots(
        state,
        "Actually this is for a business conference instead.",
        GAZETTEER,
    )
    assert state.slots == {"color": ("blue",)}
    assert state.scenario_buffer == "for a business conference"


@pytest.mark.parametrize(
    "message",
    [
        "Preferably blue.",
        "Under $100.",
        "Nike if possible.",
        "waterproof please",
    ],
)
def test_structured_detail_preserves_current_scenario(message: str) -> None:
    assert next_scenario("for a beach holiday", message) == (
        "for a beach holiday"
    )


def test_explicit_new_unstructured_scenario_replaces_current() -> None:
    assert next_scenario(
        "for a beach holiday",
        "Actually this is for a business conference instead.",
    ) == "for a business conference"


def test_structured_use_case_override_clears_old_scenario_without_duplication() -> None:
    message = "Actually this is for hiking instead."
    extraction = extract_slots(message, GAZETTEER)

    assert extraction.slots == {"use_case": ("hiking",)}
    assert update_scenario_buffer(
        "for a beach holiday", message, extraction
    ) == ""


def test_additive_detail_keeps_base_and_only_latest_detail() -> None:
    scenario = next_scenario("for a beach holiday", "with lots of walking")
    assert scenario == "for a beach holiday — with lots of walking"

    scenario = next_scenario(scenario, "while carrying a backpack")
    assert scenario == "for a beach holiday — while carrying a backpack"


@pytest.mark.parametrize(
    ("current", "message"),
    [
        ("for a beach holiday", "It's not for the beach anymore."),
        ("for my honeymoon", "Forget the honeymoon part."),
    ],
)
def test_explicit_matching_scenario_rejection_clears(
    current: str, message: str
) -> None:
    assert next_scenario(current, message) == ""


@pytest.mark.parametrize(
    "message",
    [
        "not sure about the color",
        "Forget the honeymoon part.",
        "It's not for work anymore.",
    ],
)
def test_weak_or_nonmatching_rejection_preserves_scenario(message: str) -> None:
    assert next_scenario("for a beach holiday", message) == (
        "for a beach holiday"
    )


def test_fully_structured_request_does_not_create_scenario() -> None:
    assert next_scenario("", "I want blue running shoes.") == ""


@pytest.mark.parametrize(
    "message",
    [
        "Actually, please ignore my earlier preference.",
        "Actually, ignore my earlier preference. What I need is: waterproof.",
    ],
)
def test_evaluator_override_control_text_preserves_scenario(message: str) -> None:
    assert next_scenario("for a beach holiday", message) == (
        "for a beach holiday"
    )


def test_mixed_b5_and_b6_transition_updates_each_state_domain() -> None:
    state = init_state("mixed")
    state.slots = {"material": ("leather",)}
    state.slot_override_flags = {"material": False}
    state.scenario_buffer = "for a beach holiday"

    update_slots(
        state,
        "No leather anymore; this is for a winter work trip instead.",
        GAZETTEER,
    )

    assert state.slots == {}
    assert state.scenario_buffer == "for a winter work trip"


def test_mixed_structured_use_case_and_b5_deletion_clears_stale_scenario() -> None:
    state = init_state("mixed-use-case")
    state.slots = {"material": ("leather",)}
    state.slot_override_flags = {"material": False}
    state.scenario_buffer = "for a beach holiday"

    update_slots(
        state,
        "No leather anymore; this is actually for hiking.",
        GAZETTEER,
    )

    assert state.slots == {"use_case": ("hiking",)}
    assert state.scenario_buffer == ""


def test_pure_helper_does_not_mutate_extraction() -> None:
    extraction = extract_slots("Something for my honeymoon.", GAZETTEER)
    before = deepcopy(extraction)

    update_scenario_buffer("", "Something for my honeymoon.", extraction)

    assert extraction == before
