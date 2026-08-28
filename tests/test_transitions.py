"""Table-driven tests for Owner B's B4 state transition engine."""

from copy import deepcopy

import pytest

from extract import (
    USE_LLM_EXTRACTION,
    ExtractionResult,
    SlotObservation,
    SlotOperation,
    apply_extraction_result,
    apply_slot_operation,
    apply_slot_operations,
    build_attribute_gazetteer,
    extract_slots_llm,
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


# --------------------------------------------------------------------------
# B9: optional LLM extraction path — off by default, falls back to B3.
# extract_slots_llm() is a deliberate stub (no LLM access provided, §1.2);
# these tests protect the fallback wiring, not real LLM extraction.
# --------------------------------------------------------------------------

def test_use_llm_extraction_defaults_to_off():
    assert USE_LLM_EXTRACTION is False


def test_extract_slots_llm_stub_always_returns_none():
    gazetteer = build_attribute_gazetteer(FIXTURE_CATALOG)
    assert extract_slots_llm("I want a jacket", gazetteer) is None


def test_update_slots_default_never_calls_llm_path(monkeypatch):
    """use_llm_extraction defaults to USE_LLM_EXTRACTION (False) — the LLM
    path must not even be attempted on an ordinary call."""
    called = []
    monkeypatch.setattr(
        "extract.extract_slots_llm", lambda *a, **k: called.append(1) or None
    )
    state = init_state("b9-default-off")
    gazetteer = build_attribute_gazetteer(FIXTURE_CATALOG)
    update_slots(state, "I want a jacket", gazetteer)
    assert called == []


def test_update_slots_llm_flag_on_falls_back_to_b3_on_stub_none(monkeypatch):
    """With the flag on, extract_slots_llm() is tried first; since the stub
    always returns None, the B3 result must still land in state — the
    fallback contract (§8.2 step B9's definition of done) holds even
    though no real LLM call happens."""
    calls = []
    original = extract_slots_llm

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr("extract.extract_slots_llm", spy)
    state = init_state("b9-flag-on")
    gazetteer = build_attribute_gazetteer(FIXTURE_CATALOG)
    update_slots(state, "I want a red jacket", gazetteer, use_llm_extraction=True)
    assert calls == [1]  # the LLM path was actually attempted
    assert state.slots.get("color") == ("red",)  # and B3's fallback result landed


def test_update_slots_llm_flag_on_uses_llm_result_when_available(monkeypatch):
    """If extract_slots_llm() succeeds (non-None), update_slots() must use
    that result directly rather than also running B3."""
    llm_result = ExtractionResult(
        slots={"color": "blue"},
        residual_scenario=None,
        observations=(SlotObservation("color", "blue", "gazetteer"),),
    )
    monkeypatch.setattr("extract.extract_slots_llm", lambda *a, **k: llm_result)
    state = init_state("b9-llm-succeeds")
    gazetteer = build_attribute_gazetteer(FIXTURE_CATALOG)
    update_slots(state, "anything at all", gazetteer, use_llm_extraction=True)
    assert state.slots.get("color") == ("blue",)  # from the LLM stub, not B3's parse of "anything at all"
