"""Contract tests for Qikun's B1 session-state schema."""

from state import (
    CLARIFICATION_ATTRIBUTES,
    MULTI_VALUE_SLOT_KEYS,
    SLOT_KEYS,
    init_state,
)


def test_slot_schema_is_the_agreed_fixed_key_set() -> None:
    assert SLOT_KEYS == (
        "department",
        "category",
        "brand",
        "price_min",
        "price_max",
        "price_target",
        "color",
        "material",
        "style",
        "size",
        "feature",
        "use_case",
    )


def test_clarification_vocabulary_is_distinct_from_internal_slots() -> None:
    assert CLARIFICATION_ATTRIBUTES == (
        "category",
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
        "other",
    )
    assert "department" in SLOT_KEYS
    assert "department" not in CLARIFICATION_ATTRIBUTES
    assert "price_max" in SLOT_KEYS
    assert "price_max" not in CLARIFICATION_ATTRIBUTES
    assert "budget" in CLARIFICATION_ATTRIBUTES
    assert "budget" not in SLOT_KEYS
    assert "other" in CLARIFICATION_ATTRIBUTES
    assert "other" not in SLOT_KEYS


def test_descriptive_slots_support_multiple_active_values() -> None:
    assert MULTI_VALUE_SLOT_KEYS == (
        "color",
        "material",
        "style",
        "size",
        "feature",
        "use_case",
    )

    state = init_state("multi-value")
    state.slots["feature"] = ("water resistant", "machine washable")
    state.slots["use_case"] = ("winter", "outdoor work")

    assert state.slots["feature"] == ("water resistant", "machine washable")
    assert state.slots["use_case"] == ("winter", "outdoor work")


def test_init_state_has_empty_conversation_state() -> None:
    state = init_state("session-1")

    assert state.session_id == "session-1"
    assert state.turn == 0
    assert state.slots == {}
    assert state.slot_override_flags == {}
    assert state.scenario_buffer == ""
    assert state.asked_attributes == set()
    assert state.pending_clarification is None
    assert state.override_reference_values == {}
    assert state.canonical_intent == ""
    assert state.canonical_vector is None
    assert state.track == "browse"


def test_profile_information_stays_outside_explicit_slots() -> None:
    user_profile = {
        "preference_tags": ["comfort", "warmth"],
        "rating_style": "usually positive",
    }

    state = init_state("session-2", user_profile)

    assert state.slots == {}
    assert state.scenario_buffer == ""
    assert state.profile_terms == ["warmth"]
    assert not hasattr(state, "user_profile")


def test_mutable_defaults_are_not_shared_between_sessions() -> None:
    first = init_state("first")
    second = init_state("second")

    first.slots["category"] = "running shoes"
    first.slot_override_flags["category"] = False
    first.asked_attributes.add("brand")
    first.pending_clarification = "brand"
    first.override_reference_values["feature"] = ("lightweight",)

    assert second.slots == {}
    assert second.slot_override_flags == {}
    assert second.asked_attributes == set()
    assert second.pending_clarification is None
    assert second.override_reference_values == {}
