"""Adversarial end-to-end contract tests across Owner B's B1-B8 pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

from extract import build_attribute_gazetteer, update_slots
from state import (
    ExplicitSlots,
    SessionState,
    Track,
    init_state,
    pick_track,
    reconstruct_canonical,
    set_pending_clarification,
)
from utils import FIXTURE_CATALOG


CATALOG = deepcopy(FIXTURE_CATALOG) + [
    {
        "parent_asin": "B00AUDITNIKE",
        "categories": ["Root", "Men", "Shoes", "Running"],
        "store": "Nike",
        "details": {},
    },
    {
        "parent_asin": "B00AUDITADIDAS",
        "categories": ["Root", "Men", "Shoes", "Running"],
        "store": "Adidas",
        "details": {},
    },
]
GAZETTEER = build_attribute_gazetteer(CATALOG)


@dataclass(frozen=True)
class Snapshot:
    slots: ExplicitSlots
    scenario: str
    pending: str | None
    canonical: str
    track: Track


def _embed(text: str) -> np.ndarray:
    """Small deterministic stand-in for Owner A's shared encoder interface."""
    return np.array([len(text), sum(text.encode("utf-8"))], dtype=np.float32)


def turn(state: SessionState, message: str) -> Snapshot:
    update_slots(state, message, GAZETTEER)
    reconstruct_canonical(state, _embed)
    pick_track(state)
    return Snapshot(
        slots=deepcopy(state.slots),
        scenario=state.scenario_buffer,
        pending=state.pending_clarification,
        canonical=state.canonical_intent,
        track=state.track,
    )


def test_case_1_ordinary_accumulation() -> None:
    state = init_state("ordinary")

    assert turn(state, "I'm looking for running shoes.") == Snapshot(
        {"category": "Running"},
        "",
        None,
        "category: Running",
        "buy",
    )
    assert turn(state, "Blue please.") == Snapshot(
        {"category": "Running", "color": ("blue",)},
        "",
        None,
        "category: Running; color: blue",
        "buy",
    )
    assert turn(state, "Waterproof would be useful too.") == Snapshot(
        {
            "category": "Running",
            "color": ("blue",),
            "feature": ("waterproof",),
        },
        "",
        None,
        "category: Running; color: blue; features: waterproof",
        "buy",
    )


def test_case_2_scalar_override_removes_stale_brand_everywhere() -> None:
    state = init_state("scalar-override")
    turn(state, "Nike running shoes.")

    snapshot = turn(state, "Actually Adidas.")

    assert snapshot.slots == {"category": "Running", "brand": "Adidas"}
    assert snapshot.canonical == "category: Running; brand: Adidas"
    assert "nike" not in snapshot.canonical.casefold()
    assert snapshot.track == "buy"


def test_case_3_multi_value_partial_deletion_preserves_sibling() -> None:
    state = init_state("partial-delete")
    turn(state, "Something waterproof and lightweight.")

    snapshot = turn(state, "I don't need waterproof anymore.")

    assert snapshot.slots == {"feature": ("lightweight",)}
    assert snapshot.canonical == "features: lightweight"
    assert "waterproof" not in snapshot.canonical
    assert snapshot.track == "browse"


def test_case_4_multi_value_replacement_removes_old_material() -> None:
    state = init_state("multi-replace")
    turn(state, "Leather would be good.")

    snapshot = turn(state, "Forget leather, cotton would be better.")

    assert snapshot.slots == {"material": ("cotton",)}
    assert snapshot.canonical == "material: cotton"
    assert "leather" not in snapshot.canonical


def test_case_5_scenario_accumulates_only_bounded_detail() -> None:
    state = init_state("scenario-detail")

    first = turn(state, "Something for a beach holiday.")
    second = turn(state, "Preferably blue.")
    third = turn(state, "With lots of walking.")

    assert first == Snapshot(
        {},
        "Something for a beach holiday.",
        None,
        "Something for a beach holiday.",
        "browse",
    )
    assert second.scenario == "Something for a beach holiday."
    assert second.slots == {"color": ("blue",)}
    assert third.scenario == (
        "Something for a beach holiday — With lots of walking."
    )
    assert third.slots == {"color": ("blue",)}
    assert third.canonical == (
        "color: blue; Something for a beach holiday — With lots of walking."
    )


def test_case_6_scenario_override_by_use_case_has_no_duplication() -> None:
    state = init_state("scenario-override")
    turn(state, "Something for a beach holiday.")

    snapshot = turn(state, "Actually this is for hiking instead.")

    assert snapshot.slots == {"use_case": ("hiking",)}
    assert snapshot.scenario == ""
    assert snapshot.canonical == "use case: hiking"
    assert snapshot.canonical.casefold().count("hiking") == 1
    assert "beach" not in snapshot.canonical.casefold()
    assert snapshot.track == "browse"


def test_case_7_structured_deletion_and_scenario_override_are_independent() -> None:
    state = init_state("mixed-override")
    turn(state, "Black leather shoes for a beach holiday.")

    snapshot = turn(
        state,
        "No leather anymore; this is for a winter work trip instead.",
    )

    assert snapshot.slots == {
        "category": "Shoes",
        "color": ("black",),
    }
    assert snapshot.scenario == "for a winter work trip"
    assert snapshot.canonical == (
        "category: Shoes; color: black; for a winter work trip"
    )
    assert "leather" not in snapshot.canonical.casefold()
    assert "beach" not in snapshot.canonical.casefold()
    assert snapshot.track == "browse"


def test_case_8_pending_clarification_is_one_shot_but_history_survives() -> None:
    state = init_state("clarification")
    state.asked_attributes.add("style")
    set_pending_clarification(state, "style")

    first = turn(state, "For that, what matters is: classic.")
    second = turn(state, "For that, what matters is: cotton.")

    assert first.slots == {"style": ("classic",)}
    assert first.pending is None
    assert state.asked_attributes == {"style"}
    assert second.slots == {
        "style": ("classic",),
        "material": ("cotton",),
    }
    assert "cotton" not in second.slots["style"]
    assert second.pending is None


def test_case_9_other_clarification_uses_content_and_clears_pending() -> None:
    state = init_state("other")
    state.asked_attributes.add("other")
    set_pending_clarification(state, "other")

    snapshot = turn(state, "For that, what matters is: waterproof.")

    assert snapshot.slots == {"feature": ("waterproof",)}
    assert "other" not in snapshot.slots
    assert snapshot.pending is None
    assert state.asked_attributes == {"other"}


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (("around $100", "actually under $80"), {"price_max": "80"}),
        (("under $100", "actually around $150"), {"price_target": "150"}),
        (
            ("$50 to $100", "actually at least $120"),
            {"price_min": "120"},
        ),
    ],
)
def test_case_10_price_corrections_do_not_leave_stale_semantics(
    messages: tuple[str, str], expected: ExplicitSlots
) -> None:
    state = init_state("price-correction")
    turn(state, messages[0])

    snapshot = turn(state, messages[1])

    assert snapshot.slots == expected


def test_case_10_compatible_target_and_upper_bound_accumulate() -> None:
    state = init_state("target-plus-maximum")
    turn(state, "around $100")

    snapshot = turn(state, "Also, definitely under $120.")

    assert snapshot.slots == {
        "price_max": "120",
        "price_target": "100",
    }
    assert snapshot.canonical == "under $120; around $100"


def test_case_10_explicit_additive_word_blocks_actually_revision() -> None:
    state = init_state("actually-additive-price")
    turn(state, "around $100")

    snapshot = turn(state, "Actually, I also need it under $120.")

    assert snapshot.slots == {
        "price_max": "120",
        "price_target": "100",
    }


def test_case_10_lower_and_upper_bounds_accumulate_into_range() -> None:
    state = init_state("range-accumulation")
    turn(state, "at least $50")

    snapshot = turn(state, "Also under $100.")

    assert snapshot.slots == {"price_min": "50", "price_max": "100"}
    assert snapshot.canonical == "between $50 and $100"


def test_case_10_target_and_range_can_coexist_from_one_utterance() -> None:
    state = init_state("target-plus-range")

    snapshot = turn(
        state,
        "I'd like around $80, somewhere between $60 and $100.",
    )

    assert snapshot.slots == {
        "price_min": "60",
        "price_max": "100",
        "price_target": "80",
    }
    assert snapshot.canonical == "between $60 and $100; around $80"


def test_case_10_malformed_single_utterance_range_is_not_silently_repaired() -> None:
    state = init_state("malformed-range")

    snapshot = turn(state, "$100 to $50")

    assert snapshot.slots == {"price_min": "100", "price_max": "50"}
    assert snapshot.canonical == "between $100 and $50"


def test_case_10_price_revision_preserves_every_non_price_domain() -> None:
    state = init_state("price-isolation")
    state.slots = {
        "category": "Running",
        "brand": "Nike",
        "feature": ("waterproof",),
        "price_target": "100",
    }
    state.scenario_buffer = "for a beach holiday"

    snapshot = turn(state, "Actually under $80.")

    assert snapshot.slots == {
        "category": "Running",
        "brand": "Nike",
        "feature": ("waterproof",),
        "price_max": "80",
    }
    assert snapshot.scenario == "for a beach holiday"


def test_case_10_repeated_same_budget_is_state_no_op() -> None:
    state = init_state("repeat-price")
    first = turn(state, "under $100")
    flags_before = deepcopy(state.slot_override_flags)

    second = turn(state, "under $100")

    assert second == first
    assert state.slot_override_flags == flags_before == {"price_max": False}


def test_case_10_same_price_representation_can_be_corrected() -> None:
    state = init_state("same-price-key")
    turn(state, "under $100")

    snapshot = turn(state, "Actually under $80.")

    assert snapshot.slots == {"price_max": "80"}
    assert snapshot.canonical == "under $80"


def test_case_10_evaluator_override_removes_old_budget_only() -> None:
    state = init_state("budget-override")
    turn(state, "I'm looking for Running. budget at most $100")
    turn(state, "Maybe around $80 too.")

    snapshot = turn(
        state,
        "Actually, ignore my earlier preference. What I need is: leather.",
    )

    assert snapshot.slots == {
        "category": "Running",
        "price_target": "80",
        "material": ("leather",),
    }
    assert state.override_reference_values == {}


def test_case_10_evaluator_fallback_preserves_later_price_information() -> None:
    state = init_state("budget-fallback")
    turn(state, "I'm looking for Running. budget at most $100")
    turn(state, "Maybe around $80 too.")

    snapshot = turn(state, "Actually, please ignore my earlier preference.")

    assert snapshot.slots == {
        "category": "Running",
        "price_target": "80",
    }
    assert state.override_reference_values == {}


def test_case_10_evaluator_fallback_preserves_later_same_key_price() -> None:
    state = init_state("budget-fallback-same-key")
    turn(state, "I'm looking for Running. budget at most $100")
    turn(state, "Also under $120.")

    snapshot = turn(state, "Actually, please ignore my earlier preference.")

    assert snapshot.slots == {
        "category": "Running",
        "price_max": "120",
    }
    assert state.override_reference_values == {}


@pytest.mark.parametrize(
    ("old_preference", "old_slot", "old_value"),
    [
        ("color: black", "color", ("black",)),
        ("leather", "material", ("leather",)),
        ("lightweight", "feature", ("lightweight",)),
        ("brand: Nike", "brand", "Nike"),
        ("style: relaxed fit", "style", ("relaxed fit",)),
        ("size: large", "size", ("large",)),
        ("use_case: hiking", "use_case", ("hiking",)),
        ("under $100", "price_max", "100"),
    ],
)
def test_case_11_evaluator_fallback_invalidates_only_referenced_preference(
    old_preference: str,
    old_slot: str,
    old_value: str | tuple[str, ...],
) -> None:
    state = init_state(f"fallback-{old_slot}")
    turn(state, f"I'm looking for Running. {old_preference}")
    turn(state, "I also need it waterproof and blue.")

    snapshot = turn(state, "Actually, please ignore my earlier preference.")

    assert snapshot.slots["category"] == "Running"
    assert snapshot.slots["feature"] == ("waterproof",)
    assert snapshot.slots["color"] == ("blue",)
    assert old_value not in snapshot.slots.values()
    assert state.override_reference_values == {}
    assert snapshot.track == "buy"


def test_case_11_full_evaluator_override_preserves_later_unrelated_state() -> None:
    state = init_state("official-full-override")
    turn(state, "I'm looking for Running. color: black")
    turn(state, "I also need it waterproof and leather.")

    snapshot = turn(
        state,
        "Actually, ignore my earlier preference. What I need is: blue.",
    )

    assert snapshot.slots == {
        "category": "Running",
        "color": ("blue",),
        "material": ("leather",),
        "feature": ("waterproof",),
    }
    assert "black" not in snapshot.canonical.casefold()
    assert state.override_reference_values == {}


@pytest.mark.parametrize(
    "message",
    [
        "I'm not sure about the color.",
        "I don't mind Nike.",
        "Nothing too flashy.",
    ],
)
def test_case_12_false_negation_does_not_delete_active_constraints(
    message: str,
) -> None:
    state = init_state("false-negation")
    state.slots = {"brand": "Nike", "color": ("black",)}

    snapshot = turn(state, message)

    assert snapshot.slots == {"brand": "Nike", "color": ("black",)}


def test_case_12_indifference_does_not_create_a_brand_constraint() -> None:
    state = init_state("indifference")

    snapshot = turn(state, "I don't mind Nike.")

    assert snapshot.slots == {}


def test_case_12_additive_discourse_does_not_replace_or_delete() -> None:
    state = init_state("additive-discourse")
    state.slots = {"color": ("black",), "feature": ("lightweight",)}

    first = turn(state, "Actually I also need waterproof.")
    second = turn(state, "Black is fine, but breathable would be useful too.")

    assert first.slots == {
        "color": ("black",),
        "feature": ("lightweight", "waterproof"),
    }
    assert second.slots == {
        "color": ("black",),
        "feature": ("lightweight", "waterproof", "breathable"),
    }


def test_case_13_route_downgrades_after_specific_category_deletion() -> None:
    state = init_state("route-downgrade")
    first = turn(state, "Blue running shoes.")
    second = turn(state, "I don't care about category anymore.")

    assert first.track == "buy"
    assert second.slots == {"color": ("blue",)}
    assert second.canonical == "color: blue"
    assert second.track == "browse"


def test_case_14_metadata_cannot_contaminate_canonical_or_route() -> None:
    first = init_state("metadata-a", {"preference_tags": ["weather"]})
    second = init_state("metadata-b")
    for state in (first, second):
        state.slots = {"category": "Running", "color": ("blue",)}
        state.scenario_buffer = "for a beach holiday"

    first.slot_override_flags = {"color": True}
    first.override_reference_values = {"color": ("black",)}
    first.asked_attributes = {"brand", "style"}
    first.pending_clarification = "material"
    first.canonical_intent = "stale intent"
    first.canonical_vector = np.array([999], dtype=np.float32)
    first.track = "browse"

    second.canonical_intent = "different stale intent"
    second.canonical_vector = np.array([-1], dtype=np.float32)
    second.track = "buy"

    reconstruct_canonical(first, _embed)
    reconstruct_canonical(second, _embed)

    assert first.canonical_intent == second.canonical_intent
    np.testing.assert_array_equal(first.canonical_vector, second.canonical_vector)
    assert pick_track(first) == pick_track(second) == "buy"


def test_case_15_interleaved_sessions_are_fully_isolated() -> None:
    gazetteer_before = deepcopy(GAZETTEER)
    first = init_state("isolation-a")
    second = init_state("isolation-b")

    first_turn = turn(first, "Something for a beach holiday.")
    second_turn = turn(second, "I'm looking for Running. color: black")

    first.asked_attributes.add("style")
    set_pending_clarification(first, "style")
    second.asked_attributes.add("other")
    set_pending_clarification(second, "other")

    first_reply = turn(first, "For that, what matters is: classic.")
    second_reply = turn(second, "For that, what matters is: waterproof.")
    second_override = turn(
        second, "Actually, please ignore my earlier preference."
    )

    assert first_turn.slots == {}
    assert first_reply.slots == {"style": ("classic",)}
    assert first_reply.scenario == "Something for a beach holiday."
    assert first.asked_attributes == {"style"}
    assert first.override_reference_values == {}

    assert second_turn.slots == {
        "category": "Running",
        "color": ("black",),
    }
    assert second_reply.slots == {
        "category": "Running",
        "color": ("black",),
        "feature": ("waterproof",),
    }
    assert second_override.slots == {
        "category": "Running",
        "feature": ("waterproof",),
    }
    assert second.asked_attributes == {"other"}
    assert second.override_reference_values == {}
    assert first.pending_clarification is None
    assert second.pending_clarification is None
    assert GAZETTEER == gazetteer_before
