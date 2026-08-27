"""Focused tests for Owner B's B3 single-utterance extractor."""

import pytest

from extract import build_attribute_gazetteer, extract_slots
from state import ClarificationAttribute
from utils import FIXTURE_CATALOG


GAZETTEER = build_attribute_gazetteer(FIXTURE_CATALOG)


def slots(message: str) -> dict:
    return extract_slots(message, GAZETTEER).slots


def test_extracts_department_and_most_specific_category() -> None:
    result = extract_slots("I'm looking for men's running shoes.", GAZETTEER)

    assert result.slots == {"department": "Men", "category": "Running"}
    assert result.residual_scenario is None


def test_evaluator_looking_for_template_prefers_catalogue_category() -> None:
    assert slots("I'm looking for Running.") == {"category": "Running"}


def test_extracts_catalogue_brand_on_word_boundaries() -> None:
    assert slots("I want Saucony running shoes.") == {
        "category": "Running",
        "brand": "Saucony",
    }
    assert slots("The word sauconyesque is not a brand request.") == {}


def test_existing_message_only_call_shape_uses_shared_catalogue() -> None:
    assert extract_slots("I want Saucony shoes.").slots == {
        "category": "Shoes",
        "brand": "Saucony",
    }


def test_extracts_controlled_color_and_material() -> None:
    assert slots("I want a blue cotton shirt.") == {
        "color": ("blue",),
        "material": ("cotton",),
    }


def test_extracts_use_case_only_with_supporting_context() -> None:
    assert slots("I need something for hiking.") == {"use_case": ("hiking",)}
    assert slots("Work is a generic word here.") == {}


def test_running_product_phrase_is_not_duplicated_as_use_case() -> None:
    assert slots("I want running shoes.") == {"category": "Running"}


def test_extracts_small_high_precision_feature_vocabulary() -> None:
    assert slots("I need something waterproof and lightweight for hiking.") == {
        "feature": ("waterproof", "lightweight"),
        "use_case": ("hiking",),
    }


def test_extracts_explicit_style_phrase_without_gazetteer_entry() -> None:
    assert slots("I'd prefer a relaxed fit.") == {"style": ("relaxed fit",)}


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Keep it under $100.", {"price_max": "100"}),
        ("Keep it at most $100.", {"price_max": "100"}),
        ("My limit is below 80.", {"price_max": "80"}),
        ("I can spend over $50.", {"price_min": "50"}),
        ("I can spend at least $50.", {"price_min": "50"}),
        (
            "My budget is between $40 and $90.",
            {"price_min": "40", "price_max": "90"},
        ),
        ("My range is $40 to $90.", {"price_min": "40", "price_max": "90"}),
    ],
)
def test_extracts_explicit_budget_bounds(message: str, expected: dict) -> None:
    assert slots(message) == expected


def test_size_requires_an_explicit_size_construction() -> None:
    assert slots("I need shoes in size 10.") == {
        "category": "Shoes",
        "size": ("10",),
    }
    assert slots("I want a Saucony Cohesion 10.") == {"brand": "Saucony"}


def test_evaluator_reply_with_one_arbitrary_value_is_explicit_feature() -> None:
    result = extract_slots(
        "For that, what matters is: waterproof.", GAZETTEER
    )

    assert result.slots == {"feature": ("waterproof",)}
    assert result.residual_scenario is None
    assert result.observations[0].source == "requirement_phrase"


def test_evaluator_reply_classifies_two_controlled_values() -> None:
    assert slots("For that, what matters is: cotton; color: black.") == {
        "color": ("black",),
        "material": ("cotton",),
    }


def test_evaluator_reply_supports_catalogue_detail_labels() -> None:
    assert slots("For that, what matters is: Color: Auburn; Material: Mesh.") == {
        "color": ("auburn",),
        "material": ("mesh",),
    }


def test_evaluator_approximate_budget_is_a_target_not_an_upper_bound() -> None:
    assert slots("For that, what matters is: budget around $64.99.") == {
        "price_target": "64.99"
    }


@pytest.mark.parametrize(
    ("message", "target"),
    [
        ("Something around $100.", "100"),
        ("Something about $80.", "80"),
        ("Something roughly $50.", "50"),
    ],
)
def test_approximate_currency_is_preserved_as_price_target(
    message: str, target: str
) -> None:
    assert slots(message) == {"price_target": target}


@pytest.mark.parametrize(
    ("requested_attribute", "reply", "expected"),
    [
        ("style", "classic", {"style": ("classic",)}),
        ("material", "cotton", {"material": ("cotton",)}),
        (
            "feature",
            "waterproof; breathable",
            {"feature": ("waterproof", "breathable")},
        ),
        ("use_case", "hiking", {"use_case": ("hiking",)}),
        ("size", "large", {"size": ("large",)}),
    ],
)
def test_requested_attribute_authoritatively_classifies_evaluator_reply(
    requested_attribute: ClarificationAttribute, reply: str, expected: dict
) -> None:
    result = extract_slots(
        f"For that, what matters is: {reply}.",
        GAZETTEER,
        requested_attribute=requested_attribute,
    )

    assert result.slots == expected
    assert all(
        observation.source == "clarification_context"
        for observation in result.observations
    )


def test_other_uses_content_fallback_instead_of_becoming_a_slot() -> None:
    result = extract_slots(
        "For that, what matters is: waterproof.",
        GAZETTEER,
        requested_attribute="other",
    )

    assert result.slots == {"feature": ("waterproof",)}
    assert "other" not in result.slots
    assert result.observations[0].source == "requirement_phrase"


def test_requested_budget_distinguishes_target_from_strict_bound() -> None:
    target = extract_slots(
        "For that, what matters is: $100.",
        GAZETTEER,
        requested_attribute="budget",
    )
    bound = extract_slots(
        "For that, what matters is: under $100.",
        GAZETTEER,
        requested_attribute="budget",
    )

    assert target.slots == {"price_target": "100"}
    assert bound.slots == {"price_max": "100"}


def test_requested_attribute_is_ignored_outside_evaluator_reply_shape() -> None:
    result = extract_slots(
        "Classic.", GAZETTEER, requested_attribute="style"
    )

    assert result.slots == {}


def test_no_requested_attribute_keeps_conservative_fallback() -> None:
    assert slots("For that, what matters is: classic.") == {
        "feature": ("classic",)
    }


def test_observations_retain_order_needed_for_later_negation_analysis() -> None:
    color = extract_slots("Not black, blue instead.", GAZETTEER)
    material = extract_slots(
        "I don't want leather, cotton would be better.", GAZETTEER
    )

    assert color.slots == {"color": ("black", "blue")}
    assert [(item.slot, item.value) for item in color.observations] == [
        ("color", "black"),
        ("color", "blue"),
    ]
    assert material.slots == {"material": ("leather", "cotton")}
    assert [(item.slot, item.value) for item in material.observations] == [
        ("material", "leather"),
        ("material", "cotton"),
    ]


def test_evaluator_reply_preserves_order_for_two_style_values() -> None:
    assert slots("For that, what matters is: relaxed fit; long sleeve.") == {
        "style": ("relaxed fit", "long sleeve")
    }


def test_evaluator_reply_preserves_order_for_two_feature_values() -> None:
    assert slots("For that, what matters is: waterproof; breathable.") == {
        "feature": ("waterproof", "breathable")
    }


def test_vague_scenario_stays_unstructured() -> None:
    result = extract_slots("Something for a beach trip.", GAZETTEER)

    assert result.slots == {}
    assert result.residual_scenario == "Something for a beach trip."


def test_honeymoon_does_not_infer_product_category_or_feature() -> None:
    result = extract_slots(
        "I want something nice for my honeymoon.", GAZETTEER
    )

    assert result.slots == {}
    assert result.residual_scenario == "I want something nice for my honeymoon."


def test_longest_overlapping_catalogue_phrase_wins() -> None:
    gazetteer = build_attribute_gazetteer(
        [
            {
                "categories": ["Root", "Men", "Jackets", "Jackets & Coats"],
                "store": "Acme",
                "details": {},
            }
        ]
    )

    assert extract_slots("Show me jackets and coats.", gazetteer).slots == {
        "category": "Jackets & Coats"
    }


def test_message_with_nothing_extractable_returns_empty_result() -> None:
    result = extract_slots("Hello there.", GAZETTEER)

    assert result.slots == {}
    assert result.residual_scenario is None
    assert result.observations == ()
