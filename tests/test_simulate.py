"""Tests for Owner D's user simulator and session driver (§8.4 D3-D6)."""

from __future__ import annotations

import pytest

import simulate
from extract import build_attribute_gazetteer, update_slots
from state import init_state
from telemetry import telemetry_path_ctx
from utils import FIXTURE_CATALOG

GAZ = build_attribute_gazetteer(FIXTURE_CATALOG)
CATALOG_INDEX = simulate.build_catalog_index(FIXTURE_CATALOG)
FIXTURE_ASINS = [row["parent_asin"] for row in FIXTURE_CATALOG]


def make_session(asin: str, scenario: str) -> dict:
    session = {
        "sample_id": f"{asin}-{scenario}",
        "scenario_type": scenario,
        "ground_truth": {"parent_asin": asin},
        "user_profile": {},
    }
    return simulate.attach_target_record(session, CATALOG_INDEX)


def slots_after_scripted_turns(session: dict, n: int) -> tuple[object, dict]:
    """Feed `n` scripted simulate_turn outputs through update_slots."""
    state = init_state(session["sample_id"])
    history: list[str] = []
    slots_after_first: dict = {}
    for i in range(n):
        message = simulate.simulate_turn(session, history)
        update_slots(state, message, GAZ)
        history.append(message)
        if i == 0:
            slots_after_first = dict(state.slots)
    return state, slots_after_first, history


# --------------------------------------------------------------------------
# D3 — facet extractor
# --------------------------------------------------------------------------
def test_extract_target_facets_reads_categories_and_details_not_title() -> None:
    facets = simulate.extract_target_facets(FIXTURE_CATALOG[0])
    assert facets["department"] == "Men"
    assert facets["category"] == "Jackets & Coats"
    assert facets["color"] == "Auburn"
    assert facets["material"] == "Polyester"
    assert facets["brand"] == "London Fog"
    assert facets["price"] == pytest.approx(64.99)
    # nothing lifted verbatim from the title
    title_words = FIXTURE_CATALOG[0]["title"].lower().split()
    assert "auburn" in title_words  # the title *does* contain it...
    assert facets["category_noun"] == "jacket"  # ...but the noun came from categories


def test_extract_target_facets_singular_category_noun() -> None:
    assert simulate.extract_target_facets(FIXTURE_CATALOG[1])["category_noun"] == "running shoe"
    assert simulate.extract_target_facets(FIXTURE_CATALOG[2])["category_noun"].endswith("up")


def test_extract_target_facets_tolerates_missing_details() -> None:
    facets = simulate.extract_target_facets(FIXTURE_CATALOG[2])
    assert "color" not in facets and "material" not in facets
    assert facets["department"] == "Women"
    assert facets["brand"] == "SunDaze"


# --------------------------------------------------------------------------
# D4 — per-scenario release policy
# --------------------------------------------------------------------------
@pytest.mark.parametrize("asin", FIXTURE_ASINS)
def test_buying_opens_with_at_least_two_extractable_attributes(asin: str) -> None:
    state, first, _ = slots_after_scripted_turns(make_session(asin, "buying"), 1)
    assert len(state.slots) >= 2, first


@pytest.mark.parametrize("asin", FIXTURE_ASINS)
def test_browsing_opener_carries_no_extractable_attribute(asin: str) -> None:
    state, first, _ = slots_after_scripted_turns(make_session(asin, "browsing"), 1)
    assert state.slots == {}


@pytest.mark.parametrize("asin", FIXTURE_ASINS)
def test_boundary_withholds_attributes_across_turns(asin: str) -> None:
    state, _, _ = slots_after_scripted_turns(make_session(asin, "boundary"), 4)
    assert state.slots == {}


@pytest.mark.parametrize("asin", FIXTURE_ASINS)
def test_every_intent_override_session_overwrites_or_deletes_a_slot(asin: str) -> None:
    """The required guarantee: no intent_override session ships without a pivot.

    A hallucination in this path leaves 15% of the score (all `hard`)
    unrepresented in the training corpus (§8.4 D4).
    """
    session = make_session(asin, "intent_override")
    state, slots_after_first, history = slots_after_scripted_turns(session, 4)

    assert simulate.override_contradiction_shipped(history), history
    removed = [key for key in slots_after_first if key not in state.slots]
    overwritten = [key for key, flag in state.slot_override_flags.items() if flag]
    assert removed or overwritten, (slots_after_first, state.slots, state.slot_override_flags)


def test_intent_override_without_colour_still_contradicts() -> None:
    """FIXTURE_CATALOG[2] has empty `details` — the public_0002/public_0003 case."""
    session = make_session(FIXTURE_ASINS[2], "intent_override")
    state, slots_after_first, history = slots_after_scripted_turns(session, 4)

    assert any(h.startswith("not ") and "instead" in h for h in history), history
    removed = [key for key in slots_after_first if key not in state.slots]
    overwritten = [key for key, flag in state.slot_override_flags.items() if flag]
    assert removed or overwritten


def test_simulate_turn_is_deterministic() -> None:
    session = make_session(FIXTURE_ASINS[0], "browsing")
    history = ["I need something for an upcoming trip"]
    assert simulate.simulate_turn(session, history) == simulate.simulate_turn(session, history)


def test_simulate_turn_without_target_record_degrades_gracefully() -> None:
    bare = {"sample_id": "x", "scenario_type": "buying", "ground_truth": {"parent_asin": "ZZZ"}}
    assert isinstance(simulate.simulate_turn(bare, []), str)


# --------------------------------------------------------------------------
# D5 — clarification answering
# --------------------------------------------------------------------------
def test_answer_clarification_returns_record_value_when_present() -> None:
    session = make_session(FIXTURE_ASINS[0], "buying")
    assert simulate.answer_clarification(session, "color") == "auburn"
    assert simulate.answer_clarification(session, "material") == "polyester"
    assert simulate.answer_clarification(session, "budget") == "around $65"


def test_answer_clarification_returns_no_preference_when_absent() -> None:
    session = make_session(FIXTURE_ASINS[2], "boundary")
    assert simulate.answer_clarification(session, "color") == simulate.NO_PREFERENCE
    assert simulate.answer_clarification(session, "size") == simulate.NO_PREFERENCE
    assert simulate.answer_clarification(session, "budget") == simulate.NO_PREFERENCE


def test_answer_clarification_unknown_attribute_is_no_preference() -> None:
    session = make_session(FIXTURE_ASINS[0], "buying")
    assert simulate.answer_clarification(session, "other") == simulate.NO_PREFERENCE


# --------------------------------------------------------------------------
# D6 — instrumented session driver
# --------------------------------------------------------------------------
def _agent():
    from agent import Agent

    return Agent("data/nonexistent-catalog.jsonl")  # falls back to FIXTURE_CATALOG


@pytest.mark.parametrize("scenario", ["buying", "browsing", "intent_override", "boundary"])
def test_run_session_produces_a_transcript(scenario: str, tmp_path) -> None:
    session = {
        "sample_id": f"s-{scenario}",
        "scenario_type": scenario,
        "ground_truth": {"parent_asin": FIXTURE_ASINS[0]},
        "user_profile": {},
    }
    # run_session() drives Agent.respond(), which calls log_turn() with no
    # explicit path — without this redirect it silently appends to the
    # real data/telemetry.jsonl (DEFAULT_TELEMETRY_PATH) on every test run.
    with telemetry_path_ctx(str(tmp_path / "telemetry.jsonl")):
        result = simulate.run_session(_agent(), session, CATALOG_INDEX, max_turns=6)
    assert 1 <= result["turns"] <= 6
    assert all(step["recommended"] for step in result["transcript"])


def test_run_session_never_puts_the_target_asin_in_a_user_message(tmp_path) -> None:
    session = {
        "sample_id": "leak-check",
        "scenario_type": "buying",
        "ground_truth": {"parent_asin": FIXTURE_ASINS[1]},
        "user_profile": {},
    }
    with telemetry_path_ctx(str(tmp_path / "telemetry.jsonl")):
        result = simulate.run_session(_agent(), session, CATALOG_INDEX, max_turns=6)
    assert all(FIXTURE_ASINS[1] not in step["user"] for step in result["transcript"])


def test_run_session_intent_override_transcript_contains_a_contradiction(tmp_path) -> None:
    session = {
        "sample_id": "override-transcript",
        "scenario_type": "intent_override",
        "ground_truth": {"parent_asin": FIXTURE_ASINS[2]},
        "user_profile": {},
    }
    with telemetry_path_ctx(str(tmp_path / "telemetry.jsonl")):
        result = simulate.run_session(_agent(), session, CATALOG_INDEX, max_turns=8)
    users = [step["user"] for step in result["transcript"]]
    assert any(u.startswith("not ") and "instead" in u for u in users), users
