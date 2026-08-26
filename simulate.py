"""User simulator: holds a hidden intent card, emits turns per scenario type.

Design doc §6.5.2 (Fallback: a self-built user simulator) and §7.2
interface contract `simulate_turn(session, history) -> str`.

Owner: Chellappan (Simulator and training corpus). §8.4, step D3 (facet extractor
over target records), step D4 (BLOCKING — the simulator itself), step D5
(clarification answering).

Sessions ship without dialogue (§2.5); this module exists only because
configuration C ("No simulator at all", §6.5.1) or B ("Single-turn
evaluation only") may hold. Under configuration A (a bundled simulator is
supplied), this module is unused. §6.5.1 D1 is BLOCKING and must resolve
that question before this module's real logic is written.

Everything below is a stub. Function bodies return fixture values only.
"""

from __future__ import annotations

# Per-scenario release policy (§6.5.2 table).
SCENARIO_POLICIES: dict[str, str] = {
    "buying": "Emit two or more concrete attributes on turn 1; add detail steadily.",
    "browsing": "Open with a scenario phrase carrying no extractable attribute; concede specifics only when asked.",
    "intent_override": "State an attribute the target does not have, then contradict it on a later turn with the true value.",
    "boundary": "Withhold most attributes; answer clarifications minimally.",
}


def extract_target_facets(target_record: dict) -> dict[str, str]:
    """Pull department, category, colour, material, and detail phrases from a target.

    Design doc §6.5.2: "Each session gives the target ASIN, and the
    catalogue gives that product's full record ... Turn content is drawn
    from the target's own attributes, released progressively rather than
    all at once." Owner Chellappan, §8.4 step D3.

    STUB: returns a fixture facet dict tagged with the target's ASIN,
    rather than parsing `target_record`'s categories/details fields.

    Args:
        target_record: The full catalogue row for the session's
            ground-truth target.

    Returns:
        A dict of facet name -> value, e.g. {"department": "Mens",
        "category": "Jackets & Coats", ...}.
    """
    asin = target_record.get("parent_asin", "UNKNOWN")
    return {"department": f"[STUB dept for {asin}]", "category": f"[STUB category for {asin}]"}


def simulate_turn(session: dict, history: list[str]) -> str:
    """Produce the next user utterance for this session.

    Design doc §7.2 interface contract: `simulate_turn(session, history) ->
    str`. §6.5.2: release policy varies by `session["scenario_type"]`
    (buying / browsing / intent_override / boundary).

    STUB: ignores the scenario-specific release policy and `history`'s
    content (only its length), returning a fixture utterance tagged with
    the turn index and scenario type.

    Args:
        session: One row from public_set.jsonl (sample_id, scenario_type,
            difficulty_bucket, category_bucket, ground_truth, user_profile).
        history: Prior utterances (both user and agent) this session, in
            order, used to decide how much detail to release next.

    Returns:
        The next user utterance.
    """
    scenario_type = session.get("scenario_type", "browsing")
    turn_index = len(history)
    return f"[STUB simulate_turn scenario={scenario_type} turn={turn_index}]"


def answer_clarification(session: dict, attribute: str) -> str:
    """Answer a clarifying question from the target's own record.

    Design doc §6.5.2: "When ask_attribute is set, the simulator answers
    from the target's record if it carries that attribute, and replies 'no
    preference' otherwise — which is what makes the answerability prior in
    Step 5 measurable." Owner Chellappan, §8.4 step D5.

    STUB: always returns "no preference", never inspecting whether the
    target record actually carries `attribute`.

    Args:
        session: One row from public_set.jsonl.
        attribute: The slot key the agent asked about (from
            clarify.pick_attribute()).

    Returns:
        The target's value for `attribute`, or "no preference".
    """
    return "no preference"
