"""User simulator + instrumented session driver.

The simulator holds a hidden intent card derived from the session's
ground-truth product and emits user turns per scenario type. `run_session()`
drives one full `reset()`/`respond()` conversation with it, so the telemetry
log (and, from that, the ranker training corpus) can be produced offline.

Design doc §6.5.1 / §6.5.2 (dialogue source, self-built simulator) and §8.4:
step D3 (facet extractor over target records), step D4 (the simulator with
per-scenario release policies), step D5 (clarification answering), step D6
(own loop over reset()/respond()).

Owner: Chellappan (Simulator and training corpus). §8.4.

WORKING ASSUMPTION — configuration C (§6.5.1): the kit ships no user
simulator, so this module is a required component and Owner D also owns the
conversation loop. If the kit turns out to bundle one (configuration A),
`simulate_turn` / `answer_clarification` / `run_session` become unused and
logging moves inside `respond()` instead.

Two contracts this module fixes for the instrumented loop (Owner D owns both
ends, so it may):

* `simulate_turn(session, history)` — `history` is the list of prior *user*
  utterances for the session (strings), oldest first. `len(history)` is the
  0-indexed number of the turn being produced. Agent replies are not passed
  in (they are stub text and carry no simulator signal).
* When the previous agent turn set `ask_attribute`, the driver calls
  `answer_clarification(session, attribute)` instead of `simulate_turn` —
  except for `intent_override` sessions before the contradiction has shipped
  (see `run_session`), which always get `simulate_turn` so 15% of the score
  is never left unrepresented (§8.4 D4: "Override sessions always emit a
  contradiction").

Phrasing bias (the doc requires Owner D to document it, §6.5.2 / §8.4):
turns are built from the target's own `categories` / `details` / `features`
fields — never its `title` — because echoing title wording back makes
retrieval artificially easy and inflates internal Hit Rate. Absolute scores
from a self-simulated run are internal diagnostics only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Per-scenario release policy (§6.5.2 table, §8.4 D4).
# ---------------------------------------------------------------------------
SCENARIO_POLICIES: dict[str, str] = {
    "buying": (
        "Emit two or more concrete, extractable attributes on turn 1; add one "
        "more detail each following turn."
    ),
    "browsing": (
        "Open with a scenario phrase that carries no extractable attribute; "
        "concede specifics only when the agent asks."
    ),
    "intent_override": (
        "State an attribute the target does NOT have, then contradict it on a "
        "later turn with the true value. Always emits the contradiction — "
        "falling back to category, then department, when the target carries no "
        "colour."
    ),
    "boundary": "Withhold most attributes; answer clarifications minimally.",
}

DEFAULT_SCENARIO = "browsing"
MAX_TURNS = 10  # §1.2: exceeding ten turns scores zero.

# Colours the extractor's fixed gazetteer always recognises regardless of
# catalogue size (extract._FIXED_COLOR_LOOKUPS). Used to pick a false colour
# for the intent_override path that is guaranteed to slot on turn 1.
_PALETTE: tuple[str, ...] = (
    "black",
    "blue",
    "green",
    "red",
    "gray",
    "white",
    "brown",
    "purple",
    "orange",
)

# Singular-form fixups for category leaves. Everything else falls through the
# generic rule in `_singularise`.
_SINGULAR_OVERRIDES: dict[str, str] = {
    "accessories": "accessory",
    "boots": "boot",
    "coats": "coat",
    "dresses": "dress",
    "jeans": "pair of jeans",
    "pants": "pair of pants",
    "shoes": "shoe",
    "shorts": "pair of shorts",
    "sneakers": "sneaker",
    "sunglasses": "pair of sunglasses",
    "trunks": "pair of trunks",
    "watches": "watch",
}

# Extractor fixed use-case vocabulary (extract._FIXED_USE_CASES). When a
# category leaf collides with one of these ("Running"), we keep the parent
# noun too so the phrase reads as a product ("running shoe"), not a use case.
_USE_CASE_WORDS: frozenset[str] = frozenset(
    {"hiking", "running", "gym", "winter", "outdoor", "work"}
)

_BROWSING_OPENERS: tuple[str, ...] = (
    "I need something for an upcoming trip",
    "looking for a gift for someone close to me",
    "something for a special occasion coming up",
    "just want to treat myself to something nice",
)
_BROWSING_FOLLOWUPS: tuple[str, ...] = (
    "nothing too over the top",
    "something that would get plenty of use",
    "I'll know it when I see it",
    "open to suggestions really",
)
_BOUNDARY_OPENERS: tuple[str, ...] = (
    "just browsing for now",
    "not sure what I want yet",
)
_BOUNDARY_FOLLOWUPS: tuple[str, ...] = (
    "still deciding",
    "hard to say",
    "keep going",
)

# Common apparel nouns for the intent_override "wrong category" fallback.
_FALLBACK_CATEGORY_NOUNS: tuple[str, ...] = (
    "jacket",
    "shirt",
    "shoe",
    "belt",
    "hat",
    "scarf",
    "dress",
    "bag",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts (empty list if it is absent)."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_catalog_index(catalog: Iterable[dict]) -> dict[str, dict]:
    """Index catalogue rows by `parent_asin` for O(1) target lookup."""
    return {row["parent_asin"]: row for row in catalog if row.get("parent_asin")}


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def _singularise(phrase: str) -> str:
    """Best-effort singular, lowercase noun for a category leaf.

    Handles "Jackets & Coats" -> "jacket" (first segment), irregular plurals
    via `_SINGULAR_OVERRIDES`, and the common -ies / -es / -s endings. §8.4
    D4: "Emits singular nouns."
    """
    head = re.split(r"\s*[&/,]\s*", phrase.strip())[0].strip()
    words = head.split()
    if not words:
        return head.lower()
    last = words[-1].lower()
    if last in _SINGULAR_OVERRIDES:
        words[-1] = _SINGULAR_OVERRIDES[last]
    elif last.endswith("ies") and len(last) > 3:
        words[-1] = last[:-3] + "y"
    elif last.endswith(("ses", "shes", "ches", "xes")):
        words[-1] = last[:-2]
    elif last.endswith("s") and not last.endswith("ss"):
        words[-1] = last[:-1]
    else:
        words[-1] = last
    return " ".join(w.lower() for w in words)


def scenario_type_of(session: dict) -> str:
    return session.get("scenario_type") or DEFAULT_SCENARIO


def target_asin_of(session: dict) -> Optional[str]:
    gt = session.get("ground_truth")
    if isinstance(gt, dict):
        return gt.get("parent_asin")
    return None


def attach_target_record(session: dict, catalog_index: dict[str, dict]) -> dict:
    """Return `session` with `_target_record` set from `catalog_index`.

    Mutates and returns the same dict. The simulator reads `_target_record`
    directly, keeping `simulate_turn(session, history)` free of a catalogue
    parameter.
    """
    asin = target_asin_of(session)
    if asin is not None and asin in catalog_index:
        session["_target_record"] = catalog_index[asin]
    return session


# ---------------------------------------------------------------------------
# D3 — facet extractor over target records
# ---------------------------------------------------------------------------
def extract_target_facets(target_record: dict) -> dict[str, object]:
    """Pull the intent-card facets from a target's catalogue record.

    Design doc §6.5.2 / §8.4 D3. Draws only from `categories`, `details`,
    `store`, `price` and `features` — never `title` — so simulated phrasing
    does not echo the exact tokens the indexes were built from.

    Returns a dict with any of: `department`, `category`, `category_noun`,
    `color`, `material`, `style`, `brand`, `price` (float), and
    `feature_phrases` (list[str]).
    """
    facets: dict[str, object] = {}
    if not isinstance(target_record, dict):
        return facets

    cats = [c.strip() for c in _string_list(target_record.get("categories")) if c.strip()]
    # Department: categories[1] (100% coverage per §3.2), i.e. the node just
    # below the "Clothing, Shoes & Jewelry" root.
    if len(cats) >= 2:
        facets["department"] = cats[1]
    if cats:
        leaf = cats[-1]
        facets["category"] = leaf
        noun = _singularise(leaf)
        if noun in _USE_CASE_WORDS and len(cats) >= 2:
            noun = f"{noun} {_singularise(cats[-2])}".strip()
        facets["category_noun"] = noun

    details = target_record.get("details")
    details_ci: dict[str, object] = {}
    if isinstance(details, dict):
        details_ci = {str(k).casefold(): v for k, v in details.items()}
    for facet_key, detail_key in (("color", "color"), ("material", "material"), ("style", "style")):
        value = details_ci.get(detail_key)
        if isinstance(value, str) and value.strip():
            facets[facet_key] = value.strip()

    store = target_record.get("store")
    if isinstance(store, str) and store.strip():
        facets["brand"] = store.strip()

    price = target_record.get("price")
    if isinstance(price, (int, float)) and not isinstance(price, bool):
        facets["price"] = float(price)

    phrases: list[str] = []
    for feature in _string_list(target_record.get("features")):
        p = feature.strip().lower().rstrip(".")
        if 2 <= len(p.split()) <= 5 and p not in phrases:
            phrases.append(p)
    if phrases:
        facets["feature_phrases"] = phrases[:4]

    return facets


def _facets_for(session: dict) -> dict[str, object]:
    record = session.get("_target_record")
    if isinstance(record, dict):
        return extract_target_facets(record)
    return {}


def _pick(options: tuple[str, ...], session: dict) -> str:
    """Deterministic choice from `options`, stable per session."""
    key = session.get("sample_id") or target_asin_of(session) or ""
    return options[hash(key) % len(options)]


# ---------------------------------------------------------------------------
# D4 — per-scenario turn generation
# ---------------------------------------------------------------------------
def _dept_prefix(facets: dict[str, object]) -> str:
    dept = str(facets.get("department", "")).strip().lower()
    if not dept:
        return ""
    if dept.endswith("'s") or dept.endswith("s"):
        return f"{dept} "
    return f"{dept}'s "


def _buying_turn(facets: dict[str, object], i: int) -> str:
    noun = str(facets.get("category_noun") or "item")
    color = str(facets.get("color", "")).lower()
    material = str(facets.get("material", "")).lower()
    brand = str(facets.get("brand", ""))
    style = str(facets.get("style", "")).lower()
    phrases: list[str] = list(facets.get("feature_phrases", []) or [])
    dept_p = _dept_prefix(facets)

    if i == 0:
        adjectives = [a for a in (color, material) if a]
        if adjectives:
            return f"I'm looking for a {' '.join(adjectives)} {dept_p}{noun}".replace("  ", " ")
        # No colour/material on the record: still open with >= 2 attributes
        # by combining department + category with brand or style.
        if brand:
            return f"I'm looking for a {dept_p}{noun} by {brand}"
        if style:
            return f"I'm looking for a {style} {dept_p}{noun}"
        return f"I'm looking for a {dept_p}{noun}, something well reviewed"
    if i == 1:
        if brand:
            return f"ideally {brand}"
        if style:
            return f"{style} would suit me"
        if phrases:
            return phrases[0]
        return "a well made one"
    detail_index = i - 2
    if detail_index < len(phrases):
        return phrases[detail_index]
    if style and not brand:
        return f"{style} style"
    return "that's about it"


def _browsing_turn(session: dict, i: int) -> str:
    if i == 0:
        return _pick(_BROWSING_OPENERS, session)
    return _BROWSING_FOLLOWUPS[(i - 1) % len(_BROWSING_FOLLOWUPS)]


def _boundary_turn(session: dict, i: int) -> str:
    if i == 0:
        return _pick(_BOUNDARY_OPENERS, session)
    return _BOUNDARY_FOLLOWUPS[(i - 1) % len(_BOUNDARY_FOLLOWUPS)]


def override_plan(facets: dict[str, object]) -> tuple[str, str, str]:
    """Choose (attribute, false_value, true_value) for an override session.

    §8.4 D4 / STEP 2: contradict colour when the target carries one, else
    "category or department". Order here is colour -> department -> category:

    * colour  — stays natural ("I want a blue jacket"); the false colour is
      from the extractor's fixed palette so it always slots.
    * department — the true value is `categories[1]` (100% coverage, §3.2),
      so it is in the gazetteer by construction; the false value is the
      opposite department, also present in any real catalogue. Reliable.
    * category — last resort. A bare singular noun ("belt") often will not
      exact-match a catalogue category path, so this path can degrade to a
      plain add; only used when the record has neither colour nor department.

    Both fallbacks open with a labelled turn 0 (`"department: <value>"`),
    which the extractor slots deterministically, so the later contradiction
    always has something to erase.
    """
    color = str(facets.get("color", "")).strip().lower()
    if color:
        false_color = next((c for c in _PALETTE if c != color), "black")
        return ("color", false_color, color)

    dept = str(facets.get("department", "")).strip().lower()
    if dept:
        false_dept = "men" if dept.startswith(("wom", "girl", "lad")) else "women"
        return ("department", false_dept, dept)

    noun = str(facets.get("category_noun", "")).strip().lower()
    false_noun = next((n for n in _FALLBACK_CATEGORY_NOUNS if n != noun), "jacket")
    return ("category", false_noun, noun or "item")


def override_contradiction_shipped(history: list[str]) -> bool:
    """Whether a prior user turn already carried the "not X, Y instead" pivot."""
    return any(
        h.strip().lower().startswith("not ") and " instead" in h.lower() for h in history
    )


def _intent_override_turn(facets: dict[str, object], i: int, history: list[str]) -> str:
    attribute, false_value, true_value = override_plan(facets)
    noun = str(facets.get("category_noun") or "item")

    if i == 0:
        if attribute == "color":
            return f"I want a {false_value} {noun}"
        # category / department: labelled form guarantees the false value slots
        return f"{attribute}: {false_value}"
    if not override_contradiction_shipped(history):
        if attribute == "category":
            return f"not a {false_value}, a {true_value} instead"
        return f"not {false_value}, {true_value} instead"
    # Contradiction already shipped: reinforce the true value, then add detail.
    if attribute != "color" and not any(
        f"{attribute}: {true_value}" in h.lower() for h in history
    ):
        return f"{attribute}: {true_value}"
    material = str(facets.get("material", "")).lower()
    style = str(facets.get("style", "")).lower()
    phrases: list[str] = list(facets.get("feature_phrases", []) or [])
    extras = [x for x in (material, style, *phrases) if x]
    if extras:
        return extras[(i - 2) % len(extras)]
    return "that's the main thing"


def simulate_turn(session: dict, history: list[str]) -> str:
    """Produce the next user utterance for `session`.

    Design doc §7.2 interface contract `simulate_turn(session, history) ->
    str`. `history` is the prior user utterances (see module docstring);
    `len(history)` is the 0-indexed turn being produced.

    Requires `session["_target_record"]` (set by `attach_target_record`) to
    draw real facets; without it the turn falls back to a generic,
    attribute-free phrase so the loop still runs.
    """
    scenario = scenario_type_of(session)
    i = len(history)
    facets = _facets_for(session)

    if not facets:
        return _browsing_turn(session, i)
    if scenario == "buying":
        return _buying_turn(facets, i)
    if scenario == "intent_override":
        return _intent_override_turn(facets, i, history)
    if scenario == "boundary":
        return _boundary_turn(session, i)
    return _browsing_turn(session, i)


# ---------------------------------------------------------------------------
# D5 — clarification answering
# ---------------------------------------------------------------------------
# Map the evaluator-facing ask_attribute vocabulary (state.CLARIFICATION_
# ATTRIBUTES) onto the simulator's facet keys.
_ATTRIBUTE_TO_FACET: dict[str, str] = {
    "category": "category_noun",
    "color": "color",
    "material": "material",
    "style": "style",
    "brand": "brand",
    "budget": "price",
}

NO_PREFERENCE = "no preference"


def answer_clarification(session: dict, attribute: str) -> str:
    """Answer a clarifying question from the target's own record.

    Design doc §6.5.2 / §8.4 D5: "Answers from the target record when it
    carries the asked attribute; returns 'no preference' otherwise. This is
    what makes the answerability prior measurable." Scenario-agnostic: the
    "minimal" boundary policy is expressed only through `simulate_turn`
    withholding, not by refusing to answer here.
    """
    facets = _facets_for(session)
    facet_key = _ATTRIBUTE_TO_FACET.get(attribute)
    if facet_key is None:
        return NO_PREFERENCE
    value = facets.get(facet_key)
    if attribute == "budget":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"around ${value:.0f}"
        return NO_PREFERENCE
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return NO_PREFERENCE


# ---------------------------------------------------------------------------
# D6 — instrumented session driver (own loop over reset()/respond())
# ---------------------------------------------------------------------------
def _recommended_asins(result: dict) -> list[str]:
    recs = result.get("recommendations") or []
    asins: list[str] = []
    for rec in recs:
        if isinstance(rec, dict) and rec.get("parent_asin"):
            asins.append(rec["parent_asin"])
        elif isinstance(rec, str):
            asins.append(rec)
    return asins


def run_session(
    agent: object,
    session: dict,
    catalog_index: dict[str, dict],
    *,
    max_turns: int = MAX_TURNS,
    top_k: int = 10,
) -> dict:
    """Drive one full conversation for `session` and return its transcript.

    Design doc §8.4 D6 ("Own loop over reset()/respond()"). The driver knows
    the ground-truth ASIN (it is the instrumented harness, not `respond()`),
    uses it only to decide when the session has converged, and never passes
    it into `agent.respond()`.

    Turn routing:

    * previous agent turn set `ask_attribute` -> `answer_clarification`,
      *unless* this is an `intent_override` session whose contradiction has
      not shipped yet -> then always `simulate_turn` (§8.4 D4).
    * otherwise -> `simulate_turn`.

    Stops at `max_turns`, or one turn after the target first reaches rank 1
    (a converged session), whichever comes first.

    Returns `{session_id, scenario_type, target_asin, turns, converged_turn,
    transcript}` where `transcript` is a list of
    `{turn, user, ask_attribute, recommended, target_rank}` dicts.
    """
    session = attach_target_record(dict(session), catalog_index)
    session_id = session.get("sample_id") or session.get("session_id") or "session"
    scenario = scenario_type_of(session)
    target_asin = target_asin_of(session)

    agent.reset(session_id, session.get("user_profile") or {})

    user_history: list[str] = []
    transcript: list[dict] = []
    pending_attribute: Optional[str] = None
    converged_turn: Optional[int] = None

    for turn in range(1, max_turns + 1):
        force_simulate = scenario == "intent_override" and not override_contradiction_shipped(
            user_history
        )
        if pending_attribute is not None and not force_simulate:
            user_message = answer_clarification(session, pending_attribute)
        else:
            user_message = simulate_turn(session, user_history)

        result = agent.respond(session_id, user_message, turn=turn, top_k=top_k)
        user_history.append(user_message)

        recommended = _recommended_asins(result)
        target_rank = (
            recommended.index(target_asin) + 1
            if target_asin is not None and target_asin in recommended
            else None
        )
        pending_attribute = result.get("ask_attribute")
        transcript.append(
            {
                "turn": turn,
                "user": user_message,
                "ask_attribute": pending_attribute,
                "recommended": recommended,
                "target_rank": target_rank,
            }
        )

        if converged_turn is not None:
            break
        if target_rank == 1:
            converged_turn = turn

    return {
        "session_id": session_id,
        "scenario_type": scenario,
        "target_asin": target_asin,
        "turns": len(transcript),
        "converged_turn": converged_turn,
        "transcript": transcript,
    }
