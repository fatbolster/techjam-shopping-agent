"""
Contract tests for the eleven ranking features (design doc §3.4 Step 6).

Every constraint here traces to a measurement in §2 of the design document.
The comment above each group names the finding it protects.

REWRITE NOTE: this file originally targeted a Candidate/SessionState shape
that predates the ones the rest of the codebase actually settled on
(Candidate(asin=, bm25_raw=, cos_raw=) built by retrieval.py; SessionState
the real dataclass Qikun's B-lane builds and 230 other tests already
exercise). Rewritten against those real shapes rather than the other way
around, since changing Candidate/SessionState to match the old test would
have broken retrieval.py and every B-lane test. Every substantive
assertion from the original file is preserved; two were adjusted to match
what's actually implemented rather than what was originally assumed:

  - price_fit's priced-item branch is an unimplemented fixture stub (issue
    #21 step 3 scoped only the null-safety branch), so "cheaper item
    should not score below a pricier one" isn't a real invariant yet —
    dropped, with a note.
  - rating_style_fit is a deliberate binary split ("usually positive" vs.
    everything else — see its RATING_STYLE_EXPECTS_HIGH comment in
    features.py), not a three-way distinction — "mixed" and "critical"
    both correctly collapse to 0.0. Adjusted from "three distinct values"
    to "usually positive differs from the other two, which agree."
"""

import math
import pytest

from features import (
    FEATURE_NAMES,
    PRICE_FIT_NEUTRAL,
    RATING_STYLE_EXPECTS_HIGH,
    bm25_norm,
    brand_match,
    category_match,
    cos_sim,
    extract_features,
    feature_vector,
    pop,
    price_fit,
    rare_tag_match,
    rating,
    rating_style_fit,
    slot_coverage,
    department_match,
    DEPARTMENT_NEUTRAL,
)
from indexes import Indexes
from state import RARE_TAGS, SessionState
from utils import Candidate


# --------------------------------------------------------------------------
# Fixtures — shapes mirror real indexes.build_facts_dict() output
# (asin -> {dept, cat3, store, price, rating_number, pop, rating, blob})
# --------------------------------------------------------------------------

@pytest.fixture
def facts():
    """The facts dict: asin -> record. Built once at startup (§3.2 Index 3)."""
    return {
        # popular, priced, full metadata — resembles a ground-truth target
        "B_POPULAR": {
            "dept": "Men",
            "cat3": "Clothing, Shoes & Jewelry > Men > Clothing > Jackets & Coats",
            "store": "london fog",
            "price": 89.99,
            "rating_number": 13854,
            "pop": math.log1p(13854) / math.log1p(100000),
            "rating": 4.4,
            "blob": ("london fog men's auburn zip-front golf jacket "
                     "100% polyester water resistant lightweight "
                     "clothing, shoes & jewelry men clothing jackets & coats"),
        },
        # long tail, NO PRICE — 78.9% of the catalogue looks like this (§2.2)
        "B_NOPRICE": {
            "dept": "Women",
            "cat3": "Clothing, Shoes & Jewelry > Women > Clothing > Dresses",
            "store": "goood times",
            "price": None,
            "rating_number": 3,
            "pop": math.log1p(3) / math.log1p(100000),
            "rating": 5.0,
            "blob": ("goood tiimes plus size black caftan for women "
                     "100% polyester one size fits most 1x to 3x "
                     "clothing, shoes & jewelry women clothing dresses casual"),
        },
        # thermal wording, for the rare-tag tests
        "B_THERMAL": {
            "dept": "Men",
            "cat3": "Clothing, Shoes & Jewelry > Men > Clothing > Jackets & Coats",
            "store": "wantdo",
            "price": 59.99,
            "rating_number": 307,
            "pop": math.log1p(307) / math.log1p(100000),
            "rating": 4.2,
            "blob": ("wantdo men's winter warmth windproof hooded jacket "
                     "thermal fleece lining insulated"),
        },
        # no rare-tag terms anywhere, control for the above
        "B_PLAIN": {
            "dept": "Men",
            "cat3": "Clothing, Shoes & Jewelry > Men > Clothing > Shirts",
            "store": "generic",
            "price": 19.99,
            "rating_number": 12,
            "pop": math.log1p(12) / math.log1p(100000),
            "rating": 3.8,
            "blob": "generic men's short sleeve cotton shirt button down",
        },
    }


@pytest.fixture
def indexes(facts):
    """A minimal real Indexes bundle — only `.facts` is read by features.py."""
    return Indexes(
        catalog=[],
        fts_conn=None,
        embedding_matrix=None,
        embedding_asins=[],
        facts=facts,
        category_lists={},
    )


@pytest.fixture
def state_empty():
    """Turn 1 of a browsing session — nothing extracted yet."""
    return SessionState(session_id="empty")


@pytest.fixture
def state_buying():
    """A buying session with several constraints accumulated."""
    return SessionState(
        session_id="buying",
        slots={
            "department": "Men",
            "category": "jacket",
            "color": "black",
            "price_target": "100",
        },
        profile_terms=["performance"],
        rating_style="usually positive",
    )


# --------------------------------------------------------------------------
# Candidate dataclass — the contract with Owner A (retrieval), confirmed in
# C1 (#20): utils.Candidate(asin, bm25_raw, cos_raw, sources), built fresh
# every turn by retrieval.py.
# --------------------------------------------------------------------------

def test_candidate_stream_scores_default_to_zero():
    """A popularity-stream candidate is contributed by no text stream.

    §8.3 step C4: "Popularity-stream candidates carry bm25 = 0 and compete
    on other features, as intended." retrieval.py must be able to
    construct one without supplying either score.
    """
    c = Candidate(asin="B_POPULAR")
    assert c.bm25_raw == 0.0
    assert c.cos_raw == 0.0


def test_candidate_accepts_partial_stream_scores():
    """Keyword found it, semantic did not. Both states must be representable."""
    c = Candidate(asin="B_POPULAR", bm25_raw=17.3)
    assert c.bm25_raw == 17.3
    assert c.cos_raw == 0.0


# --------------------------------------------------------------------------
# price_fit — protects §2.2: 78.9% of catalogue rows have price = None
# --------------------------------------------------------------------------

def test_price_fit_neutral_constant_is_strictly_interior():
    """The neutral value must not be an implicit exclusion.

    If PRICE_FIT_NEUTRAL were 0.0, a missing price would be indistinguishable
    from the worst possible price fit, and 78.9% of the catalogue would be
    silently penalised. It must sit strictly between the extremes.
    """
    assert 0.0 < PRICE_FIT_NEUTRAL < 1.0


def test_price_fit_returns_neutral_when_price_missing(facts, state_buying):
    """price = None must return the neutral value regardless of stated budget."""
    got = price_fit(Candidate(asin="B_NOPRICE"), facts, state_buying)
    assert got == pytest.approx(PRICE_FIT_NEUTRAL)


def test_price_fit_never_returns_none_or_nan(facts, state_buying, state_empty):
    # NOTE: the priced-item branch is a fixture_score() stub (issue #21
    # step 3 scoped only the null-safety branch, not a real budget-fit
    # formula), so this only protects "never crashes / never NaN", not
    # "cheaper scores at least as well as pricier" — that invariant does
    # not exist in the implementation yet.
    for state in (state_buying, state_empty):
        for asin in facts:
            got = price_fit(Candidate(asin=asin), facts, state)
            assert isinstance(got, float)
            assert not math.isnan(got)


def test_price_fit_survives_unknown_asin(facts, state_buying):
    """A candidate missing from facts must be treated like a null price."""
    got = price_fit(Candidate(asin="B_DOES_NOT_EXIST"), facts, state_buying)
    assert got == pytest.approx(PRICE_FIT_NEUTRAL)


# --------------------------------------------------------------------------
# department_match — the one structured attribute worth reading directly,
# and protects Change 2: a non-department bucket is unknown, never wrong
# --------------------------------------------------------------------------

@pytest.fixture
def dept_facts():
    return {
        "B_MEN": {"dept": "Men", "blob": ""},
        "B_WOMEN": {"dept": "Women", "blob": ""},
        "B_BUCKET": {"dept": "Boot Shop", "blob": ""},
        "B_NODEPT": {"dept": None, "blob": ""},
    }


def test_department_match_rewards_the_stated_department(dept_facts):
    state = SessionState(session_id="s", slots={"department": "Men"})
    assert department_match(Candidate(asin="B_MEN"), dept_facts, state) == pytest.approx(1.0)


def test_department_match_zeroes_the_opposite_department(dept_facts):
    """The reported bug: a men's query must not be satisfied by women's."""
    state = SessionState(session_id="s", slots={"department": "Men"})
    assert department_match(Candidate(asin="B_WOMEN"), dept_facts, state) == pytest.approx(0.0)


def test_department_match_is_neutral_for_a_non_department_bucket(dept_facts):
    """Guards Change 2. `categories[1]` holds 203 values and ~20% are store
    or product-type buckets; 30 of 200 targets sit under one. Scoring those
    0 would re-create the department filter as a penalty and lose them."""
    state = SessionState(session_id="s", slots={"department": "Men"})
    got = department_match(Candidate(asin="B_BUCKET"), dept_facts, state)
    assert got == pytest.approx(DEPARTMENT_NEUTRAL)
    assert got > department_match(Candidate(asin="B_WOMEN"), dept_facts, state)


def test_department_match_is_neutral_when_nothing_is_known(dept_facts):
    state_none = SessionState(session_id="s")
    assert department_match(Candidate(asin="B_MEN"), dept_facts, state_none) == pytest.approx(DEPARTMENT_NEUTRAL)
    state = SessionState(session_id="s", slots={"department": "Men"})
    assert department_match(Candidate(asin="B_NODEPT"), dept_facts, state) == pytest.approx(DEPARTMENT_NEUTRAL)
    assert department_match(Candidate(asin="B_MISSING"), dept_facts, state) == pytest.approx(DEPARTMENT_NEUTRAL)


def test_department_match_is_case_insensitive(dept_facts):
    state = SessionState(session_id="s", slots={"department": "  men "})
    assert department_match(Candidate(asin="B_MEN"), dept_facts, state) == pytest.approx(1.0)


def test_department_match_separates_unisex_listing_from_slot_coverage(dept_facts):
    """The residue word boundaries cannot reach.

    A women's listing whose text genuinely says "for men women" scores full
    slot_coverage on a stated "Men" — correctly, the word is there. Only the
    structured department tells the two apart.
    """
    facts = {"B_UNISEX": {
        "dept": "Women",
        "blob": "trendoux winter gloves for men women - upgraded touch screen",
    }}
    state = SessionState(session_id="s", slots={"department": "Men"})
    cand = Candidate(asin="B_UNISEX")
    assert slot_coverage(cand, facts, state) == pytest.approx(1.0)
    assert department_match(cand, facts, state) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# slot_coverage — protects §2.3: structured attributes barely exist
# (details.Color 4.9%, details.Material 4.1%, no size field at all)
# --------------------------------------------------------------------------

def test_slot_coverage_empty_slots_does_not_divide_by_zero(facts, state_empty):
    got = slot_coverage(Candidate(asin="B_POPULAR"), facts, state_empty)
    assert isinstance(got, float)
    assert not math.isnan(got)
    assert 0.0 <= got <= 1.0


def test_slot_coverage_matches_text_not_structured_fields(facts):
    """B_NOPRICE has no details.Material, but its blob says 'polyester'.

    §2.3: attribute matching must operate over text. A structured-field
    lookup would score this 0 and lose the product.
    """
    state = SessionState(session_id="s", slots={"material": "polyester"})
    got = slot_coverage(Candidate(asin="B_NOPRICE"), facts, state)
    assert got > 0.0


def test_slot_coverage_is_a_fraction_of_slot_terms(facts):
    """Two slots, one present in the blob -> roughly half."""
    state = SessionState(
        session_id="s", slots={"color": "black", "category": "wristwatch"}
    )
    got = slot_coverage(Candidate(asin="B_NOPRICE"), facts, state)
    assert 0.0 < got < 1.0


def test_slot_coverage_all_terms_present_scores_maximum(facts):
    state = SessionState(
        session_id="s", slots={"color": "black", "category": "caftan"}
    )
    got = slot_coverage(Candidate(asin="B_NOPRICE"), facts, state)
    assert got == pytest.approx(1.0)


def test_slot_coverage_no_terms_present_scores_zero(facts):
    state = SessionState(
        session_id="s", slots={"color": "turquoise", "category": "snowboard"}
    )
    got = slot_coverage(Candidate(asin="B_NOPRICE"), facts, state)
    assert got == pytest.approx(0.0)


def test_slot_coverage_men_does_not_match_womens(facts):
    """The gender slot must not be satisfied by the "men" inside "women's".

    A plain `term in blob` substring test conflated the two: on the real
    50,000-row catalogue 91.4% of blobs contain the substring "men" but
    only 29.7% contain the word, so 30,850 products scored full credit on
    the one feature meant to carry gender — women's products included.
    """
    facts = dict(facts)
    facts["B_WOMENS"] = {
        "dept": "Women",
        "blob": "prettygarden women's summer wrap maxi dress, floral v neck",
    }
    state = SessionState(session_id="s", slots={"department": "Men"})
    got = slot_coverage(Candidate(asin="B_WOMENS"), facts, state)
    assert got == pytest.approx(0.0)


def test_slot_coverage_men_still_matches_mens_possessive(facts):
    """Tightening to word boundaries must not cost the true positives."""
    state = SessionState(session_id="s", slots={"department": "Men"})
    got = slot_coverage(Candidate(asin="B_POPULAR"), facts, state)
    assert got == pytest.approx(1.0)


def test_slot_coverage_term_does_not_match_inside_longer_word(facts):
    """The collision is general, not a gender special case.

    "red" must not be satisfied by "shredded", so the fix is a boundary
    rule applied to every slot term rather than a hardcoded gender list.
    """
    facts = dict(facts)
    facts["B_SHREDDED"] = {"dept": "Grocery", "blob": "shredded parmesan cheese"}
    state = SessionState(session_id="s", slots={"color": "red"})
    got = slot_coverage(Candidate(asin="B_SHREDDED"), facts, state)
    assert got == pytest.approx(0.0)


def test_slot_coverage_matches_terms_with_non_word_characters(facts):
    """A term may start or end in punctuation, where a bare \\b flips sense."""
    facts = dict(facts)
    facts["B_COTTON"] = {"dept": "Men", "blob": "tee made of 100% cotton, preshrunk"}
    state = SessionState(session_id="s", slots={"material": "100% cotton"})
    got = slot_coverage(Candidate(asin="B_COTTON"), facts, state)
    assert got == pytest.approx(1.0)


def test_slot_coverage_flattens_multi_value_slots(facts):
    """A multi-value slot (tuple) is checked term-by-term, not as one blob."""
    state = SessionState(
        session_id="s", slots={"material": ("polyester", "wool")}
    )
    got = slot_coverage(Candidate(asin="B_NOPRICE"), facts, state)
    assert got == pytest.approx(0.5)  # "polyester" present, "wool" absent


# --------------------------------------------------------------------------
# rare_tag_match — protects §2.4: only three of eight tags carry signal
# (fit -5.1, durability -6.1, style +1.7, material -0.4 vs popularity-matched
#  baseline; only performance/warmth/weather show real lift)
# --------------------------------------------------------------------------

def test_rare_tags_covers_exactly_the_three_measured_tags():
    """§2.4: the five common tags measured at or below chance and are excluded.

    Note: this filtering happens in state.py's derive_profile_terms(), not
    in features.py — rare_tag_match trusts state.profile_terms is already
    filtered by the time it sees it. This is the real filter's source of
    truth, checked exactly.
    """
    assert set(RARE_TAGS) == {"warmth", "weather", "performance"}


def test_rare_tag_match_zero_when_profile_terms_empty(facts):
    """~78% of sessions carry none of the three. The feature must be inert."""
    state = SessionState(session_id="s", profile_terms=[])
    got = rare_tag_match(Candidate(asin="B_THERMAL"), facts, state)
    assert got == pytest.approx(0.0)


def test_rare_tag_match_fires_on_matching_product(facts):
    state = SessionState(session_id="s", profile_terms=["warmth"])
    got = rare_tag_match(Candidate(asin="B_THERMAL"), facts, state)
    assert got > 0.0


def test_rare_tag_match_zero_for_product_without_the_terms(facts):
    state = SessionState(session_id="s", profile_terms=["warmth"])
    got = rare_tag_match(Candidate(asin="B_PLAIN"), facts, state)
    assert got == pytest.approx(0.0)


def test_rare_tag_match_unions_multiple_rare_tags(facts):
    """A session may carry two rare tags; term lists union rather than conflict."""
    state = SessionState(session_id="s", profile_terms=["weather", "performance"])
    got = rare_tag_match(Candidate(asin="B_POPULAR"), facts, state)
    assert isinstance(got, float)
    assert not math.isnan(got)


def test_derive_profile_terms_excludes_the_five_common_tags():
    """End-to-end: raw preference_tags -> filtered profile_terms -> rare_tag_match.

    Exercises the real split of responsibility (state.py filters, features.py
    trusts the filter) rather than assuming rare_tag_match itself filters —
    the original version of this test constructed profile_terms directly
    with common tags in it, which the real pipeline never does.
    """
    from state import derive_profile_terms

    raw_tags = ["fit", "comfort", "durability", "material", "style", "warmth"]
    filtered = derive_profile_terms({"preference_tags": raw_tags})
    assert filtered == ["warmth"]


# --------------------------------------------------------------------------
# rating_style_fit — protects §2.4.1: real (p=0.003) but must not be
# paired with average_prior_rating, which is perfectly collinear with it
# --------------------------------------------------------------------------

def test_rating_style_fit_reads_rating_style(facts):
    state = SessionState(session_id="s", rating_style="usually positive")
    got = rating_style_fit(Candidate(asin="B_POPULAR"), facts, state)
    assert isinstance(got, float)
    assert not math.isnan(got)


def test_rating_style_fit_ignores_average_prior_rating(facts):
    """§2.4.1: the two fields are the same signal. Only rating_style is used.

    SessionState has no average_prior_rating field at all, so this sets it
    as an ad-hoc extra attribute (Python allows this on a plain dataclass
    instance) purely to prove rating_style_fit never reads it.
    """
    state = SessionState(session_id="s", rating_style="critical")
    state.average_prior_rating = 1.0
    a = rating_style_fit(Candidate(asin="B_POPULAR"), facts, state)
    state.average_prior_rating = 5.0
    b = rating_style_fit(Candidate(asin="B_POPULAR"), facts, state)
    assert a == pytest.approx(b)


def test_rating_style_fit_is_a_deliberate_binary_split(facts):
    """"usually positive" differs from the other two, which agree with each other.

    §2.4.1's measured target means put "usually positive" (4.413) clearly
    apart from "mixed" (4.305) and "critical" (4.282), which sit close to
    each other — RATING_STYLE_EXPECTS_HIGH implements that as a binary
    split, not a three-way distinction. (The original version of this test
    expected three distinct values; that was never actually the design.)
    """
    positive = rating_style_fit(
        Candidate(asin="B_POPULAR"),
        facts,
        SessionState(session_id="s", rating_style=RATING_STYLE_EXPECTS_HIGH),
    )
    mixed = rating_style_fit(
        Candidate(asin="B_POPULAR"), facts, SessionState(session_id="s", rating_style="mixed")
    )
    critical = rating_style_fit(
        Candidate(asin="B_POPULAR"), facts, SessionState(session_id="s", rating_style="critical")
    )
    assert positive > 0.0
    assert mixed == pytest.approx(0.0)
    assert critical == pytest.approx(0.0)
    assert mixed == pytest.approx(critical)


def test_rating_style_fit_handles_missing_style(facts):
    """rating_style defaults to None; the feature must degrade, not raise."""
    state = SessionState(session_id="s")
    got = rating_style_fit(Candidate(asin="B_POPULAR"), facts, state)
    assert isinstance(got, float)
    assert not math.isnan(got)


# --------------------------------------------------------------------------
# pop — protects §2.1 and §3.2: precomputed at startup, not per candidate
# --------------------------------------------------------------------------

def test_pop_is_read_from_facts_not_recomputed(facts):
    """§3.2: 'precomputed pop = log1p(rn)/log1p(100000)'.

    pop() never reads rating_number at all — it only reads the
    precomputed `pop` field — so poisoning rating_number while leaving
    `pop` intact must leave the result unaffected.
    """
    poisoned = {**facts, "B_POPULAR": {**facts["B_POPULAR"], "rating_number": 0}}
    got = pop(Candidate(asin="B_POPULAR"), poisoned)
    assert got == pytest.approx(facts["B_POPULAR"]["pop"])


def test_pop_orders_by_popularity(facts):
    """13,854 reviews must outrank 3 reviews. §2.1: 63% of targets are top-1%."""
    hi = pop(Candidate(asin="B_POPULAR"), facts)
    lo = pop(Candidate(asin="B_NOPRICE"), facts)
    assert hi > lo


# --------------------------------------------------------------------------
# Remaining features — range and type contracts
# --------------------------------------------------------------------------

def test_bm25_norm_is_normalised():
    pool = [Candidate(asin="B_POPULAR", bm25_raw=17.3), Candidate(asin="B_NOPRICE", bm25_raw=5.0)]
    got = bm25_norm(pool[0], pool)
    assert got == pytest.approx(1.0)  # max in the pool


def test_bm25_norm_zero_for_popularity_stream_candidate():
    """Stream candidates carry bm25_raw = 0 and must score 0 here, not raise."""
    pool = [Candidate(asin="B_POPULAR"), Candidate(asin="B_NOPRICE", bm25_raw=5.0)]
    got = bm25_norm(pool[0], pool)
    assert got == pytest.approx(0.0)


def test_bm25_norm_handles_pool_with_no_keyword_scores():
    """Every candidate popularity/semantic-only: must not divide by zero."""
    pool = [Candidate(asin="B_POPULAR"), Candidate(asin="B_NOPRICE")]
    got = bm25_norm(pool[0], pool)
    assert got == pytest.approx(0.0)


def test_cos_sim_zero_for_popularity_stream_candidate():
    got = cos_sim(Candidate(asin="B_POPULAR"))
    assert got == pytest.approx(0.0)


def test_cos_sim_returns_raw_value_directly():
    got = cos_sim(Candidate(asin="B_POPULAR", cos_raw=0.42))
    assert got == pytest.approx(0.42)


def test_rating_is_scaled_to_unit_interval(facts):
    got = rating(Candidate(asin="B_NOPRICE"), facts)  # rating 5.0
    assert 0.0 <= got <= 1.0
    assert got == pytest.approx(1.0)


def test_category_match_fires_on_matching_group(facts):
    state = SessionState(session_id="s", slots={"category": "jacket"})
    hit = category_match(Candidate(asin="B_POPULAR"), facts, state)
    miss = category_match(Candidate(asin="B_NOPRICE"), facts, state)
    assert hit > miss


def test_category_match_does_not_exclude_on_mismatch(facts):
    """§3.4: category leans, it never filters. A mismatch must still score."""
    state = SessionState(session_id="s", slots={"category": "jacket"})
    got = category_match(Candidate(asin="B_NOPRICE"), facts, state)
    assert got is not None
    assert not math.isnan(got)


def test_pop_is_clamped_to_one(facts):
    """pop is a normalised prior; the 5 products over 100k ratings must not
    exceed it. build_facts_dict() computes the ratio unbounded, and pop
    carries the largest fitted weight, so an overflow is a bonus no other
    product can reach."""
    facts = dict(facts)
    facts["B_MEGA"] = {"pop": 1.1222, "blob": ""}
    got = pop(Candidate(asin="B_MEGA"), facts)
    assert got == pytest.approx(1.0)


def test_pop_leaves_normal_values_untouched(facts):
    facts = dict(facts)
    facts["B_NORMAL"] = {"pop": 0.4242, "blob": ""}
    assert pop(Candidate(asin="B_NORMAL"), facts) == pytest.approx(0.4242)


def test_category_match_mens_does_not_match_womens_path(facts):
    """A stated "mens" category must not score on a women's category path.

    Same collision as slot_coverage, on the other text feature: "men" is a
    substring of "womens", so 59.1% of the real catalogue took credit for
    a token its path does not contain.
    """
    facts = dict(facts)
    facts["B_WOMENS"] = {
        "cat_path": "clothing, shoes & jewelry > women > clothing > dresses",
        "blob": "",
    }
    for stated in ("mens", "men"):
        state = SessionState(session_id="s", slots={"category": f"{stated} jacket"})
        got = category_match(Candidate(asin="B_WOMENS"), facts, state)
        assert got == pytest.approx(0.0), stated


def test_category_match_still_matches_across_singular_and_plural(facts):
    """Boundary matching must keep the number-insensitivity of the old form."""
    facts = dict(facts)
    facts["B_DRESS"] = {
        "cat_path": "clothing, shoes & jewelry > women > clothing > dresses",
        "blob": "",
    }
    for stated in ("dress", "dresses"):
        state = SessionState(session_id="s", slots={"category": f"women {stated}"})
        got = category_match(Candidate(asin="B_DRESS"), facts, state)
        assert got == pytest.approx(1.0), stated


def test_brand_match_uses_store_field(facts):
    state = SessionState(session_id="s", slots={"brand": "london fog"})
    hit = brand_match(Candidate(asin="B_POPULAR"), facts, state)
    miss = brand_match(Candidate(asin="B_PLAIN"), facts, state)
    assert hit > miss


def test_brand_match_is_case_insensitive(facts):
    state = SessionState(session_id="s", slots={"brand": "London Fog"})
    got = brand_match(Candidate(asin="B_POPULAR"), facts, state)
    assert got > 0.0


# --------------------------------------------------------------------------
# feature_vector / extract_features — the contract with rank.py and with
# Owner D's telemetry corpus
# --------------------------------------------------------------------------

def test_feature_vector_has_one_value_per_feature_name(indexes, state_buying):
    pool = [Candidate(asin="B_POPULAR", bm25_raw=12.0, cos_raw=0.7)]
    vec = feature_vector(pool[0], pool, indexes, state_buying)
    assert len(vec) == len(FEATURE_NAMES)
    # Pinned: the corpus, the fitted ranker's coefficients and telemetry's
    # rows are all positional in FEATURE_NAMES order, so changing the count
    # invalidates every persisted model and must be a deliberate edit.
    assert len(FEATURE_NAMES) == 11


def test_feature_vector_order_is_stable(indexes, state_buying):
    """Telemetry logs positionally; a reordering silently corrupts the corpus."""
    pool = [Candidate(asin="B_POPULAR", bm25_raw=12.0)]
    a = feature_vector(pool[0], pool, indexes, state_buying)
    b = feature_vector(pool[0], pool, indexes, state_buying)
    assert a == b


def test_feature_names_exclude_average_prior_rating():
    """§2.4.1: collinear with rating_style. Including both destabilises the fit."""
    assert "average_prior_rating" not in FEATURE_NAMES
    assert "rating_style_fit" in FEATURE_NAMES


def test_feature_vector_all_floats_no_nan(indexes, facts, state_buying, state_empty):
    for state in (state_buying, state_empty):
        for asin in facts:
            pool = [Candidate(asin=asin, bm25_raw=5.0, cos_raw=0.3)]
            vec = feature_vector(pool[0], pool, indexes, state)
            for name, v in zip(FEATURE_NAMES, vec):
                assert isinstance(v, float), f"{name} returned {type(v)}"
                assert not math.isnan(v), f"{name} returned NaN"


def test_feature_vector_survives_popularity_stream_candidate(indexes, state_buying):
    """The whole vector must compute for a candidate no text stream found."""
    pool = [Candidate(asin="B_POPULAR")]
    vec = feature_vector(pool[0], pool, indexes, state_buying)
    assert len(vec) == len(FEATURE_NAMES)
    assert all(not math.isnan(v) for v in vec)


def test_feature_vector_survives_unknown_asin(indexes, state_buying):
    """Defensive: a candidate missing from facts must not crash the turn."""
    pool = [Candidate(asin="B_DOES_NOT_EXIST")]
    try:
        vec = feature_vector(pool[0], pool, indexes, state_buying)
    except KeyError:
        pytest.fail("feature_vector must handle an asin absent from facts")
    assert len(vec) == len(FEATURE_NAMES)


def test_extract_features_and_feature_vector_agree(indexes, state_buying):
    """The dict and tuple forms must be the same values, same order."""
    pool = [Candidate(asin="B_POPULAR", bm25_raw=12.0, cos_raw=0.7)]
    as_dict = extract_features(pool[0], pool, indexes, state_buying)
    as_tuple = feature_vector(pool[0], pool, indexes, state_buying)
    assert tuple(as_dict[name] for name in FEATURE_NAMES) == as_tuple
