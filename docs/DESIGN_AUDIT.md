# Design audit — implementation vs. `Shopping-Copilot-Technical-Design.pdf` and `evaluator/evaluator.py`

Report only. No code changed.

Sources audited: the design doc (all sections), `evaluator/evaluator.py`,
and the implementation (`agents/our_agent.py`, `retrieval.py`,
`indexes.py`, `features.py`, `rank.py`, `clarify.py`, `extract.py`,
`state.py`). Measurements below were run against the real
`data/catalog.jsonl` (50,000 rows), `data/public_set.jsonl` (200
sessions), and `data/telemetry.jsonl` (4,497 logged turns).

**Headline:** two findings are material enough to change scores. D-1 (the
evaluator's intent-override phrasing does not trigger our negation
detector) silently disables the architecture's stated principal
differentiator on 15% of sessions. D-2 (junk department values used as a
hard filter) can delete the target irrecoverably on the buying track.
Everything in §3 verified as specified except two items, noted there.

---

## 1. DIVERGENCES

### D-1 — Intent override does not fire on the evaluator's phrasing · **BUG, highest severity**

**Where:** `extract.py:1494` `detect_slot_operations()` / the B5 negation
patterns; exercised via `agents/our_agent.py:respond()`.

**Doc says:** §3.1 — the pure-vector failure ("cannot forget") is
"disqualifying, because intent override is named in the brief and
constitutes 15% of sessions." §5.2 — canonical reconstruction "rebuilds
the query from a dictionary that no longer contains black, so there is
nothing to suppress. This is the behaviour named in Pillar II." §3.4 Step
1 lists the negation patterns: "`not X`, `instead of X`, `actually Y`".

**Code does:** the detector matches the `"not X, Y instead"` construction
that `simulate.py` (our own simulator) emits. The evaluator emits a
different string — `evaluator.py:85`:

```python
"message": f"Actually, ignore my earlier preference. What I need is: {new_value}."
```

Replayed through `update_slots()`:

```
turn 1  "I am looking for Boot Shop. I prefer a different style."
        slots: {'department': 'Boot Shop', 'category': 'Boot Shop',
                'brand': 'Style', 'style': ('Boot',)}
turn 2  "Actually, ignore my earlier preference. What I need is: leather."
        slots: {'department': 'Boot Shop', 'category': 'Boot Shop',
                'brand': 'Style', 'style': ('Boot',), 'material': ('leather',)}
```

Nothing is deleted. The superseded constraints persist into the canonical
intent string for the rest of the session — precisely the monotonic-memory
failure §3.1 calls disqualifying, reproduced inside the symbolic layer that
exists to prevent it.

**Correct or bug:** **Bug.** The doc's intent is right and the mechanism
is built; it is keyed to the wrong surface form. Note the doc's own
pattern list includes `actually Y`, which *would* have matched — so this
is an implementation gap against the doc, not a doc that went stale.

**Contributing factor visible in the same trace:** `"I prefer a different
style."` extracts `brand: 'Style'` and `style: ('Boot',)`, and `Boot Shop`
lands in *both* `department` and `category`. §9 already names slot
extraction as "the weakest link"; this is that limitation firing on real
evaluator input.

---

### D-2 — The department filter has no whitelist and filters on junk values · **BUG**

**Where:** `retrieval.py:114-121` (`keyword_stream`), against
`indexes.py:354` (`dept = categories[1]`).

**Doc says:** §3.2 Index 3 — "Department is taken from `categories[1]`
(100% coverage) rather than `details.Department` (87%)." §3.4 Step 4 —
keyword stream "filters department + category, buying only." §5.3's
governing principle — "Filtering deletes permanently, so it acts only on
stated constraints over **well-populated fields**." §9 — "A hallucinated
constraint on the buying track causes a hard filter to delete the target
irrecoverably."

**Code does:** exact case-insensitive equality against
`facts[asin]['dept']`, with no validation that the value is a real
department:

```python
if department and (facts.get("dept") or "").casefold() != department.casefold():
    continue
```

Measured on the real catalogue — `categories[1]` is **100% populated but
not a department field**:

| | |
|---|---|
| distinct `categories[1]` values | **203** |
| rows where it is Women/Men/Girls/Boys/Baby | 40,155 (80.3%) |
| rows where it is something else | **9,845 (19.7%)** |
| distinct non-department values | 198 |

Top non-department values: `Novelty & More` (3,376), **`Westlake`**
(1,136), **`Boot Shop`** (1,131), `Sport Specific Clothing` (1,114),
`Luggage & Travel Gear` (976), `Costumes & Accessories` (937).

These reach the slot dictionary: `extract.py`'s `_looks_like_promo_noise()`
(line 159) rejects digits/`$`/`%`/`clearance|test|outlet|markdown`, which
catches `Toddler Test` and `Swimwear TEST` but **not** `Westlake`, `Boot
Shop`, `Novelty & More`, or `Sport Specific Clothing`. The built gazetteer
holds **123 department entries**, and `westlake` → `'Westlake'` resolves.

**Blast radius:** **30 of the 200 ground-truth targets (15%)** sit under a
non-canonical `categories[1]` — including 8 under `Sport Specific
Clothing`, 6 under `Novelty & More`, 6 under `Luggage & Travel Gear`, 5
under `Boot Shop`. Any buying-track turn that extracts a junk department
not equal to the target's own junk department deletes that target from the
keyword stream permanently.

**Correct or bug:** **Bug**, and it is the §9 failure mode the doc
predicted. The doc's "100% coverage" claim for `categories[1]` is true but
misleading — 100% *populated*, ~80% *meaningful as a department*. §5.3's
"well-populated fields" test is satisfied on a technicality while its
intent is violated.

---

### D-3 — `price_fit`'s priced branch is a hash, not a budget fit · **BUG (latent)**

**Where:** `features.py:157`.

**Doc says:** §3.4 Step 6 feature table — `price_fit` = "0.5 when null,
else **fit to stated budget**".

**Code does:** returns `fixture_score("price_fit:" + candidate.asin)` — a
deterministic MD5-derived pseudo-score keyed on the ASIN. It ignores
`state` entirely:

```
price_fit(priced item, no budget stated)  = 0.0597…
price_fit(priced item, budget "$20" set)  = 0.0597…   ← identical
```

The null branch is correct (see V-1). The priced branch injects
**deterministic per-ASIN noise** into the feature vector for the 21.1% of
rows that carry a price.

**Correct or bug:** **Bug**, though currently low-impact: the fitted
regression assigned `price_fit` a weight of **−0.67** (small, and the
sign is uninterpretable precisely because the input is noise), and only 1
of 4,497 logged turns carried a price constraint in the canonical intent.
The module docstring is honest that this is unimplemented; the risk is
that it reads as a *feature* in the ablation table rather than as noise.

---

### D-4 — `slot_coverage` treats price slots as text terms · **BUG (latent)**

**Where:** `features.py:237` — `for value in state.slots.values():`
iterates *all* slots.

**Doc says:** §3.4 Step 6 — "fraction of **slot terms** present in the
text blob"; §2.3 — text matching applies to *attributes* (color, material,
style…).

**Code does:** includes `price_min` / `price_max` / `price_target` as
substring probes. With `slots = {'price_max': '20', 'color': ('red',)}`
against a blob containing "red": coverage = 0.5, and the bare string
`"20"` is searched in the blob — matching `$20`, `2019`, `120`, `20%`
spuriously, and diluting the denominator when absent.

**Correct or bug:** **Bug**, currently near-zero impact (1 of 4,497 turns
sets a price slot). Worth fixing as a correctness matter, not a scoring
one.

---

### D-5 — Ten features are computed, but the diagram says nine · **Doc is out of date**

**Where:** `features.py:32` `FEATURE_NAMES` (10 entries); `rank.py`
`HANDSET_WEIGHTS` (10 entries).

**Doc says:** §3.4 Step 6 body says "ten features" and the table lists
ten. But §4's diagram block ⑥ says "RANK — **9 features** → logistic
regression" and lists nine (`rating_style_fit` omitted). §6.6's objective
note also says "modest at **nine** features" and "given only **nine**
weights".

**Code does:** ten, consistently, everywhere.

**Correct or bug:** **Doc is out of date** — the code is right. §2.4.1
narrates restoring `rating_style_fit` after initially cutting it ("a
correction we made under measurement"); the diagram and §6.6 were not
updated to match. Harmless, but it makes the doc self-contradictory.

---

### D-6 — `MAX_CLARIFICATIONS_PER_SESSION = 3` strands 15% of turns · **Divergence; see §2, DC-3**

**Where:** `clarify.py:71`.

**Doc says:** §3.4 Step 5 — "A per-session cap on clarifications protects
MTTC."

**Code does:** caps at 3. Measured over the 4,497 logged turns: **203
sessions exhaust the cap**, after which **679 turns (15.1% of all turns)**
run with `ask_attribute=None`. Per the evaluator (see DC-3), such a turn
returns a content-free reply and discloses nothing — so those turns cannot
improve the ranking and simply burn toward the 10-turn limit.

**Correct or bug:** the *cap* is doc-faithful; its *rationale* is
falsified by the evaluator (DC-3). Flagging as a divergence from intent
rather than from letter. Not fixing under this report.

---

### D-7 — We ask about `brand` most often; the doc says that is the wrong question · **Divergence from stated intent**

**Where:** `clarify.py:53-68` (`ANSWERABILITY_PRIOR`, `ASK_THRESHOLD = 1.0`).

**Doc says:** §3.4 Step 5's boxed rationale — "Naive entropy would always
ask about brand… A bad question costs Hit Rate, not merely MTTC.
Weighting by answerability inverts the ranking correctly: category 2.84,
department 1.67, brand 0.73."

**Code does:** the priors are set as the doc intends (`category` 0.9,
`brand` 0.3), but brand's entropy over a ~200-candidate pool is large
enough to win anyway. Measured ask distribution over 1,273 asks:

| attribute | asks | share |
|---|---:|---:|
| **brand** | **404** | **31.7%** |
| color | 255 | 20.0% |
| use_case | 210 | 16.5% |
| style | 180 | 14.1% |
| category | 105 | 8.2% |
| material | 62 | 4.9% |
| feature / budget / size | 57 | 4.5% |

The exact inversion §3.4 Step 5 says the answerability weighting exists to
prevent is occurring in production. `category` — the doc's top-scoring
attribute — is asked a quarter as often as brand.

**Correct or bug:** **Divergence.** The mechanism is implemented as
specified; the constants do not achieve the doc's stated goal on real
pools. A prior of 0.3 is not enough to overcome an entropy gap of
H≈7.26 vs H≈3.34. Candidate for E7's grid search.

---

### D-8 — Track is recomputed per turn from a state the evaluator rarely populates · **Observation, not a defect**

**Where:** `state.py:427` `pick_track()`; `retrieval.py:114`.

Worth recording alongside D-2: the department filter only fires on the
buying track, and the buying track requires a department slot / 2+ hard
constraints / a leaf-category noun. So D-2's blast radius is gated by how
often the evaluator's phrasing produces those. This audit did not measure
buy/browse split per scenario; recommended before prioritising D-2's fix.

---

## 2. DOC CORRECTIONS

### DC-1 — §6.5.1 / §6.5.2: the kit ships a simulator · **CONFIRMED**

§6.5.1 presents three configurations and §6.5.2 specifies a fallback
self-built simulator. `evaluator/evaluator.py` bundles its own user
simulator outright — `initial_message()` (line 154), `customer_reply()`
(line 166), `behavior_for()` (line 74), driven by `intent_card()` (line
52) derived per-session from the target product. This is **configuration
A**: "Run it directly. Log inside `respond()`. No additional work."

`simulate.py` (D3–D6) is therefore not on the scoring path. It remains
useful — it produced the 24,000-row training corpus that C5's regression
was fitted on — but §6.5.2's framing of it as a *required component* is
superseded. Also note §6.5.2's stated mitigation ("draws from categories
and details rather than title") does not apply to the evaluator's
simulator, which builds `hard_constraints` from `features` + `details`
and `target_category` from the **title** (`intent_card()`, line 53).

**Correction:** mark §6.5.1 resolved as configuration A; demote §6.5.2 to
"internal corpus generation only".

---

### DC-2 — §3.4 Step 6: token cost is reported but not scored · **CONFIRMED**

§3.4 Step 6's boxed note: the LLM rerank's token cost is "reported in
`usage` **and scored under Efficiency**", and the component "is retained
only if ablation shows it earns its token cost." §6.3 calls the LLM row
"the only genuine trade-off… the decision depends on their relative
weighting inside TechnicalScore."

The evaluator accumulates tokens (lines 245–250) and reports them as
`reported_token_usage`, but **Efficiency is computed purely from MTTC**
(line 279):

```python
efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
technical_score = 0.50 * hit_rate_at_10 + 0.30 * mrr + 0.20 * efficiency
```

`total_tokens` appears nowhere in `technical_score`. There is no token
term at any weight.

**Correction:** §3.4 Step 6 and §6.3's "LLM row is the only genuine
trade-off" are wrong. Locally there is **no token penalty at all** — an
LLM rerank that improves MRR is free. §6.3's expected direction for
`− LLM rerank` ("MRR falls, Efficiency rises sharply") is wrong on the
second clause: Efficiency would not move.

*Caveat worth keeping in the doc:* this is the **local** evaluator. A
private harness may score tokens differently, and §1.2's "≤2 LLM calls
per turn" runtime constraint still binds regardless.

---

### DC-3 — §3.4 Step 5: asking is how information is gained, not a cost · **CONFIRMED, and stronger than stated**

§1.2 frames "every question must justify its cost" and §3.4 Step 5 gates
asking behind a threshold. The doc's own boxed "INTERFACE FINDING: ASKING
IS NEARLY FREE" already moves toward the truth, but the evaluator makes it
stronger than "nearly free" — asking is the **only** disclosure channel.

`customer_reply()` (line 166):

```python
if not attribute:
    return "Those options are not quite right yet. Ask me about one specific attribute.", boundary_used
```

With `ask_attribute=None` the simulated user discloses **nothing** — no
new constraint, no new vocabulary. Only a set `ask_attribute` causes
`disclosed.update(matches)` and returns "For that, what matters is: …".

Measured: **3,224 of 4,497 turns (71.7%)** run with `ask_attribute=None`.
Every one of those is a turn on which the agent learned nothing and the
pool could not narrow.

**Correction:** §3.4 Step 5's cost framing is inverted for this evaluator.
Not asking is the expensive move. Combined with D-6's cap, the current
policy spends 71.7% of turns in a state where no information can arrive.

---

### DC-4 — §6.1's MTTC definition disagrees with the evaluator · **NEW, found in this audit**

§6.1: "MTTC = (1/N) Σ turns_to_conversion. Lower is better. **Sessions
exceeding 10 turns terminate with zero score.**"

The evaluator (`metric_summary()`, line 193):

```python
mttc = statistics.fmean(
    item["first_hit_turn"] if item["first_hit_turn"] is not None else MAX_TURNS + 1
    for item in sessions
)
```

A non-converging session contributes **11**, not zero — a *penalty* that
raises the mean, not a zero that lowers it. Since Efficiency = (11 − MTTC)
/ 10, a fully non-converging run yields Efficiency = 0.0 exactly. The
doc's "zero score" phrasing describes the *end state* but inverts the
*mechanism*, and our own `evaluate.py:86` implemented the doc's version
(scoring 0.0 for unconverged), making our internal MTTC **not comparable**
to the official one.

**Correction:** §6.1 should state that unconverged sessions contribute
`MAX_TURNS + 1 = 11` to the MTTC mean.

---

### DC-5 — §3.4 Step 5's `ask_attribute` vocabulary excludes `department` · **NEW, confirms a code decision**

§3.4 Step 5's worked example ranks "category 2.84, **department 1.67**,
brand 0.73", implying department is askable. The evaluator's
`ALLOWED_ATTRIBUTES` (line 17) has no `department` — an out-of-vocabulary
attribute is silently coerced to `"other"` (line 172).

`clarify.py`'s `ANSWERABILITY_PRIOR` correctly excludes `department`
already, and its comment says so. **No code change needed** — recording it
because the doc's example would mislead a reader into adding it.

---

### DC-6 — §6.6's "~4,200 rows" and "~20 negatives per session" · **Doc is out of date**

§6.6 step 4: "Sample ~20 negatives **per session**"; step 5: "fit on the
resulting ~4,200 rows". §3.4 Step 6 repeats "fitted on roughly 4,200 rows".

The implementation samples ~20 negatives **per turn** (`telemetry.py`
`build_training_rows`, correctly — features change as slots accumulate, so
turn 1 and turn 4 are different rows), producing **24,000 rows over 200
sessions** in the last real run, not 4,200.

**Correct or bug:** the code is right and the doc's arithmetic assumed one
row-set per session. §9's "Effective sample size is ~200, not 4,200" still
holds and is if anything more important at 24,000 rows — worth restating
as "~200, not 24,000".

---

## 3. VERIFY THESE SPECIFICALLY

All seven checked against the real catalogue and live code.

### V-1 — `price_fit` returns a neutral value for `None`, not 0.0 · ✅ **PASS**

`features.py:132,155-156`. `PRICE_FIT_NEUTRAL = 0.5`, returned whenever
the facts record is missing or `price is None`. Verified live:
`price_fit(null price) = 0.5`. The constant is named (not an inline
`0.5`) with a comment explaining that a strictly-interior value keeps the
78.9% of null-price rows uninformative rather than penalised — the §2.2
invariant holds.

⚠️ The **priced** branch is a stub — see **D-3**.

### V-2 — `slot_coverage` substring-matches the text blob, not structured fields · ✅ **PASS**

`features.py:243-245`. Reads `facts[asin]["blob"]` (which is
`product_text(row).lower()`, `indexes.py:361`), never `details.Color` /
`details.Material`. Verified live: with `color='red'`, `material='cotton'`
present **only** in the blob and the structured fields `None`,
`slot_coverage = 1.0`. §2.3's finding (Color 4.9%, Material 4.1%
populated) is respected.

⚠️ Price slots leak into the term list — see **D-4**.

### V-3 — `rare_tag_match` covers only warmth / weather / performance · ✅ **PASS**

`state.py:168` — `RARE_TAGS = ("performance", "warmth", "weather")`.
`derive_profile_terms()` (line 190) filters the profile's tags to that
tuple, and `features.py:270-274` checks only `state.profile_terms`. The
five tags §2.4 measured at or below chance (fit, material, comfort, style,
durability) can never reach the feature. Exactly §2.4's three
positive-lift tags (+11.6, +11.6, +9.6), no more.

### V-4 — `rating_style_fit` reads `rating_style`, not `average_prior_rating` · ✅ **PASS**

`features.py:315` reads `getattr(state, "rating_style", None)` and
compares against `RATING_STYLE_EXPECTS_HIGH = "usually positive"`.
`state.py:288` populates it from `user_profile.get("rating_style")`
verbatim. `average_prior_rating` appears **nowhere** in `features.py` or
`rank.py` — grep-verified. §2.4.1's collinearity hazard ("only one may be
used") is respected, and `state.py:224` documents the deliberate choice.

### V-5 — FTS5 `bm25()` sign is flipped exactly once at the retrieval boundary · ✅ **PASS**

`indexes.py:168` — `return [(asin, -score) for asin, score in cursor.fetchall()]`.
Verified live on the same query:

```
raw sqlite bm25()  : [('B00FIXTURE1', -0.9848610811622407)]   more negative = better
keyword_search()   : [('B00FIXTURE1',  0.9848610811622407)]   higher = better
```

Flipped once, at the boundary, as §3.2 Index 1 specifies. No second flip
downstream: `features.py:63-66` (`bm25_norm`) divides by the pool max and
assumes higher-is-better, consistent. The `ORDER BY score` in the SQL
(line 164) correctly orders on the *raw* value (ascending = best first)
before the flip.

### V-6 — The browsing track applies NO filters · ✅ **PASS**

- `keyword_stream` (`retrieval.py:114`): filter block is guarded by
  `if state.track == "buy" and DEPARTMENT_FILTER_ENABLED:` — unreachable
  on browse.
- `semantic_stream`: no facts-based rejection at all. It applies a
  per-department **diversity cap** on browse
  (`CATEGORY_DIVERSITY_MAX_SHARE = 0.3`, line 51), which is *not* a
  filter: capped candidates go to a `leftover` list and are used to refill
  the quota if the cap undersupplies, so no candidate is removed from
  consideration for a query-relevance reason. Consistent with §3.4 Step
  4's "MMR or per-category caps applied on the browsing track".
- `popularity_stream`: ignores the query entirely, as specified.

§3.4 Step 4's "WHY BROWSING NEVER FILTERS" holds.

### V-7 — The department filter whitelists Women/Men/Girls/Boys/Baby and ignores junk · ❌ **FAIL**

**There is no whitelist.** See **D-2** for the full finding. Summary:
`retrieval.py:120` does bare case-insensitive equality against
`facts[asin]['dept']` = `categories[1]`, which holds **203 distinct
values**, of which **19.7% of rows (9,845)** are not a department —
including `Westlake` (1,136 rows) and `Boot Shop` (1,131 rows) by name.
`extract.py`'s promo-noise filter does not reject them, and the gazetteer
resolves `westlake → 'Westlake'`. **30 of 200 ground-truth targets (15%)**
live under a non-canonical department and are exposed to irrecoverable
deletion on the buying track.

---

---

## 4. POST-AUDIT MEASUREMENT (added after the report above)

The audit was written before instrumenting pool recall under the
evaluator's own phrasing. That measurement (§6.2's decisive diagnostic)
substantially reorders the priorities, and one prediction in the report
was wrong.

### Pool recall under evaluator phrasing — 200 sessions

| | |
|---|---|
| target ever in pool | **148 (74.0%)** |
| Hit Rate@10 | 116 (58.0%) |
| in pool but ranked >10 | **32** — ranking loss |
| never retrieved at all | **52** — retrieval loss |

| scenario | pool recall | hit@10 | ranking loss |
|---|---:|---:|---:|
| intent_override | **0.900** | 0.633 | **0.267** |
| boundary | 0.700 | 0.500 | 0.200 |
| buying | 0.775 | 0.637 | 0.138 |
| browsing | **0.650** | 0.512 | 0.138 |

### N-1 — The buying track runs 98.3% of the time · **BUG, new, not in the audit above**

Measured track split across all turns: **`buy` 1,043 / `browse` 18**.

`state.py:pick_track()` returns `"buy"` whenever a specific category slot
is set. `evaluator.py:initial_message()` (line 154) opens *every* session
— browsing included — with `"I'm looking for {category}…"`, where
`coarse_category()` derives that phrase from the target's own category
path. So the category slot is set on turn 1 of essentially every session
and the browse track is effectively unreachable.

Consequence: the 80 browsing sessions, which §3.4 Step 4 says must
**never** filter, are hard-filtered — on a department value drawn from the
203-value junk-laden `categories[1]` field (**D-2**). This is §1's named
failure mode verbatim: "applying a filter to an exploratory query deletes
the right answer on turn one." Browsing has the lowest pool recall (0.650)
of any slice, consistent with this.

**D-2 and N-1 compound**: the junk-department filter is not an edge case
gated behind rare buying turns, it is active on ~98% of all turns.

### Correction to the audit's own priority call

The report above ranked **D-1** (override phrasing) first and I then
verbally downgraded it on the grounds that `intent_override` scored 0.633
— its second-best slice. **That reasoning was wrong.** Measured against
what is *achievable*, intent_override is the **worst** slice: pool recall
0.900 against hit@10 0.633 is a **0.267 ranking loss**, the largest of
any scenario. The targets are being retrieved and then discarded by the
ranker — exactly the signature of an undeleted stale constraint polluting
the canonical intent. D-1 is real and belongs near the top.

### Experiment run: clarification policy (DC-3 + D-6) — **CONFIRMED, large win**

Changed two constants in `clarify.py` (`ASK_THRESHOLD` 1.0 → 0.15,
`MAX_CLARIFICATIONS_PER_SESSION` 3 → 10) and re-ran the full evaluator:

| metric | before | after | delta |
|---|---:|---:|---:|
| Hit Rate@10 | 0.580 | **0.750** | **+0.170** |
| MRR | 0.302 | 0.384 | +0.083 |
| MTTC | 5.73 | 4.74 | −0.99 (better) |
| Efficiency | 0.527 | 0.626 | +0.099 |
| **Technical score** | **0.486** | **0.616** | **+0.130** |

Per-session: **35 miss→hit, 1 hit→miss, net +34**. Mean rank among hits
unchanged (1.95 → 1.95), so this is pure recall gain, not a ranking
trade. Browsing gained most (0.512 → 0.762, +0.250).

A risk flagged before the run — that more disclosed slots would mean more
filtering on the near-universal buy track, and therefore *more* deletions
— **did not materialise**. The information gain outweighed it. Note this
also means the filter's damage (D-2 / N-1) is still unaddressed and its
fix is additive to this gain, not redundant with it.

---

## Suggested priority

Revised after the §4 measurements. Score at time of writing: **0.616**
(Hit@10 0.750, MRR 0.384), up from 0.486.

| | Finding | Status | Why |
|---|---|---|---|
| — | **DC-3 + D-6** asking policy | ✅ **done, +0.130** | 71.7% of turns disclosed nothing; two constants |
| 1 | **N-1 + D-2/V-7** track & department filter | open | 98.3% buy track means the junk-department filter runs on ~every turn, including all 80 browsing sessions that must never filter. Targets the **52 never-retrieved** sessions |
| 2 | **D-1** override phrasing | open | intent_override has the worst ranking loss of any slice (0.267 against 0.900 pool recall). Targets the **32 in-pool-but-unranked** |
| 3 | **D-7** brand over-asking | open | Now more impactful, since we ask far more often; grid-searchable (E7) |
| 4 | **D-3 / D-4** feature stubs | open | Latent; near-zero score impact, but they pollute the ablation table's interpretability |
| 5 | Doc edits DC-1…DC-6, N-1 | open | No runtime effect |

Items 1 and 2 partition the remaining 84→50 misses almost exactly:
retrieval loss (52 sessions) and ranking loss (32 sessions) respectively.
Neither was visible from the aggregate metrics alone — both required the
pool-recall instrumentation in §4.
