# The Pipeline, End to End

How one shopper utterance becomes ten ranked ASINs.

This document traces the actual code path, not the design intent. Section
markers such as §3.4 in the source refer to the team's internal technical
design document, which is not distributed with this repository; where the
implementation diverges from it, the divergence is stated here. Every claim
below is anchored to a `file:line` you can open.

---

## 0. The contract we are held to

The evaluator (`evaluator/evaluator.py`) owns the loop. It constructs one
`Agent`, then for each session calls `reset()` once and `respond()` up to ten
times. Our agent never drives the conversation; it only answers.

```python
Agent(catalog_path: str | Path = "data/catalog.jsonl") -> None
reset(session_id: str, user_profile: dict) -> None
respond(session_id: str, user_message: str, turn: int, top_k: int) -> dict
```

These signatures are frozen to match the kit's baseline agent exactly — no
added, removed, or reordered parameters (`agents/our_agent.py:1-30`). Two
consequences shape the whole design:

1. **The harness owns `turn`.** We store what we are given; we never increment
   a counter of our own.
2. **One `Agent` serves many sessions.** State is a `session_id -> SessionState`
   map (`agents/our_agent.py:106`), not a single field.

`respond()` must return:

```python
{
  "message": str,                                  # free text back to the shopper
  "ask_attribute": str | None,                     # optional clarifying question
  "recommendations": [{"parent_asin": str}, ...],  # <= top_k
  "usage": dict,                                   # token accounting
}
```

`ask_attribute` and `recommendations` occupy the **same** return object. The
decision is therefore never "ask *or* answer" — it is "answer, and additionally
ask when a question would earn its keep" (`clarify.py:264-266`).

**What we are scored on** (`evaluator/evaluator.py:279-280`):

```
efficiency      = clamp((11 - MTTC) / 10, 0, 1)
technical_score = 0.50 * Hit@10 + 0.30 * MRR + 0.20 * efficiency
```

Hit@10 dominates, which is why every retrieval decision below biases toward
recall — an extra stream carries no recall risk, a filter always does.

---

## 1. Offline: what exists before the first word

`Agent.__init__` runs once per process, in roughly five seconds plus encoding
time (`agents/our_agent.py:64-106`). It loads `data/catalog.jsonl` (50,000
products) and builds five things.

### The four indexes (`indexes.py:27-53`)

| Index | Built by | Shape | Serves |
|---|---|---|---|
| **FTS5 table** | `build_fts5_index()` `indexes.py:66` | in-memory SQLite, 6 indexed columns | keyword stream |
| **Embedding matrix** | `build_embedding_matrix()` `indexes.py:216` | `(50000, 384)` float32, L2-normalised | semantic stream |
| **Facts dict** | `build_facts_dict()` `indexes.py:309` | ASIN -> dept, cat3, store, price, rating, pop, text blob | ranking features, clarification entropy |
| **Category lists** | `build_category_lists()` `indexes.py:387` | category path -> ASINs, pre-sorted by `rating_number` desc | popularity stream |

The FTS5 table is populated from the **raw fields**, one column each, so every
column can carry its own bm25 weight (`indexes.py:55-61`):

```
title 6.0 | categories 4.0 | features 2.5 | details 2.5 | store 1.5 | description 1.0
```

The embedding matrix and the facts blob instead consume `product_text()`
(`utils.py:140-165`) — one flat string per product, no field markers, because
an embedding has no notion of columns. Two representations of the same
catalogue, each shaped for its consumer.

Embeddings come from `sentence-transformers/all-MiniLM-L6-v2`, 384-d, used
frozen (`indexes.py:176`), cached to `data/embeddings.npy`. The same model
later encodes the query, so products and intent occupy one space.

### The attribute gazetteer (`extract.py:202`)

`build_attribute_gazetteer()` mines the catalogue for the legal values of each
slot — the actual brands, colours, materials, categories that exist. It
deliberately does **not** call `product_text()`: tokenising a flat blob would
turn arbitrary description words into hard constraints (`extract.py:211-214`).
Values are filtered for safety and promo noise before entering.

### The fitted ranker (`agents/our_agent.py:100-105`)

Loaded from `models/ranker.json` if present; falls back to `HANDSET_WEIGHTS`
if not. It is **not** a constructor parameter, because the frozen baseline
contract has no such parameter. A clean clone that has never run
`scripts/fit_ranker.py` still works — just worse (0.641 vs 0.696).

### Session start

`reset()` calls `init_state()` (`state.py:265`), which distils the raw
`user_profile` down to two read-only fields:

- `profile_terms` — `derive_profile_terms()` (`state.py:171`) keeps only the
  three tags with measured lift: `("performance", "warmth", "weather")`.
  Everything else sits near chance and is discarded.
- `rating_style` — carried verbatim, for one ranking feature.

Both are **loaded once and never written again**. The governing invariant
(`state.py:10-12`): *inferred preference never merges with what the shopper
actually said.* Profile signal reaches the ranker as features; it never
becomes a slot, and never enters the query string.

---

## 2. The per-turn pipeline

`Agent.respond()` (`agents/our_agent.py:124-188`) executes seven steps in a
fixed order. Nothing loops, nothing branches on turn number.

```
  user_message
       |
       v
  [1] update_slots()          extract.py     -- what did they say?
       |                                        writes / overwrites / deletes
       v
  [2] reconstruct_canonical() state.py       -- rebuild the query from scratch
       |                                        render() + MiniLM embed
       v
  [3] pick_track()            state.py       -- buy or browse
       |
       v
  [4] retrieve()              retrieval.py   -- 3 streams -> union -> floor check
       |                                        pool of ~180-230 candidates
       +--------------------------+
       |                          |
       v                          v
  [5] pick_attribute()        [6] rank()     -- entropy x answerability
      clarify.py                  rank.py       10 features -> logistic score
       |                          |             sort -> truncate 30 -> top 10
       +------------+-------------+
                    |
                    v
              [7] log_turn()    telemetry.py -- append-only JSONL, feeds training
                    |
                    v
      {message, ask_attribute, recommendations[10], usage}
```

---

### Step 1 — Slot extraction and state mutation

**`update_slots(state, user_message, gazetteer)`** — `extract.py:1956`

The slot dictionary is the agent's only memory of what the shopper literally
said. Twelve keys, in two classes (`state.py:36-51`):

| Single-value | Multi-value |
|---|---|
| `department`, `category`, `brand`, `price_min`, `price_max`, `price_target` | `color`, `material`, `style`, `size`, `feature`, `use_case` |

A single-value slot is overwritten on re-statement; a multi-value slot
accumulates. The turn proceeds in five sub-stages:

1. **Consume the pending clarification.** If last turn asked "what colour?",
   `consume_pending_clarification()` returns `"color"`, and extraction runs in
   *clarification context* — so a bare reply of "blue" is routed to the right
   slot instead of through the generic, context-blind parser
   (`agents/our_agent.py:163-168`). Without this wiring the entire
   `pending_clarification` mechanism would be dead code.

2. **Extract findings.** `extract_slots()` (`extract.py:1115`) runs a
   deterministic cascade over the utterance: budget patterns, size patterns,
   gazetteer phrase matches (with context guards so "North Face" isn't read as
   a use case, and a bare brand token isn't claimed without syntactic
   support), controlled-vocabulary style values, requirement classification,
   and use-case detection. Whatever text survives unclaimed becomes the
   **residual scenario**. An LLM path exists (`extract_slots_llm()`,
   `extract.py:1081`) but `USE_LLM_EXTRACTION = False` and it always falls
   back — there is no model access in this environment.

3. **Plan operations.** `detect_slot_operations()` (`extract.py:1676`) reads
   the utterance *and the pre-transition state* and emits typed transitions
   rather than mutating anything:

   ```
   upsert | replace | delete_value | delete_slot
   ```

   This is where "actually not black, blue" becomes
   `delete_value(color, "black")` + `upsert(color, "blue")` rather than a
   colour list that quietly grows to `("black", "blue")`. Negation scope,
   ambiguous negation, and whole-slot rejection each have their own guard
   (`extract.py:1432-1500`).

4. **Apply them.** `apply_slot_operations()` (`extract.py:1311`) executes the
   plan. Values are canonicalised and de-duplicated by normalised identity, so
   "Black" and "black" cannot both occupy the slot.

5. **Update the scenario buffer.** `update_scenario_buffer()`
   (`extract.py:1911`) independently decides whether this turn replaces,
   extends, or clears the un-slotted intent ("for a beach trip"). An anchored
   rejection — "forget the honeymoon part" — clears it.

**Why the deletion actually matters:** because the query is rebuilt from the
dictionary in Step 2, and the dictionary no longer holds `black`, the rebuilt
query *cannot* contain `black`. Correctness follows from the data structure
rather than from remembering to subtract.

---

### Step 2 — Canonical reconstruction

**`reconstruct_canonical(state, embed_text)`** — `state.py:398`

Every turn, from scratch:

```python
state.canonical_intent = render(state)
state.canonical_vector = embed_text(state.canonical_intent)
```

`render()` (`state.py:349`) walks `CANONICAL_SLOT_ORDER` — a fixed
retrieval-facing order, *not* insertion order — and emits labelled clauses,
then price, then the scenario buffer:

```
department: Men; category: Jackets; color: blue; features: waterproof; under $150; for a beach trip
```

Price semantics stay distinct: `at least $X` / `under $X` / `between $X and $Y`
/ `around $X` (`state.py:376-390`). An empty state renders `""`, and the
vector is then `None`.

Only `slots` and `scenario_buffer` are read. Not history, not the raw
utterance, not the profile. The string is logged verbatim every turn, which
makes override behaviour directly auditable in the telemetry.

---

### Step 3 — Track routing

**`pick_track(state)`** — `state.py:427`

One rule:

```python
track = "buy" if (category slot is non-empty and not in BROAD_CATEGORY_VALUES) else "browse"
```

`BROAD_CATEGORY_VALUES` (`state.py:145`) holds the catalogue's non-discriminating
nodes — `root`, `clothing, shoes & jewelry`, `men`, `women`, `kids`. A category
outside that set is treated as specific enough to authorise the restrictive
path.

The result is derived fresh and replaces `state.track` every call. No prior
track, scenario text, profile, or transition metadata participates — routing is
memoryless by construction.

> **Honest caveat.** The evaluator opens every session with "I'm looking for
> {category}…", so this returns `"buy"` on **98.3% of turns (1,043 vs 18)**
> (measured, `retrieval.py:92-96`). The two-track design is real in code but
> near-degenerate in practice on this benchmark.

---

### Step 4 — Multi-stream retrieval

**`retrieve(state, track, indexes)`** — `retrieval.py:456`

Three independent streams run against the same canonical intent, each with a
per-track quota (`retrieval.py:30`):

| Track | keyword | semantic | popularity |
|---|---|---|---|
| `buy` | 120 | 40 | 20 |
| `browse` | 60 | **150** | 20 |

The asymmetry *is* the routing: a shopper who named a category wants lexical
precision; a shopper describing a situation needs the embedding to do the work.

**Keyword stream** (`retrieval.py:184`) — FTS5 `MATCH` with column-weighted
bm25, over-fetching `quota * KEYWORD_OVERFETCH` (3x) before any filtering, so
post-filter truncation can never leave an undersized pool. The buy-track
department/category filter exists but is **off**
(`DEPARTMENT_FILTER_ENABLED = False`, `retrieval.py:101`). It was measured over
all 200 public sessions and only hurts:

```
metric            filter ON   filter OFF
Hit Rate@10          0.750       0.790
MRR                  0.3845      0.3860
MTTC                 4.74        4.49
technical score      0.6155      0.6410
```

The mechanism is a data defect, not a logic bug: the filter compares the stated
department against `categories[1]`, which is 100% populated but only ~80%
meaningful — buckets like "Boot Shop" and "Novelty & More" occupy the rest, so
a *correct* filter for "Men" deletes a men's boot for not being labelled "Men".
30 of the 200 public targets (15%) sit under such a value. The code path is
retained deliberately so the ablation table in the writeup has both arms.

**Semantic stream** (`retrieval.py:265`) — brute-force exact kNN:
`matrix @ query_vec`, then `argpartition` for top-k, ~5 ms over 50k rows
(`indexes.py:276`). No FAISS, no HNSW, no vector database — exact search at
this scale is both compliant and fast enough. This stream **never filters, on
either track**. On `browse` only, it over-fetches 3x and applies a diversity
cap: no single department may exceed `floor(quota * 0.3)` of the selections,
with leftovers back-filled in similarity order if the cap underfills the quota
(`retrieval.py:302-327`).

**Popularity stream** (`retrieval.py:330`) — ignores the query entirely.
Walks `category_lists` (pre-sorted by `rating_number` desc) for up to `quota`
ASINs not already pooled. This is the recall backstop: it cannot be wrong about
the query because it never reads it.

**Union and floor check** (`retrieval.py:382`, `retrieval.py:412`) —
`union_dedupe()` merges the streams into one candidate per ASIN, keeping each
stream's raw contribution (`bm25_raw`, `cos_raw`) and unioning `sources`.
Order is insertion order; no re-scoring happens here. Then `floor_check()`:
if the pool is under `POOL_FLOOR = 50`, popularity is re-invoked to top it up,
and a placeholder guarantees the pool is never empty so downstream consumers
always have a row.

A fourth, labels-free FTS source (`clean_keyword_stream`, `retrieval.py:234`)
is implemented but **off** (`CLEAN_KEYWORD_ENABLED = False`): on the paired
200-session run it recovered four misses but displaced three existing top-10
targets — kept for ablation, not made default.

**The union is the whole recall argument.** A candidate is lost only if all
three streams miss it simultaneously, and the three fail for uncorrelated
reasons — wrong words, wrong embedding neighbourhood, unpopular.

Pool out: up to 180 candidates on the buy track and 230 on browse (quota sums,
before de-duplication), as `Candidate` objects (`utils.py:28`), each carrying
`asin`, `bm25_raw`, `cos_raw`, `sources`.

---

### Step 5 — Clarification decision

**`pick_attribute(pool, state, indexes=...)`** — `clarify.py:255`

For each attribute the shopper has not already filled and we have not already
asked (`clarify.py:287-291`):

```
score(a) = answerability_prior(a) x H(a)          # clarify.py:252
H(a)     = -Sum_v p(v) log2 p(v)                  # over the attribute's values
                                                   # across the *current pool*
```

Entropy measures how much the pool disagrees about the attribute — asking about
something every candidate shares teaches nothing. The hand-set answerability
prior (`clarify.py:53`) discounts questions shoppers cannot answer well:

```
category 0.9 | style 0.6 | color 0.5 | material 0.5 | feature 0.45 | use_case 0.45 | size 0.4 | budget 0.4 | brand 0.3
```

The argmax is asked if it clears `ASK_THRESHOLD = 0.15` (`clarify.py:79`).
Three attributes — `brand`, `budget`, `size` — are **deferred**
(`clarify.py:97`): their entropy still counts, but they lose to any
non-deferred attribute that also clears the bar, because they produced no
informative replies in the transcript. If nothing structured clears the
threshold, `"other"` is asked once as a broad fallback; its reply is still
parsed into real slots and never creates an `other` field.

Two threshold decisions were tuned against the evaluator's measured behaviour
rather than the design doc: `ASK_THRESHOLD` was lowered from 1.0, and
`MAX_CLARIFICATIONS_PER_SESSION` was raised from 3 to 10 (`clarify.py:90`) —
the cap exists to protect MTTC, but a silent turn cannot converge either, so
capping questions was costing more than it saved.

When an attribute is chosen, `set_pending_clarification()` records it so the
*next* turn's extraction runs in clarification context (Step 1, sub-stage 1).

Note the vocabulary split: `pick_attribute` returns the **evaluator-facing**
`ClarificationAttribute` (`state.py:62`), not the internal `SlotKey`. So
`budget` maps to three internal price slots, and `department` — useful
internally — can never be asked at all.

---

### Step 6 — Feature extraction and ranking

**`rank(pool, state, indexes, ranker, top_k)`** — `rank.py:369`

Every candidate in the pool gets an eleven-dimensional feature vector
(`features.py:36`, extracted at `features.py:503`):

| # | Feature | What it reads | Source |
|---|---|---|---|
| 1 | `bm25_norm` | `bm25_raw` / pool max — lexical relevance | `features.py:51` |
| 2 | `cos_sim` | dot product with the canonical query vector | `features.py:74` |
| 3 | `pop` | `log1p(rating_number) / log1p(1e5)` — purchase-frequency prior | `features.py:92` |
| 4 | `rating` | `average_rating / 5` | `features.py:112` |
| 5 | `price_fit` | fit to stated budget; `0.5` neutral when price is null | `features.py:140` |
| 6 | `category_match` | whole-word token match against the **full category path** | `features.py:165` |
| 7 | `brand_match` | `store` vs the `brand` slot | `features.py:221` |
| 8 | `department_match` | `dept` vs the `department` slot; `0.5` neutral for a non-department bucket | `features.py:267` |
| 9 | `slot_coverage` | fraction of slot terms present as whole words in the text blob | `features.py:397` |
| 10 | `rare_tag_match` | any of the three profile rare tags present | `features.py:432` |
| 11 | `rating_style_fit` | `rating/5` when the profile expects high ratings, else 0 | `features.py:469` |

Features 1–2 carry retrieval signal forward; 3–4 are catalogue priors; 5–9 are
constraint satisfaction against the slot dictionary; 10–11 are the only place
the user profile touches the pipeline.

Feature 8 is the only one reading a structured catalogue field rather than
text. §2.3 argues attribute matching must operate over text, but measured
that on `details.Color` (4.9%) and `details.Material` (4.1%); `categories[1]`
is 100% populated, so department is the exception. It leans and never
filters — a candidate under one of `categories[1]`'s store buckets
("Boot Shop") scores the 0.5 neutral, not 0, which is what separates it from
the department *filter* Change 2 removed.

Scoring is a plain weighted sum (`rank.py:143-149`):

```python
score = sum(weights[name] * value for name, value in features.items())
```

`weights` comes from the fitted logistic regression (`models/ranker.json`) when
present, else `HANDSET_WEIGHTS` (`rank.py:53`). The fitted model's scaler is
folded into the weights and intercept at fit time (`rank.py:270-274`), so scoring
at inference stays a single dot product with no scaler round-trip.

Then: sort descending, truncate to `TOP_K_TRUNCATE = 30`, return the first
`top_k = 10` (`rank.py:396-406`).

**The LLM rerank does not exist.** `llm_rerank()` (`rank.py:343`) returns
`None` unconditionally and is documented as a deliberate won't-do — no model
access. `use_llm_rerank` defaults to `False` and the `Agent` never passes it.
Grepping the repo for `anthropic|openai|claude|gpt-` returns nothing. The
pipeline is fully functional with zero LLM calls, which was the point.

---

### Step 7 — Telemetry

**`log_turn(...)`** — `telemetry.py:86`

One append-only JSONL row per **turn**, not per session — turn 1 and turn 4 are
genuinely different data because the slots have changed
(`telemetry.py:130-153`):

```json
{"session_id": ..., "turn": ..., "track": ..., "canonical_intent": ...,
 "n_hard_slots": ..., "pool_size": ..., "ask_attribute": ...,
 "feature_names": [...10...],
 "candidates": [{"asin": ..., "sources": [...], "features": [...10 floats...]}]}
```

No labels are written. "Which stream contained the target" is derived offline
by joining `sources` against ground truth, so nothing at inference time ever
touches a label.

---

### The return

```python
{
  "message": f"[STUB reply] turn={turn} track={track} pool_size={len(pool)}",
  "ask_attribute": ask_attribute,
  "recommendations": [{"parent_asin": asin} for asin in ranked_asins],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
}
```

`message` is still a diagnostic stub (`agents/our_agent.py:182`). The evaluator
scores `recommendations` and `ask_attribute`; free text is not part of the
technical score, so it was never worth building without an LLM. `usage` is
honestly zero — we make no model calls.

---

## 3. A worked example

Session opens. Profile: `{"preference_tags": ["comfort", "warmth"], "rating_style": "usually positive"}`.
`reset()` distils this to `profile_terms = ["warmth"]` (comfort is not a rare
tag) and `rating_style = "usually positive"`.

**Turn 1 — "I'm looking for a men's jacket"**

| Stage | Result |
|---|---|
| Extract | `upsert(department, "Men")`, `upsert(category, "Jackets")` |
| Canonical | `"department: Men; category: Jackets"` -> 384-d vector |
| Track | `category` is specific -> **buy** |
| Retrieve | keyword 120 (over-fetch 360, no filter) + semantic 40 + popularity 20 -> union ~170 |
| Clarify | `color` has high pool entropy x 0.5 prior -> clears 0.15 -> **ask "color"**; `pending_clarification = color` |
| Rank | `category_match` and `bm25_norm` dominate; jackets rise |
| Return | 10 ASINs + `ask_attribute: "color"` |

**Turn 2 — "something for cooler weather, black"**

| Stage | Result |
|---|---|
| Extract | pending `color` consumed -> `upsert(color, "black")`; "cooler weather" is unclaimed -> scenario buffer |
| Canonical | `"department: Men; category: Jackets; color: black; for cooler weather"` — **re-embedded from scratch** |
| Track | still **buy** |
| Retrieve | same quotas, new vector -> different semantic neighbourhood |
| Rank | `rare_tag_match` now fires on "warmth" products; `slot_coverage` rises for candidates whose blob contains *black* |
| Return | 10 ASINs, likely a better set |

**Turn 3 — "actually not black, blue"**

| Stage | Result |
|---|---|
| Extract | `detect_slot_operations()` emits `delete_value(color, "black")` **then** `upsert(color, "blue")` |
| Slots | `color = ("blue",)` — black is genuinely gone, not appended to |
| Canonical | `"department: Men; category: Jackets; color: blue; for cooler weather"` |
| Retrieve | the rebuilt string contains no "black", so **no black jacket can be retrieved by the keyword stream at all** |
| Clarify | `color` is now filled and already asked -> excluded; next-best attribute considered |
| Return | 10 blue jackets |

Turn 3 is the design's whole thesis in one step: the correction is enforced by
the data structure, not by a subtraction step someone had to remember to write.

---

## 4. The second pipeline: how the ranker gets its weights

The eleven features are only as good as their weights, and the weights are fitted
offline from our own logged traffic. This is a separate loop, run before
evaluation.

```
python3 -m evaluate            # ~10 min
  |  simulate.py drives a user simulator against the real Agent
  |  telemetry.log_turn() appends every turn's full pool + features
  v
data/telemetry.jsonl
  |  telemetry.build_training_rows()          telemetry.py:189
  |    join session_id -> ground-truth target
  |    label 1 = the target (when it is in that turn's pool)
  |    label 0 = 20 sampled pool negatives PER TURN (see below)
  v
data/features.jsonl        # session_id, turn, n_hard_slots, asin, [10 features], label
  |
python3 scripts/fit_ranker.py
  |  rank.fit_logistic_regression()           rank.py:152
  |    StandardScaler -> LogisticRegression
  |    GroupKFold by session_id (no session spans a fold boundary)
  |    fold the scaler into weights + intercept
  v
models/ranker.json
  |
make evaluate    # Agent.__init__ loads it -> results/output.json
```

### Negative sampling: per turn, not per session

`build_training_rows()` (`telemetry.py:189`) loops over **turns**, drawing a
fresh 20 pool negatives inside each one. The current corpus:

```
rows 23,012   positives 632   negatives 22,380
sessions 200   session-turns 1,119   rows per session-turn 20.56
```

**This deviates from the design doc.** §6.6 step 4 says "Sample ~20 negatives
*per session* from the candidate pool", and step 5's "~4,200 rows" is exactly
200 sessions x 21 rows — a figure the doc repeats in §3.4 Step 6, §2.4 and
§8.4. Sampling per turn produces a corpus **5.5x larger** (sessions average
5.6 turns).

The deviation is deliberate and defensible — turn 1 and turn 4 have genuinely
different feature values because the slots have changed, and 4,200 rows for
eleven parameters is thin — but it is a deviation, and the justifying comment
at `telemetry.py:59-60` cites a §8.4 STEP 3 sentence ("Sample ~20 negatives
PER TURN") that **does not appear in the design document**. The only "PER TURN"
in the PDF is a band label in the §4 architecture diagram, which marks the
runtime pipeline, not the sampling protocol. Treat the doc as the outdated
artefact here, not the code — but fix the citation.

### Reproducibility of the sampling

There is no global seed. `_seeded_rng()` (`telemetry.py:167`) builds a fresh
`random.Random` keyed on a **string derived from the sampling site**:

```python
random.Random(f"{session_id}|{turn}")     # telemetry.py:227
```

Sampling is therefore content-addressed rather than globally seeded: a re-run
draws identical negatives, and re-running one session cannot perturb the
negatives drawn for any other. No integer seed exists to report.

### Fit hyperparameters

`rank.py:230` is the complete model specification:

```python
LogisticRegression(class_weight="balanced", penalty="l2")
```

Everything else is an sklearn default. Stated explicitly, since "we used
logistic regression" is not a reproducible claim (values verified against
sklearn 1.4.2):

| Parameter | Value | Chosen or inherited |
|---|---|---|
| folds | `min(5, n_distinct_sessions)` → **5** | explicit, `rank.py:241` |
| grouping | `GroupKFold` on `session_id` | explicit, `rank.py:242` |
| `class_weight` | `"balanced"` | explicit (632 vs 22,380) |
| `penalty` | `"l2"` | explicit |
| `C` | **1.0** | sklearn default |
| `solver` | **lbfgs** | sklearn default |
| `max_iter` | **100** | sklearn default |
| `tol` | 1e-4 | sklearn default |
| `random_state` | `None` | sklearn default (lbfgs is deterministic) |

`max_iter=100` does not bind in practice: the fit converges in **17
iterations** on the real corpus, with no `ConvergenceWarning`.

### What the folds are for

**They are report-only, and nothing is selected on them.** The five fold models
are fit, scored, and discarded; the model that ships is refit on *every* row
(`rank.py:266`). Step 5 of the protocol validates the approach — it does not
choose which fold's model deploys, gate whether the fit is good enough to
persist, or tune anything. The held-out figures are printed by
`scripts/fit_ranker.py` and persisted onto `FittedRanker` for inspection; no
inference path reads them (`rank()` touches only `.weights`).

The grouping does matter for the figures being honest: turns from one session
share a query and are not independent, so a plain random split would leak and
inflate them (`rank.py:185-187`).

**Read AUC and AP, not accuracy.** All three are reported:

```
ROC-AUC           : 0.9769   (chance 0.5)
average precision : 0.6296   (chance 0.0275)
accuracy          : 0.9241   (majority-class baseline 0.9725 - do not quote this one)
```

Accuracy is the wrong family of metric here. With 632 positives against 22,380
negatives, always predicting "not the target" scores 0.973 — so the fitted
model's 0.924 lands *below* the do-nothing baseline and reads as "worse than
useless" for a model that in fact ranks well. Ranking is what happens at
inference, so the folds are scored as ranking: **AUC 0.977**, and average
precision of **0.63 against a chance rate of 0.0275** — a 23x lift. That is
independent corroboration of the 0.641 → 0.684 evaluator movement, arrived at
without touching the evaluator.

`cv_accuracy` is retained for continuity with §6.6 and because it was already
persisted, but `FittedRanker`'s docstring marks it do-not-quote
(`rank.py:89-100`).

This loop is why the reproduction order in the README is not optional — a clean
clone that skips it evaluates with `HANDSET_WEIGHTS` and scores ~0.641 instead
of ~0.696.

### Measured progression

| Milestone | Score |
|---|---|
| kit baseline agent | 0.107 |
| our agent, pre-refit | 0.641 |
| ranker refit on a corpus matching the current agent | 0.684 |
| `category_match` against the full category path | 0.696 |
| extraction and state repair | 0.698 |
| clarification answerability and question order (current) | **0.777** |

Per-change detail, ablations and the reproducibility note live in
[`changes.md`](changes.md); that file is the authority if these two disagree.

---

## 5. What is deliberately not in the pipeline

Being explicit about this is cheaper than a reviewer finding it.

| Thing | Status | Where | Why |
|---|---|---|---|
| LLM rerank | stub returning `None` | `rank.py:343` | no model access; closed as won't-do |
| LLM slot extraction | flag off, always falls back | `extract.py:1078` | same |
| Department/category filter | `False` | `retrieval.py:101` | measured: costs 0.025 score, helps nothing |
| Labels-free FTS stream | `False` | `retrieval.py:107` | +4 recovered, -3 displaced; net negative |
| Slot decay over time | **not implemented** | — | slots change only by explicit user action |
| Runtime-adaptive memory | **not implemented** | — | profile is distilled once at `reset()`, read-only |
| Agent free-text `message` | diagnostic stub | `agents/our_agent.py:182` | not scored, needs an LLM to be worth building |

The two "not implemented" rows are design positions, not omissions. Slot
lifetime is event-driven — write, overwrite, delete — because a shopper's
stated constraint does not become less true after three turns; it becomes
false only when they say so. And profile signal stays read-only because the
moment inferred preference can write into the slot dictionary, the canonical
string stops being a faithful record of what the shopper said, which is the
one property the override guarantee rests on.

---

## 6. Invariants worth defending

1. **The canonical string is rebuilt, never edited.** Every turn calls
   `render()` from scratch. There is no code path that mutates
   `canonical_intent` incrementally, which is why a deleted slot value cannot
   survive in the query.
2. **Inferred preference never becomes stated constraint.** `profile_terms`
   and `rating_style` reach the ranker as features 9 and 10 and reach nothing
   else.
3. **The pool is never empty.** `floor_check()` guarantees at least one row, so
   ranking and clarification always have something to operate on.
4. **No label is readable at inference.** `session_id` is joined to ground
   truth offline only; `log_turn()` writes no label.
5. **The `respond()` signature is frozen.** Ablation flags are module-level
   toggles (`retrieval.DEPARTMENT_FILTER_ENABLED`, `STREAM_QUOTAS`) precisely
   because they cannot be threaded through a parameter.
6. **Routing is memoryless.** `pick_track()` re-derives from current slots
   every turn; no prior track participates.

---

## 7. Module map

| Module | Owns | Pipeline step |
|---|---|---|
| `agents/our_agent.py` | `Agent` — `reset()` / `respond()` orchestration | all |
| `indexes.py` | FTS5, embedding matrix, facts dict, category lists | offline |
| `utils.py` | `product_text()`, the shared `Candidate` shape | offline |
| `extract.py` | gazetteer, slot extraction, negation, merge policy | 1 |
| `state.py` | slot dict, scenario buffer, canonical render, routing | 1-3 |
| `retrieval.py` | three streams, union, floor check | 4 |
| `clarify.py` | entropy x answerability clarification policy | 5 |
| `features.py` | the eleven ranking features | 6 |
| `rank.py` | scoring, logistic-regression fit, LLM rerank stub | 6 |
| `telemetry.py` | append-only JSONL logging, training corpus | 7 |
| `simulate.py` | user simulator, for corpus generation | offline |
| `evaluate.py` | our own Hit@10 / MRR / MTTC scorer | offline |
| `ablate.py` | ablation harness, scenario slicing | offline |
| `scripts/fit_ranker.py` | fits the ranker on the logged feature matrix | offline |
| `scripts/report_ranker.py` | fitted weights, correlations, near-zero flags | offline |

See also: [`README.md`](../README.md) for setup and reproduction, and
[`changes.md`](changes.md) for the full run log and per-change measurements.
