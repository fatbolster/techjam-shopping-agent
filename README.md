# Shopping Copilot

Conversational search over a 50,000-product catalogue. Symbolic state for
what the user said, vector state for what the user meant.

Full design rationale, data measurements, and the delivery plan live in
[`docs/Shopping-Copilot-Technical-Design.pdf`](docs/Shopping-Copilot-Technical-Design.pdf).
This README is the assembled, per-owner summary (§7.1, §8.5.1: "each owner
writes their own README section ... Marcus merges").

**Status:** end to end and real on the full 50,000-row catalogue —
`agent.py` runs against real data, not just fixtures. Remaining gaps:
the organizer evaluator/participant kit itself (D1/E3 use our own scorer
until it lands, see `kit/README.md`), the ablation harness (E5-E7), C8's
coefficient report, and B10's transcript-sourced regression tests. See
each file's module docstring for what it does and does not implement, and
the per-owner sections below for specifics.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 agent.py   # smoke test: three fixture turns, printed responses
```

Real data (`catalog.jsonl`, `public_set.jsonl`, the embedding matrix) is
never committed (`.gitignore` excludes `*.jsonl`/`*.npy`/`*.db`). It ships
with the organizer-supplied kit (see [`kit/README.md`](kit/README.md)) —
copy `catalog.jsonl`/`public_set.jsonl` into `data/`, then run `make data`
(or `python3 scripts/check_data.py`) to verify a clean clone has what it
needs (§8.0, §8.5 step E9).

## Architecture at a glance

Two representations of the conversation (a slot dictionary for what was
said, a canonical embedding for what was meant), three retrieval streams
(keyword, semantic, popularity) unioned into one pool, one ranking stage.
See §3-§4 of the design doc for the full rationale and system diagram.

| Module | Owns | Design doc |
|---|---|---|
| `utils.py` | `product_text()`, the shared `Candidate` shape | §3.2, §7.2 |
| `indexes.py` | FTS5, embedding matrix, facts dict, category lists | §3.2 |
| `retrieval.py` | Three streams, union, floor check | §3.4 Step 4 |
| `state.py` | Slot dict, scenario buffer, canonical render, routing | §3.3, §3.4 Steps 2-3 |
| `extract.py` | Slot extraction, negation, merge policy | §3.4 Step 1 |
| `features.py` | The ten ranking features | §3.4 Step 6 |
| `rank.py` | Scoring, logistic regression fit, LLM rerank | §3.4 Step 6, §6.6 |
| `simulate.py` | User simulator (if the kit ships none) | §6.5.2 |
| `telemetry.py` | Append-only JSONL logging, training corpus | §3.4 Step 7, §6.6 |
| `clarify.py` | Entropy x answerability clarification policy | §3.4 Step 5 |
| `ablate.py` | Ablation harness, scenario slicing | §6.3-§6.4 |
| `agent.py` | `Agent` — wires everything into `reset()`/`respond()` | §4 |

---

## Haojun — Indexes and retrieval

*Lane: `product_text()`, the FTS5 extension, the embedding matrix, the
three streams, union, and the floor check (§8.1).*

- **What's built:** `product_text()`/`load_catalog()` (A1); a real FTS5
  virtual table with the six documented column weights and sign-corrected
  bm25 (A4); real sentence-transformers/MiniLM encoding with an .npy cache
  (A5); brute-force kNN (A6, was already real); `build_facts_dict()`/
  `build_category_lists()` (A3, grouped by department — also carries
  brand/color/material/style/size beyond §3.2's literal field list, for
  clarify.py's E4); the three streams with per-track quotas, the buy-track
  keyword filter, browse-track semantic diversity cap, and popularity's
  modal-department targeting (A7); union/dedupe (already real) and a real
  floor-check relax-and-retry (A8/A9).
- **What's stubbed:** nothing in this lane; `llm_rerank()` (C7, a
  different owner's lane) remains a stub for lack of LLM access.
- **How to verify:** `python3 -m pytest tests/test_indexes.py
  tests/test_retrieval.py` (skip cleanly without sentence-transformers
  installed); `python3 -m evaluate` for an end-to-end real-data run,
  whose `recall` output is this lane's actual pool-recall numbers.
- **Known limitations:** the buy-track category filter matches
  `category` slot text as a substring of the lowercased product blob
  (cat3 is too coarse for an exact match to work); the browse-track
  diversity cap's 30% share and the department-level (not full-path)
  category grouping are both judgment calls the design doc leaves open
  ("a set share", "category path") — see retrieval.py's docstrings for
  the reasoning.

## Qikun — State and routing

*Lane: slot extraction, write/overwrite/delete, negation detection,
canonical reconstruction, the buy/browse routing rule (§8.2). Owns the
`intent_override` path — 30 sessions, all `hard`, 15% of score.*

- **What's built:** B1-B8 (schema, gazetteer, rule-based extraction, merge
  policy, negation detector, scenario buffer, canonical render, routing);
  B9 — an optional LLM extraction path behind `use_llm_extraction`
  (default off), wired with a real try-then-fall-back-to-B3 path, though
  the LLM call itself is a stub (no LLM access provided, same constraint
  as C7).
- **What's stubbed:** the LLM half of B9 only, documented as such.
- **How to verify:** `python3 -m pytest tests/test_extraction.py
  tests/test_negation.py tests/test_scenario.py tests/test_transitions.py
  tests/test_routing.py tests/test_canonical.py tests/test_gazetteer.py
  tests/test_state.py`.
- **Known limitations:** B10 (regression tests pulled from real D8
  override transcripts) is not yet written — `data/transcripts.txt` now
  has real intent_override/boundary transcripts to source them from.

## Emerson — Ranking

*Lane: the ten features, the logistic regression fit, the optional
pairwise-objective comparison, the flagged LLM rerank (§8.3).*

- **What's built:** C1-C4 (candidate shape, ten features, hand-set
  weighted scoring, missing-stream defaults); C5 — a real
  `sklearn.linear_model.LogisticRegression(class_weight="balanced",
  penalty="l2")` fit over `StandardScaler`'d features, validated with
  `GroupKFold` by `session_id`, with the scaler folded into raw-feature-
  space weights so `score_candidates()` needed no change; persistence to
  `models/ranker.json`.
- **What's stubbed:** `llm_rerank()` (C7, closed as won't-do — no LLM
  access provided).
- **Not yet done:** C8 (a notebook/script reporting fitted coefficients
  and pairwise feature correlations) — waiting on a real corpus/fit
  round-trip to report on. C6 (pairwise lambdarank comparison) closed as
  won't-do — explicitly time-permitting/off critical path per its own
  issue.
- **How to verify:** `python3 -m pytest tests/test_features.py
  tests/test_rank.py`.
- **Known limitations:** `price_fit()`'s real budget-fit formula (beyond
  null-safety) was never in scope (§2.2: 78.9% of prices are null; the
  non-null remainder isn't always numeric either — see clarify.py's
  budget-bucketing fix for what that data actually looks like).

## Chellappan — Simulator and training corpus

*Lane: the user simulator (if required), telemetry logging, instrumented
runs, and feature-matrix generation (§8.4). On the critical path — no
weight can be fitted and no ablation run until this lane delivers the
corpus.*

- **What's built:** D2-D8, real, run against the real 50,000-row
  catalogue and all 200 public sessions once Owner A's retrieval landed:
  `simulate.py`'s per-scenario release policies and conversation loop;
  `telemetry.py`'s `log_turn`/`build_training_rows`/
  `per_stream_recall_report`/`export_transcripts`; `run_instrumented_corpus()`
  as the end-to-end driver.
- **D1:** resolved as configuration C (§6.5.1) — no evaluator/kit was in
  the repo at implementation time, so this codebase owns the simulator and
  conversation loop. Reopens for real once `kit/` is populated, in case
  the actual kit turns out to bundle its own simulator (config A).
- **How to verify:** `python3 -m pytest tests/test_simulate.py
  tests/test_telemetry.py`; `python3 -m evaluate` for a real run —
  writes `data/features.jsonl` (~29,000+ labelled rows over 200 sessions),
  `data/telemetry.jsonl`, `data/transcripts.txt`, and prints the recall
  report.
- **Known limitations:** pool recall is uneven by scenario — buying
  clears ~99%, intent_override and browsing are much lower (real numbers
  from `python3 -m evaluate`'s `recall.by_scenario`), matching §6.2's
  point that pool recall names which half of the pipeline needs attention.

## Marcus — Evaluation and integration

*Lane: the clarification policy, the ablation harness, threshold tuning,
module wiring, repository health (§8.5). Owns `main`.*

- **What's built:** E1 (this repo skeleton, pinned requirements); E2
  (`agent.py` wires every module into a runnable, now-real `Agent`); E3 —
  `evaluate.py`, our own Hit Rate@10/MRR/MTTC/Efficiency scorer (§6.1's
  formulas verbatim) and `record_baseline()`, since the organizer
  evaluator itself isn't in the repo yet; E4 — `clarify.py`'s
  entropy x answerability policy against real per-candidate facts, not a
  fixture distribution; E9 — `scripts/check_data.py` / `make data`.
- **Not yet done:** E5 (config-flagged ablation re-scoring), E6
  (populating the nine-row ablation table), E7 (grid-search thresholds/
  quotas) — `ablate.py`'s `ABLATION_CONFIGS`/`AblationResult` shape is in
  place but `run_ablation()` itself is still a stub.
- **How to verify:** `python3 -m pytest` (full suite); `python3 -m
  evaluate` for the real baseline score.
- **Known limitations:** "headline figures come only from the supplied
  evaluator" (§8.4) still holds — `evaluate.py`'s numbers are this team's
  own computation against §6.1's formulas, not yet cross-checked against
  the organizer's kit.

---

## Known limitations

See §9 of the design doc for the full list (slot-extraction fragility,
popularity's long-tail blind spot, boundary-session handling, hand-set
answerability priors, effective sample size, self-generated training
corpus bias, and the two small-but-retained features). Each owner should
fold their component-specific limitations into their section above as
they land; this section stays as the pointer back to §9 until then.
