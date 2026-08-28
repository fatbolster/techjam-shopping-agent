# Shopping Copilot

Conversational search over a 50,000-product catalogue. Symbolic state for
what the user said, vector state for what the user meant.

Full design rationale, data measurements, and the delivery plan live in
[`docs/Shopping-Copilot-Technical-Design.pdf`](docs/Shopping-Copilot-Technical-Design.pdf).
This README is the assembled, per-owner summary (§7.1, §8.5.1: "each owner
writes their own README section ... Marcus merges").

**Status:** scaffold only. Every module is stubbed to its final signature
and returns fixture data — see each file's module docstring for what it
does and does not implement yet. `agent.py` runs end to end on fixtures
(`python3 agent.py`).

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

- **What's built:**
- **What's stubbed:**
- **How to verify:**
- **Known limitations:**

## Qikun — State and routing

*Lane: slot extraction, write/overwrite/delete, negation detection,
canonical reconstruction, the buy/browse routing rule (§8.2). Owns the
`intent_override` path — 30 sessions, all `hard`, 15% of score.*

- **What's built:**
- **What's stubbed:**
- **How to verify:**
- **Known limitations:**

## Emerson — Ranking

*Lane: the ten features, the logistic regression fit, the optional
pairwise-objective comparison, the flagged LLM rerank (§8.3).*

- **What's built:**
- **What's stubbed:**
- **How to verify:**
- **Known limitations:**

## Chellappan — Simulator and training corpus

*Lane: the user simulator (if required), telemetry logging, instrumented
runs, and feature-matrix generation (§8.4). On the critical path — no
weight can be fitted and no ablation run until this lane delivers the
corpus.*

- **What's built:**
- **What's stubbed:**
- **How to verify:**
- **Known limitations:**

## Marcus — Evaluation and integration

*Lane: the clarification policy, the ablation harness, threshold tuning,
module wiring, repository health (§8.5). Owns `main`.*

- **What's built:**
- **What's stubbed:**
- **How to verify:**
- **Known limitations:**

---

## Known limitations

See §9 of the design doc for the full list (slot-extraction fragility,
popularity's long-tail blind spot, boundary-session handling, hand-set
answerability priors, effective sample size, self-generated training
corpus bias, and the two small-but-retained features). Each owner should
fold their component-specific limitations into their section above as
they land; this section stays as the pointer back to §9 until then.
