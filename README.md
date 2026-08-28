# Shopping Copilot

Conversational search over a 50,000-product catalogue. Symbolic state for
what the user said, vector state for what the user meant.

Full design rationale, data measurements, and the delivery plan live in
[`docs/Shopping-Copilot-Technical-Design.pdf`](docs/Shopping-Copilot-Technical-Design.pdf).
This README summarises what is built, how to run it, and where the
remaining gaps are.

**Status:** end to end and real on the full 50,000-row catalogue —
`agents/our_agent.py` runs against real data, not just fixtures. D1 is now
resolved for real: the organizer's evaluator
(`evaluator/evaluator.py`) and reference baseline agent (`starter/`,
`agents/baseline_agent.py`) are in the repo. `starter/agent.py` is a
copy/paste slot — whatever's pasted there is what the evaluator scores;
copy `agents/our_agent.py` in to evaluate this team's real pipeline, or
`agents/baseline_agent.py` for the reference baseline — see "Official
evaluator results" below. Remaining gaps: E6 (actually running
all nine ablation configs against the real catalogue and populating the
table — the code path is real and tested, not yet executed end to end);
E7 (grid-search thresholds/quotas, not yet started). See each file's
module docstring for what it does and does not implement.

## Official evaluator results

Run via `make evaluate` (or `python3 -m evaluator.evaluator --output results/output.json`, from repo
root; needs `starter` importable as a package). The evaluator bundles its
own user simulator — configuration A (§6.5.1), resolved for real once the
kit arrived; see `kit/README.md`.

| Metric | Overall | buying | browsing | intent_override | boundary |
|---|---|---|---|---|---|
| Hit Rate@10 | 0.790 | 0.775 | 0.838 | 0.733 | 0.700 |
| MRR | 0.386 | 0.361 | 0.407 | 0.430 | 0.287 |
| MTTC | 4.49 | — | — | — | — |
| Efficiency | 0.651 | — | — | — | — |
| **Technical score** | **0.641** | — | — | — | — |

For reference, the kit's own baseline agent (`agents/baseline_agent.py`,
scored into `results/baseline.json`) gets Hit Rate@10 0.125, MRR 0.068,
technical score **0.107** — this pipeline is **6.0x** that.

(`recommended_technical_score = 0.50·HitRate + 0.30·MRR + 0.20·Efficiency`,
per the evaluator's own formula.) Full per-session detail in
`results/our_model.json`.

Score history on this evaluator, each step measured not assumed:

| change | Hit@10 | MRR | MTTC | score |
|---|---:|---:|---:|---:|
| hand-set ranking weights | 0.525 | 0.272 | 6.32 | 0.438 |
| + fitted ranker actually wired in | 0.580 | 0.302 | 5.73 | 0.486 |
| + clarification policy retuned to the evaluator | 0.750 | 0.384 | 4.74 | 0.616 |
| + department filter turned off | **0.790** | **0.386** | **4.49** | **0.641** |

The last step changed two constants in `clarify.py` (`ASK_THRESHOLD`,
`MAX_CLARIFICATIONS_PER_SESSION`) after measuring that the evaluator's
simulated user discloses nothing on a turn where `ask_attribute` is None —
so asking is the only information channel, not a cost to be rationed. See
`docs/DESIGN_AUDIT.md` DC-3. Net effect: 35 sessions went miss→hit, 1 went
hit→miss, and mean rank among hits was unchanged at ~1.95.

`evaluate.py`'s own §6.1-formula scorer remains useful for fast local
iteration (it doesn't need a full evaluator round-trip), but these are now
the headline numbers.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m agents.our_agent   # smoke test: three fixture turns, printed responses
make evaluate   # full 200-session official score -> results/output.json
```

Real data (`catalog.jsonl`, `public_set.jsonl`, the embedding matrix) is
never committed (`.gitignore` excludes `*.jsonl`/`*.npy`/`*.db`). It ships
with the organizer-supplied kit (see [`kit/README.md`](kit/README.md)) —
copy `catalog.jsonl`/`public_set.jsonl` into `data/`, then run `make data`
(or `python3 scripts/check_data.py`) to verify a clean clone has what it
needs (§8.0, §8.5 step E9).

## Running the evaluator

Run from the repo root (the folder containing `evaluator/`, `starter/`,
`data/`). The Makefile already has an `evaluate` target, and that is what
the team has been using — prefer it where the default output path suits:

```bash
make evaluate   # -> results/output.json
```

Use the explicit form when you need to control the output filename. To
evaluate our agent:

```bash
python3 -m evaluator.evaluator --output results/ours.json
```

To evaluate the baseline for comparison: open `starter/agent.py`, replace
its contents with the baseline agent code (`agents/baseline_agent.py`),
save, then run:

```bash
python3 -m evaluator.evaluator --output results/baseline.json
```

Put our agent code (`agents/our_agent.py`) back into `starter/agent.py`
afterwards.

To summarise every run in `results/`:

```bash
python3 -c "
import json,glob
for f in sorted(glob.glob('results/*.json')):
    r=json.load(open(f))
    if 'hit_rate_at_10' not in r: continue
    print(f,'hit %.3f mrr %.3f mttc %.2f score %.4f'%(
      r['hit_rate_at_10'],r['mrr'],r['mttc'],r['recommended_technical_score']))
    for k,v in sorted(r['scenario_metrics'].items()):
        print('   %-16s hit %.3f mrr %.3f'%(k,v['hit_rate_at_10'],v['mrr']))
"
```

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
| `agents/our_agent.py` | `Agent` — wires everything into `reset()`/`respond()` | §4 |

---

## Known limitations

See §9 of the design doc for the full list (slot-extraction fragility,
popularity's long-tail blind spot, boundary-session handling, hand-set
answerability priors, effective sample size, self-generated training
corpus bias, and the two small-but-retained features). Component-specific
limitations are recorded in each module's docstring; this section stays as
the pointer back to §9.
