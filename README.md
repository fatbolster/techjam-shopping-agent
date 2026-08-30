# Techjam Shopping Agent (Team fatbolster) 

A headless conversational shopping agent. It talks to a shopper across
multiple turns and surfaces the product they actually want — ranked as
high as possible, in as few turns as possible, within a hard limit of ten
turns per session.

**The problem.** Keyword search assumes the shopper already knows the
words for what they want. Type "black running shoes" and it works. Type
"something for a beach trip" and it collapses, because no product in the
catalogue is labelled that way.

**The approach.** Two representations of the conversation, kept in step:

- A **slot dictionary** holds what the shopper literally said. It supports
  write, overwrite, and delete — so when they say "actually not black,
  blue", black is genuinely gone.
- A **canonical intent string** is rebuilt from that dictionary every turn
  and re-embedded. Because the dictionary no longer holds the superseded
  value, the rebuilt query cannot contain it either.

Three retrieval streams — keyword (BM25/FTS5), semantic (MiniLM
embeddings), and popularity — run independently and are unioned into one
candidate pool, so a miss requires all three to fail at once. A ten-feature
logistic regression then ranks that pool, and the agent asks a clarifying
question whenever one would earn its keep.

Runs fully in memory over a 50,000-product catalogue, with no external
vector database and no LLM calls required.

Full design rationale and data measurements live in
[`docs/Shopping-Copilot-Technical-Design.pdf`](docs/Shopping-Copilot-Technical-Design.pdf).
An audit of the implementation against that document and against the
supplied evaluator is in [`docs/DESIGN_AUDIT.md`](docs/DESIGN_AUDIT.md).

## Architecture at a glance

One pass per turn: extract slots from the message, rebuild and re-embed the
canonical intent, route buy/browse, retrieve from three streams, decide
whether to ask a clarifying question, rank, log. See §3-§4 of the design
doc for the full rationale and system diagram.

| Module | Owns | Design doc |
|---|---|---|
| `agents/our_agent.py` | `Agent` — wires everything into `reset()`/`respond()` | §4 |
| `agents/baseline_agent.py` | The kit's reference baseline, kept for comparison | §5.1 |
| `utils.py` | `product_text()`, the shared `Candidate` shape | §3.2, §7.2 |
| `indexes.py` | FTS5, embedding matrix, facts dict, category lists | §3.2 |
| `retrieval.py` | Three streams, union, floor check | §3.4 Step 4 |
| `extract.py` | Slot extraction, negation, merge policy | §3.4 Step 1 |
| `state.py` | Slot dict, scenario buffer, canonical render, routing | §3.3, §3.4 Steps 2-3 |
| `features.py` | The ten ranking features | §3.4 Step 6 |
| `rank.py` | Scoring, logistic regression fit, LLM rerank | §3.4 Step 6, §6.6 |
| `clarify.py` | Entropy x answerability clarification policy | §3.4 Step 5 |
| `telemetry.py` | Append-only JSONL logging, training corpus | §3.4 Step 7, §6.6 |
| `simulate.py` | User simulator, for our own corpus generation | §6.5.2 |
| `evaluate.py` | Our own Hit@10/MRR/MTTC scorer, for fast local iteration | §6.1 |
| `ablate.py` | Ablation harness, scenario slicing | §6.3-§6.4 |
| `scripts/check_data.py` | Verifies `data/` has what a clean clone needs | §8.0 |
| `scripts/fit_ranker.py` | Fits the ranker on the logged feature matrix | §6.6 |
| `scripts/report_ranker.py` | Fitted weights, correlations, near-zero flags | §6.3, §2.4 |

## Requirements and setup

Python 3.12. From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The catalogue and session files are not committed (`.gitignore` excludes
`*.jsonl`/`*.npy`/`*.db`). They ship with the organizer kit — copy
`catalog.jsonl` and `public_set.jsonl` into `data/`, then verify:

```bash
make data          # or: python3 scripts/check_data.py
```

Smoke test and full test suite:

```bash
python3 -m agents.our_agent   # three fixture turns, printed responses
make test                     # or: python3 -m pytest
```

## Running the evaluation

Run from the repo root (the folder containing `evaluator/`, `starter/`,
`data/`) — the evaluator imports `starter` as a package.

`starter/agent.py` is the slot the evaluator scores: whatever is pasted
there is what gets measured. The two real agents live in `agents/`.

To evaluate our agent:

```bash
make evaluate   # -> results/output.json
```

or, to control the output filename:

```bash
python3 -m evaluator.evaluator --output results/ours.json
```

To evaluate the baseline for comparison, replace the contents of
`starter/agent.py` with `agents/baseline_agent.py`, save, then run:

```bash
python3 -m evaluator.evaluator --output results/baseline.json
```

Put `agents/our_agent.py` back into `starter/agent.py` afterwards.

### Reproducing the headline score

The fitted ranker is not committed (`models/` is gitignored, like all
generated data), and the agent falls back to hand-set weights without
it — so a clean clone must fit it before evaluating, or the score will
come out well below the committed runs:

```bash
python3 -m evaluate            # instrumented corpus -> data/features.jsonl (~10 min)
python3 scripts/fit_ranker.py  # fit -> models/ranker.json
make evaluate                  # official evaluator -> results/output.json
```

Expected: technical score **0.75-0.78**. The evaluator itself is fully
deterministic — a fixed `models/ranker.json` reproduces every metric exactly —
but the training corpus is regenerated by a stochastic simulator, so the
fitted ranker (and with it the score) varies between clean reproductions. Two
refits of identical source have measured 0.7511 and 0.7774; see the
reproducibility note in [`docs/changes.md`](docs/changes.md) for why.

| Run | Score | What it measures |
|---|---|---|
| `results/most_updated_output.json` | 0.777 | Model 8.0, current code and currently fitted ranker |
| `results/baseline.json` | 0.107 | the kit's starter agent |

Earlier reference runs (Models 5.0-7.1) are recorded in `docs/changes.md` and
recoverable from git history with `git checkout <rev> -- results/`.

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
