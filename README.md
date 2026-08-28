# Bolster Shopping Agent

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
