# Techjam Shopping Agent (Team fatbolster)

## Project Overview

<!-- TODO: fill in. -->

_[To be completed.]_

## Architecture at a Glance

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

The full per-turn architecture, from a user utterance through to the ranked
recommendations, is documented in [`docs/pipeline.md`](docs/pipeline.md).

### The per-turn pipeline

One pass per turn: extract slots from the message, rebuild and re-embed the
canonical intent, route buy/browse, retrieve from three streams, decide
whether to ask a clarifying question, rank, log. See
[`docs/pipeline.md`](docs/pipeline.md) for the full rationale and a step-by-step
trace of one turn.

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

## Setup and Installation

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

## Steps to Reproduce Your Results

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

## Limitations and Future Improvements

**Limitation 1 — No model of long-term user behaviour.**
The supplied dataset is organised at the *session* level: each of the 200
sessions belongs to a distinct user, and no user ever appears twice. There is
no second session for any shopper, so there is no purchase history, no
repeat-visit signal, and nothing to learn about how an individual's taste
persists or drifts over time. The agent therefore treats every conversation as
a cold start. What personalisation exists is confined to the static
`user_profile` supplied at `reset()`, distilled once into three preference tags
and a rating-style value and then held read-only for the session.

**Improvement 1 — A persistent per-user profile store.**
Given data in which users recur, we would carry state across sessions: a
profile updated at session close with learned brand and price affinities,
category recency, and a per-user prior folded into the ranker as additional
features. Cold-start behaviour would then degrade gracefully into the current
design rather than being the only mode available. We deliberately did not
simulate this on the supplied data — fabricating multi-session histories would
have produced a model validated only against our own invention, which fails
invisibly on transfer to real users.

**Limitation 2 — A small, stochastically generated training corpus.**
No labelled ranking data ships with the task, so the training corpus is
produced by replaying all 200 sessions through our own user simulator. That
simulator is stochastic, so each regeneration yields a different corpus: across
two runs of identical source code, the target appeared in the candidate pool on
56.5% of turns in one corpus and 43.1% in the other, and the resulting models
scored 0.7774 and 0.7511. The fit is also thin — eleven parameters against
roughly 200 effective samples, since turns within a session are not
independent — which is enough for a feature coefficient to change sign between
refits. A clean reproduction should therefore expect a score in the 0.75-0.78
range rather than an exact figure.

**Improvement 2 — Report a distribution, and regularise toward stability.**
With more time we would quote a median over three to five complete
corpus-generation and refit cycles instead of a single run, so the headline
figure carries a measured variance rather than an implied precision. We would
also trade a little peak score for transferability: stronger regularisation,
or a smaller set of less correlated features, so that coefficient signs are
stable across corpus draws. The private evaluation set is drawn from different
users and different products, and a model whose weights move under resampling
is exactly the model least likely to survive that shift.

## Team Member Contributions

Code attribution below is derived from the repository's commit history;
non-code contributions are recorded as reported by the team.

| Member | GitHub | Contribution |
|---|---|---|
| **Marcus** | `peanutbutter1212` | Category matching against the full category path (`features.py`), indexing work (`indexes.py`), and ranker retraining on a corpus matched to the current agent — the two changes that moved the score from 0.641 to 0.696. Authored the reproduction steps and reference-run table. Produced the project video. |
| **Emerson** | `fatbolster` | Devised the original concept and overall solution design, and authored the written project description. Ranking stage (`rank.py`, `features.py`): the ten ranking features, hand-set and fitted scoring, logistic-regression fit with GroupKFold validation and held-out ranking metrics. Clarification policy (`clarify.py`) and its entropy x answerability scoring. Agent orchestration (`agents/our_agent.py`), retrieval tuning, the ablation harness (`ablate.py`), and the majority of the test suite. Wrote the README for the GitHub repository, along with the rest of the project documentation. |
| **Qikun** | `qikunye` | Helped refine the initial concept, scoping it so that it was practical to complete within the TechJam timeline. Slot extraction and session state (`extract.py`, `state.py`): the slot dictionary with write/overwrite/delete semantics, negation and intent-override handling, canonical intent reconstruction, and the scenario buffer. Later clarification-answerability and retrieval optimisation across `clarify.py`, `retrieval.py` and `telemetry.py`. |
| **Chellappan** | `chellu19` | User simulator and evaluation instrumentation (`simulate.py`, `telemetry.py`): per-scenario simulation policies, the instrumented corpus run that produces the ranker's training matrix, append-only telemetry logging, and per-stream recall reporting. |
| **Haojun** | `haojun-mah` |  |

## Appendix

For detailed results — every change, its measured effect, per-scenario
breakdowns, ablation tables and the full reproducibility note — see
[`docs/changes.md`](docs/changes.md).

Further reference:

- [`docs/pipeline.md`](docs/pipeline.md) — end-to-end architecture, from a
  user utterance to the ranked recommendations, including the offline
  training loop and an explicit list of what is stubbed or disabled.
