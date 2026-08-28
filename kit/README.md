# Organizer-supplied kit — resolved (D1)

The kit turned out to place its files at the repo root, not here — this
directory is kept only as a pointer, since `kit/README.md` is what the
rest of this repo's docs link to.

- **`evaluator/evaluator.py`** — the real local evaluator. Bundles its own
  user simulator (`initial_message()`/`customer_reply()`/`behavior_for()`)
  driven by an `intent_card` derived from each session's target product —
  **configuration A** (§6.5.1), not configuration C. `simulate.py` (D3-D6)
  was the correct fallback while no kit was present, but the evaluator
  does not call into it; `simulate.py` remains useful for this team's own
  training-corpus generation (feeding C5's fit), just not for official
  scoring.
- **`starter/agent.py`** — the evaluator's actual import target
  (`from starter.agent import Agent`), but treated as a **copy/paste
  slot**, not a permanent home for any one agent's code: whatever's pasted
  there at run time is what gets scored. The two real, version-controlled
  agents live in `agents/`:
  - `agents/our_agent.py` — this team's real pipeline (identical to what
    used to be a root-level `agent.py`; every other module that needs the
    real `Agent` — `telemetry.py`, tests — imports it from here now).
  - `agents/baseline_agent.py` — the kit's own reference baseline
    ("editable weak baseline: stateless BM25 retrieval with no LLM
    dependency" — its own docstring), preserved for comparison.

  To evaluate this team's pipeline: `cp agents/our_agent.py
  starter/agent.py` first, then run the evaluator. `results/our_model.json`
  / `results/baseline.json` are examples of each, already committed.

Run it: `make evaluate` (or `python3 -m evaluator.evaluator --output
results/output.json` directly) from the repo root — needs `starter`
importable as a package, hence running from root rather than
`cd evaluator/`.

Resolved for D1: configuration A. `evaluator/`, `starter/`, and `agents/`
are all committed (not gitignored like `data/`/`models/`) — small,
essential to the repo being self-contained and re-scorable by anyone who
clones it.
