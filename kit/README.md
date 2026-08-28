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
- **`starter/agent.py`** — swapped to `from agent import Agent`, re-
  exporting this team's real `Agent` (repo root `agent.py`) so the
  evaluator scores our actual pipeline. The kit's own reference baseline
  ("editable weak baseline: stateless BM25 retrieval with no LLM
  dependency" — its own docstring) is preserved at
  `starter/baseline_agent.py` for comparison.

Run it: `python3 -m evaluator.evaluator --output output.json` from the
repo root (needs `starter` importable as a package, hence running from
root rather than `cd evaluator/`).

Resolved for D1: configuration A. Both `evaluator/` and `starter/` are
committed (not gitignored like `data/`/`models/`) — small, essential to
the repo being self-contained and re-scorable by anyone who clones it.
