# Supplied evaluator / participant kit

Drop the organizer-supplied kit here (not authored by this team, so it is
gitignored like `data/` and `models/`). Once present, D1 (§6.5.1) can be
resolved for real by reading its source instead of guessing:

- Whatever the evaluator's entrypoint/module is (e.g. `evaluate.py`,
  `evaluator.py`, or a package) — read it to determine configuration A/B/C:
  does it bundle a user simulator, call `respond()` once per session, or
  expect us to supply the whole conversation loop?
- Any baseline/reference agent it ships, for the §5.1 comparison.
- Anything defining turn-counting semantics, the `ask_attribute`
  vocabulary, or the Efficiency/usage formula.

Nothing in `agent.py`'s public contract should need to change to match the
kit — `Agent(catalog_path)`, `reset(session_id, user_profile)`,
`respond(session_id, user_message, turn, top_k)` were already written to
match the kit's baseline agent per §5.1. If the real kit's signature
differs, that supersedes what's here.
