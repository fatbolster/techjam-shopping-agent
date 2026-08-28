"""Points the evaluator at this team's real Agent (agent.py, repo root).

The organizer kit's own baseline agent — an "editable weak baseline:
stateless BM25 retrieval with no LLM dependency" (its own docstring) — is
preserved at starter/baseline_agent.py for reference/comparison. evaluator/
evaluator.py imports `from starter.agent import Agent`, so this module is
the swap point: re-exporting the real Agent here means no evaluator code
changes are needed to evaluate this team's actual pipeline instead of the
reference baseline.
"""

from agent import Agent  # noqa: F401
