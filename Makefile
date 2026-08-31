.PHONY: data setup test evaluate corpus fit reproduce doctor

# Reproducibility. The training corpus is sensitive to Python's per-process
# string-hash randomisation: set and dict iteration order reaches the corpus
# through rank tie-breaking, so two unseeded runs of identical source produce
# different corpora, different fitted rankers, and technical scores up to
# 0.026 apart (docs/changes.md, "Reproducibility note"). Every RNG in the
# codebase is already explicitly seeded; this is the one remaining source of
# variance. Pinning it makes the whole pipeline byte-reproducible.
#
# Override for a different draw with `make reproduce SEED=1`.
SEED ?= 0
export PYTHONHASHSEED = $(SEED)

# Interpreter. `make setup` creates .venv/, so prefer it automatically when
# present rather than requiring it to be activated first. This is not
# cosmetic: the system python3 may well import numpy but not
# sentence_transformers, in which case Agent() still constructs (it loads
# the precomputed data/embeddings.npy) and every respond() then raises
# inside evaluator.py:241's blanket `except Exception`. That turns a missing
# dependency into a silently well-formed results file reading 0.0 — no
# traceback, no warning. Run `make doctor` to check the interpreter first.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

# §8.0: "Never commit large binaries to the repo directly; ship a `make
# data` download script instead." §8.5 step E9. See scripts/check_data.py
# for why this verifies rather than fetches: catalog.jsonl/public_set.jsonl
# ship with the organizer kit, not an external URL this repo controls.
data:
	$(PYTHON) scripts/check_data.py

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

# Runs the organizer's local evaluator (evaluator/evaluator.py) against our
# real Agent (starter/agent.py) over all 200 public sessions. Output goes
# to results/, not the repo root, so every run is kept in one place rather
# than scattered loose files.
evaluate:
	mkdir -p results
	$(PYTHON) -m evaluator.evaluator --output results/output.json

# The two offline stages that produce models/ranker.json. Neither is needed
# to run the agent, but `make evaluate` on a clean clone scores well below
# the headline without them: models/ is gitignored, so rank() falls back to
# hand-set weights until a ranker has been fitted.
corpus:
	$(PYTHON) -m evaluate

fit:
	$(PYTHON) scripts/fit_ranker.py

# Full reproduction from a clean clone, seed-pinned end to end (~20 min).
reproduce: corpus fit evaluate

# Fails loudly if the interpreter cannot run the agent, instead of letting
# evaluator.py's blanket except turn it into a 0.0 score.
doctor:
	@echo "interpreter: $(PYTHON)"
	@$(PYTHON) -c "import numpy, sklearn, sentence_transformers; print('dependencies: OK')" \
	  || (echo "dependencies: MISSING - run 'make setup' (a missing dep scores 0.0 silently)"; exit 1)
	@$(PYTHON) -c "import json,os; p='models/ranker.json'; print('ranker:', 'present, %d weights' % len(json.load(open(p))['weights']) if os.path.exists(p) else 'ABSENT - run make fit, or scores fall back to hand-set weights')"
