.PHONY: data setup test evaluate

# §8.0: "Never commit large binaries to the repo directly; ship a `make
# data` download script instead." §8.5 step E9. See scripts/check_data.py
# for why this verifies rather than fetches: catalog.jsonl/public_set.jsonl
# ship with the organizer kit, not an external URL this repo controls.
data:
	python3 scripts/check_data.py

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	python3 -m pytest

# Runs the organizer's local evaluator (evaluator/evaluator.py) against our
# real Agent (starter/agent.py) over all 200 public sessions. Output goes
# to results/, not the repo root, so every run is kept in one place rather
# than scattered loose files.
evaluate:
	mkdir -p results
	python3 -m evaluator.evaluator --output results/output.json
