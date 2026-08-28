"""Verify the data/ directory holds what the pipeline expects (§8.0, E9).

Design doc §8.0: "Never commit large binaries to the repo directly; ship a
`make data` download script instead." Owner Marcus, §8.5 step E9: "Data
download script; verify from a clean clone."

There is no separate download URL to script against here: catalog.jsonl
and public_set.jsonl are supplied as part of the organizer's kit (the same
kit that resolves D1 — see kit/README.md), not fetched from an external
source we control. So this script is the "verify from a clean clone" half
of E9 — it checks the files a clean clone needs are actually in place and
gives a clear, specific error naming exactly what's missing and where it
comes from, rather than a script that pretends to fetch something with no
real source to fetch from.

Run directly: `python3 scripts/check_data.py` (also `make data`, see the
Makefile target of the same name). Exits non-zero with a readable message
listing every missing file if any are absent; otherwise reports row counts
for a quick sanity check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# path -> (line count is meaningful?, human description, where it comes from)
REQUIRED_FILES: dict[str, tuple[bool, str, str]] = {
    "data/catalog.jsonl": (True, "the 50,000-product catalogue (§1)", "the organizer kit"),
    "data/public_set.jsonl": (True, "the 200 public evaluation sessions (§1)", "the organizer kit"),
}
# Present only after a real corpus run (D6/E3) — checked but not required,
# since a clean clone legitimately won't have these yet.
DERIVED_FILES: dict[str, str] = {
    "data/embeddings.npy": "the cached embedding matrix (§3.2 Index 2, A5) — rebuilt automatically on first real run",
    "data/features.jsonl": "the training corpus (§8.4 D6) — produced by telemetry.run_instrumented_corpus()",
}


def check() -> int:
    missing = []
    for rel_path, (count_lines, description, source) in REQUIRED_FILES.items():
        path = REPO_ROOT / rel_path
        if not path.exists():
            missing.append(f"  {rel_path} — {description}, from {source}")
            continue
        if count_lines:
            with open(path, encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
            print(f"OK  {rel_path}: {n} rows")
        else:
            print(f"OK  {rel_path}")

    if missing:
        print("\nMissing required data files:", file=sys.stderr)
        for line in missing:
            print(line, file=sys.stderr)
        print(
            "\nThese ship with the organizer-supplied kit (see kit/README.md), "
            "not a URL this script downloads from — copy them into data/ "
            "from wherever the kit was provided, then re-run this check.",
            file=sys.stderr,
        )
        return 1

    for rel_path, description in DERIVED_FILES.items():
        path = REPO_ROOT / rel_path
        status = "present" if path.exists() else "not yet built"
        print(f"    {rel_path}: {status} ({description})")

    print("\nAll required data files present.")
    return 0


def _validate_jsonl(rel_path: str) -> None:
    """Optional deeper check: confirm each line actually parses as JSON.
    Not run by default (the row counts above are the fast path) — call
    check_data.validate_jsonl('data/catalog.jsonl') interactively if a
    row-count-only check isn't enough to trust a suspect file.
    """
    path = REPO_ROOT / rel_path
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{rel_path}:{i} is not valid JSON: {e}") from e


if __name__ == "__main__":
    sys.exit(check())
