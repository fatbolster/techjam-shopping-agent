"""Tests for scripts/check_data.py (design doc §8.5 step E9)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_data  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(check_data, "REPO_ROOT", tmp_path)
    return tmp_path


def test_check_fails_with_no_data_dir(capsys):
    assert check_data.check() == 1
    assert "Missing required data files" in capsys.readouterr().err


def test_check_reports_which_files_are_missing(tmp_path, capsys):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "catalog.jsonl").write_text('{"a": 1}\n')
    assert check_data.check() == 1
    assert "public_set.jsonl" in capsys.readouterr().err


def test_check_succeeds_when_both_required_files_present(tmp_path, capsys):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "catalog.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    (tmp_path / "data" / "public_set.jsonl").write_text('{"a": 1}\n')
    assert check_data.check() == 0
    out = capsys.readouterr().out
    assert "2 rows" in out
    assert "All required data files present" in out


def test_check_skips_blank_lines_when_counting(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "catalog.jsonl").write_text('{"a": 1}\n\n{"a": 2}\n')
    (tmp_path / "data" / "public_set.jsonl").write_text('{"a": 1}\n')
    assert check_data.check() == 0  # doesn't miscount/crash on blank lines
