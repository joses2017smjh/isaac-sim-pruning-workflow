"""tools/audit_gate_evidence.py: the parity composition and the live-loop vision ages."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_gate_evidence", ROOT / "tools/audit_gate_evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verdict(passed, reason=None):
    return {"passed": passed, "reason": reason}


def test_cycles_after_a_latch_are_separated_from_live_decisions(audit):
    rows = [
        ("a", _verdict(True)),
        ("a", _verdict(True)),
        ("a", _verdict(False, "tof_minimum_clearance")),
        ("a", _verdict(False, "tof_minimum_clearance")),
        ("a", _verdict(False, "vision_invalid")),
        ("b", _verdict(True)),
        ("b", _verdict(True)),
        ("c", _verdict(False, "vision_invalid")),
    ]
    breakdown = audit.parity_breakdown(rows)
    assert breakdown["cycles"] == 8
    assert breakdown["passing_before_any_hold"] == 4
    assert breakdown["first_holds"] == 2
    assert breakdown["first_hold_reasons"] == {"tof_minimum_clearance": 1, "vision_invalid": 1}
    assert breakdown["cycles_after_a_latch"] == 2
    assert breakdown["cycles_with_a_decision"] == 6
    assert breakdown["captures_that_never_held"] == 1


def test_the_cycles_digest_matches_the_parity_runs_convention(audit, tmp_path):
    path = tmp_path / "cycles.jsonl.gz"
    lines = ['{"capture": "a"}', '{"capture": "b"}']
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    assert audit.cycles_digest(path) == hashlib.sha256("".join(lines).encode()).hexdigest()


def test_a_pass_after_a_latch_is_refused(audit):
    with pytest.raises(audit.AuditError):
        audit.parity_breakdown([("a", _verdict(False, "operator_hold")), ("a", _verdict(True))])


def test_vision_ages_counts_every_decision_including_those_without_an_age(audit, tmp_path):
    run = tmp_path / "run_00_x"
    run.mkdir()
    rows = [{"vision_age_s": 0}, {"vision_age_s": 0}, {"vision_age_s": None}, {"vision_age_s": math.nan}]
    (run / "ros2_loop_decisions.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    ages = audit.vision_ages(tmp_path)
    assert ages == {"run_00_x": {"decisions": 4, "ages_s": {0.0: 2}, "without_age": 2}}


def test_a_run_without_a_decision_log_is_refused(audit, tmp_path):
    (tmp_path / "run_00_x").mkdir()
    with pytest.raises(audit.AuditError):
        audit.vision_ages(tmp_path)


def test_the_output_must_be_new_and_outside_the_repository(audit, tmp_path):
    existing = tmp_path / "exists.json"
    existing.write_text("{}")
    for path in (existing, ROOT / "docs/evidence/new.json"):
        with pytest.raises(SystemExit):
            audit.main(["--output", str(path)])
