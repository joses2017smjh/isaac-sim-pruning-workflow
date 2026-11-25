"""tools/jaw_reading_split.py: the split is by target, total, and divides only target 8235."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _split():
    spec = importlib.util.spec_from_file_location("jaw_reading_split", ROOT / "tools/jaw_reading_split.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_named_targets_and_the_divided_one():
    split = _split()
    assert split.side("artifacts/vision_robustness/x/run_01", "tree1_SPUR_component_15004") == "design"
    assert split.side("artifacts/vision_robustness/x/run_01", "tree1_SPUR_component_14944") == "test"
    assert (
        split.side("artifacts/vision_robustness/lighting-20260919/run_02_morning_raw", "tree0_SPUR_component_8235")
        == "design"
    )
    assert split.side("artifacts/isaac_render/job_21358986", "tree0_SPUR_component_8235") == "test"
    assert (
        not split.DESIGN_TARGETS & split.TEST_TARGETS
        and split.DIVIDED_TARGET not in split.DESIGN_TARGETS | split.TEST_TARGETS
    )
    assert (
        split.side("a/b", "tree0_SPUR_component_7524") == "design"
        and split.side("a/b", "tree1_SPUR_component_19145") == "test"
    )


def test_every_recorded_capture_falls_on_one_side_and_a_target_only_on_one():
    split = _split()
    summary = json.loads((ROOT / "docs/evidence/jaw_beams_summary_2026-11-25.json").read_text())
    sides = {}
    for row in summary["captures"]:
        which = split.side(row["capture"], row["target"])
        assert which in split.SIDES
        sides.setdefault(row["target"], set()).add(which)
    divided = [target for target, found in sides.items() if len(found) > 1]
    assert divided == ["tree0_SPUR_component_8235"]
    # Both sides hold closure captures, and each holds some of the refuted closures.
    refuted = json.loads((ROOT / "docs/evidence/jaw_presence_verdicts_2026-11-25.json").read_text())["verdicts"]["J4"]
    where = [split.side(capture, "tree0_SPUR_component_8235") for capture in refuted["closures_accepted"]]
    assert sorted(where) == ["design", "design", "design", "test", "test"]
