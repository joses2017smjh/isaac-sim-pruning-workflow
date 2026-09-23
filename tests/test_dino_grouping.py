"""Six-view DINO groups by condition and leaves rigs it was not trained on to DA2."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def dino(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("dino_generalization", TOOLS / "dino_generalization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(tree, condition, lighting, compatible, n=6):
    return [
        {
            "frame": {
                "tree_id": tree,
                "condition": condition,
                "lighting": lighting,
                "dino_rig_compatible": compatible,
                "view_id": f"v{i}",
            }
        }
        for i in range(n)
    ]


def test_conditions_under_one_light_never_share_a_group(dino):
    rows = _rows("t", "source", "source", True) + _rows("t", "source/far/isaac_wrist", "source", False)
    rows += _rows("t", "evening/gamma", "evening", True) + _rows("t", "source/sweep/training_rig", "source", False, n=8)
    groups, skipped = dino.group_rows(rows)
    assert set(groups) == {("t", "source"), ("t", "evening/gamma")}
    assert all(len(v) == 6 for v in groups.values())
    assert {(s["condition"], s["frames"]) for s in skipped} == {
        ("source/far/isaac_wrist", 6),
        ("source/sweep/training_rig", 8),
    }
    assert all(s["reason"] == "outside_dino_rig" for s in skipped)


def test_matrix_rows_without_conditions_group_by_lighting(dino):
    rows = [{"frame": {"tree_id": "t", "lighting": "noon"}} for _ in range(6)]
    groups, skipped = dino.group_rows(rows)
    assert set(groups) == {("t", "noon")} and not skipped
    smoke, _ = dino.group_rows(rows[:1], smoke=True)
    assert len(smoke[("smoke", "repeated_single_view")]) == 6
