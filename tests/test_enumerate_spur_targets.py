"""Protect target pre-registration: honest screening, fixed seed, no hand-picking."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def enumerate_targets(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("enumerate_spur_targets", tools / "enumerate_spur_targets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quad_strip(first_vertex, rings=4):
    """Build one connected tube: `rings` quads over 2*(rings+1) vertices."""
    counts, indices = [], []
    for ring in range(rings):
        base = first_vertex + 2 * ring
        counts.append(4)
        indices.extend([base, base + 1, base + 3, base + 2])
    return counts, indices


def test_two_separate_tubes_are_two_components(enumerate_targets):
    counts_a, indices_a = _quad_strip(0)
    counts_b, indices_b = _quad_strip(10)
    components = enumerate_targets.connected_components(20, counts_a + counts_b, indices_a + indices_b)
    assert len(components) == 2
    firsts = sorted(member[0] for member in components)
    assert firsts == [0, 10]
    # Components must not share vertices.
    assert set(components[0]).isdisjoint(components[1])


def test_incomplete_topology_is_refused(enumerate_targets):
    with pytest.raises(ValueError, match="complete polygon topology"):
        enumerate_targets.connected_components(8, [4, 4], [0, 1, 2, 3])
    with pytest.raises(ValueError, match="invalid vertex"):
        enumerate_targets.connected_components(4, [4], [0, 1, 2, 99])


def test_screen_keeps_rejections_for_audit(enumerate_targets):
    candidates = [
        {"component_first_vertex": 1, "max_radius_m": 0.005},
        {"component_first_vertex": 2, "max_radius_m": 0.030},
        {"component_first_vertex": 3, "max_radius_m": 0.012},
    ]
    accepted, rejected = enumerate_targets.screen_candidates(candidates, 0.012)
    # The boundary radius is inside the jaw, matching select_component's `<=`.
    assert [item["component_first_vertex"] for item in accepted] == [1, 3]
    assert [item["component_first_vertex"] for item in rejected] == [2]
    assert "rejected_because" in rejected[0]
    assert len(accepted) + len(rejected) == len(candidates)


def test_selection_is_deterministic_for_a_fixed_seed(enumerate_targets):
    pool = {
        0: [
            {
                "component_first_vertex": v,
                "max_radius_m": 0.006,
                "length_m": 0.05,
                "axis": [0, 0, 1],
                "center_m": [0, 0, 0],
            }
            for v in range(40)
        ]
    }
    first = enumerate_targets.select_targets(pool, per_tree=5, seed=20260923)
    again = enumerate_targets.select_targets(pool, per_tree=5, seed=20260923)
    assert first == again
    different = enumerate_targets.select_targets(pool, per_tree=5, seed=7)
    assert [t["component_first_vertex"] for t in different] != [t["component_first_vertex"] for t in first]
    # Output order is by vertex ID, so the record is stable regardless of draw order.
    chosen = [t["component_first_vertex"] for t in first]
    assert chosen == sorted(chosen)
    assert len(set(chosen)) == 5


def test_selection_spans_every_requested_tree(enumerate_targets):
    def pool(offset):
        return [
            {
                "component_first_vertex": offset + v,
                "max_radius_m": 0.006,
                "length_m": 0.05,
                "axis": [0, 0, 1],
                "center_m": [0, 0, 0],
            }
            for v in range(20)
        ]

    targets = enumerate_targets.select_targets({0: pool(0), 1: pool(1000)}, per_tree=4, seed=1)
    assert len(targets) == 8
    assert sorted({t["target_tree_index"] for t in targets}) == [0, 1]
    assert sum(1 for t in targets if t["target_tree_index"] == 1) == 4


def test_too_small_a_pool_fails_loudly_rather_than_shrinking_n(enumerate_targets):
    pool = {
        0: [
            {
                "component_first_vertex": v,
                "max_radius_m": 0.006,
                "length_m": 0.05,
                "axis": [0, 0, 1],
                "center_m": [0, 0, 0],
            }
            for v in range(3)
        ]
    }
    with pytest.raises(ValueError, match="fewer than the requested"):
        enumerate_targets.select_targets(pool, per_tree=10, seed=1)


def test_export_provenance_mismatch_is_refused(tmp_path, enumerate_targets):
    export = tmp_path / "export"
    export.mkdir()
    (export / "tree0.usdc").write_bytes(b"actual bytes")
    manifest = {"units": "meters", "up_axis": "Z", "artifacts": [{"path": "tree0.usdc", "sha256": "0" * 64}]}
    (export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        enumerate_targets.verify_export(export, [0])


def test_non_metre_export_is_refused(tmp_path, enumerate_targets):
    export = tmp_path / "export"
    export.mkdir()
    (export / "manifest.json").write_text(json.dumps({"units": "centimeters", "up_axis": "Z"}), encoding="utf-8")
    with pytest.raises(ValueError, match="meters and Z-up"):
        enumerate_targets.verify_export(export, [0])


def test_listed_only_takes_every_screened_listed_spur_and_never_samples(tmp_path, enumerate_targets):
    import json

    manifest = {
        "branch_geometry_candidates": {
            "tree1_SPUR": {"candidates": [{"component_first_vertex": 10}, {"component_first_vertex": 30}]}
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    listed = enumerate_targets.listed_candidates(tmp_path / "manifest.json", 1)
    assert listed == {10, 30}
    accepted = {
        1: [
            {
                "component_first_vertex": v,
                "max_radius_m": 0.006,
                "length_m": 0.05,
                "axis": [0, 0, 1],
                "center_m": [0, 0, 0],
            }
            for v in (30, 10, 20)
        ]
    }
    selected = enumerate_targets.select_listed(accepted, {1: listed})
    assert [t["component_first_vertex"] for t in selected] == [10, 30]
    assert all(t["target_tree_index"] == 1 for t in selected)
    with pytest.raises(ValueError):
        enumerate_targets.select_listed({1: []}, {1: listed})
