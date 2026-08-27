from __future__ import annotations

import pytest

from isaaclab_pruning.geometry import Cylinder, oracle_cut_candidates, oracle_cut_point


def _cylinder(part_name: str, centroid, radius: float) -> Cylinder:
    return Cylinder(
        record_id=part_name,
        part_name=part_name,
        centroid=centroid,
        orientation=(0.0, 0.0, 1.0),
        radius=radius,
        length=0.2,
    )


def test_oracle_prefers_thick_exposed_spurs() -> None:
    cylinders = [
        _cylinder("trunk_1", (0.0, 0.0, 0.0), 0.05),
        _cylinder("spur_thin_cluster", (0.10, 0.0, 0.0), 0.004),
        _cylinder("spur_thin_neighbor", (0.12, 0.0, 0.0), 0.004),
        _cylinder("spur_thick_open", (1.0, 0.0, 0.0), 0.012),
    ]

    ordered = oracle_cut_candidates(cylinders)
    assert [item.part_name for item in ordered] == [
        "spur_thick_open",
        "spur_thin_neighbor",
        "spur_thin_cluster",
    ]
    assert [item.neighbor_count for item in ordered] == [0, 1, 2]
    chosen = oracle_cut_point(cylinders)
    assert chosen.part_name == "spur_thick_open"
    assert chosen.confidence == 1.0


def test_oracle_raises_when_no_matching_organs() -> None:
    with pytest.raises(ValueError, match="No cylinders matched"):
        oracle_cut_point([_cylinder("trunk_1", (0.0, 0.0, 0.0), 0.05)])
