from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tools.fetch_sources import _fetch

MANIFEST = Path(__file__).resolve().parents[1] / "third_party" / "sources.yaml"
RIG = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "isaaclab_pruning"
    / "isaaclab_pruning"
    / "config"
    / "rigs"
    / "mock_pruner_vl53l8cx.yaml"
)


def test_source_manifest_pins_reviewed_revisions() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    rig = yaml.safe_load(RIG.read_text(encoding="utf-8"))
    sources = manifest["sources"]

    assert sources["isaaclab_sensor_learning"]["revision"] == ("5701a774af7b8579269f924689aaf79b9574a53c")
    assert sources["isaaclab_sensor_learning"]["integration"] == "fork_history"
    assert sources["isaaclab_sensor_learning"]["upstream_status"] == "unavailable_as_of_2026-08-31"
    assert sources["isaaclab_sensor_learning"]["archive_repository"].endswith(
        "joses2017smjh/isaac-sim-pruning-workflow.git"
    )
    assert sources["branch_detection_system"]["revision"] == ("dfede4c0f251358ebed7a1f90ff887847c2fbeb0")
    assert rig["source_revision"] == sources["branch_detection_system"]["revision"]
    assert sources["branch_detection_system"]["license"] == "NOASSERTION"
    assert sources["branch_detection_system"]["license_scope"]["branch_detection_system_description"] == (
        "BSD-3-Clause"
    )
    assert sources["universal_robots_ros2_description"]["branch"] == "humble"
    assert sources["universal_robots_ros2_description"]["revision"] == (
        "18e6f603b3ebc2ec479fecb62d6be544b15755e9"
    )
    assert "Rolling revision 89bbe795" in sources["universal_robots_ros2_description"]["note"]
    assert sources["ag_robot"]["revision"] == "60b3bee2323ff04d404516c6630db3626cc51fe0"
    assert sources["pybullet_tree_sim"]["license"] == "BSD-3-Clause"
    assert sources["pybullet_tree_sim"]["integration"] == "fetch_only"
    assert sources["follow_the_leader"]["integration"] == "reference_only"
    assert sources["vl53l8cx_firmware"]["revision"] == "0a9ebdf54f3bd2eb892b3fb8b9ae6ca4b30f1d83"
    assert all("revision" in source for source in sources.values())
    assert all(source["revision"] for source in sources.values())


def test_fork_history_source_reports_archive_instead_of_fetching(tmp_path) -> None:
    sources = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["sources"]
    with pytest.raises(RuntimeError, match="preserved as fork history.*joses2017smjh"):
        _fetch("isaaclab_sensor_learning", sources["isaaclab_sensor_learning"], tmp_path)
    assert not any(tmp_path.iterdir())
