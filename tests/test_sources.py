from __future__ import annotations

from pathlib import Path

import yaml

MANIFEST = Path(__file__).resolve().parents[1] / "third_party" / "sources.yaml"


def test_source_manifest_pins_reviewed_revisions() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    sources = manifest["sources"]

    assert sources["isaaclab_sensor_learning"]["revision"] == ("5701a774af7b8579269f924689aaf79b9574a53c")
    assert sources["isaaclab_sensor_learning"]["integration"] == "fork_history"
    assert sources["branch_detection_system"]["revision"] == ("dfede4c0f251358ebed7a1f90ff887847c2fbeb0")
    assert sources["ag_robot"]["revision"] == "60b3bee2323ff04d404516c6630db3626cc51fe0"
    assert sources["pybullet_tree_sim"]["license"] == "BSD-3-Clause"
    assert sources["pybullet_tree_sim"]["integration"] == "fetch_only"
    assert all("revision" in source for source in sources.values())
    assert all(source["revision"] for source in sources.values())
