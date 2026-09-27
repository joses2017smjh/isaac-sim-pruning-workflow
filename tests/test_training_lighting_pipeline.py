"""The rendered-lighting launcher registers every frame and the manifest builder keeps only verified frames."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(monkeypatch, name):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _surviving(tmp_path, trees):
    rows = []
    for tree, count in trees.items():
        for i in range(count):
            files = {}
            for kind in ("rgb", "depth", "mask"):
                files[kind] = tmp_path / f"{tree}_{i}_{kind}"
                files[kind].write_bytes(b"x")
            rows.append(
                {
                    "bark": "bark_brown_02",
                    "tree": tree,
                    "set_id": "box" if i % 2 else "box_cam1",
                    "shot": f"shot{i // 2 + 1:02d}",
                    "view": "lr"[i % 2],
                    "rgb_path": files["rgb"],
                    "depth_path": files["depth"],
                    "mask_path": files["mask"],
                }
            )
    manifest = tmp_path / "spur_train.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_plan_registers_every_frame_and_sizes_each_array_by_its_trees(monkeypatch, tmp_path):
    q = _load(monkeypatch, "queue_training_lighting")
    manifest = _surviving(tmp_path, {"lpy_envy_00001": 6, "lpy_envy_00002": 4, "lpy_envy_00004": 4})
    pilot = q.training_plan("pilot", manifest=manifest, hash_assets=False)
    assert list(pilot["trees"]) == ["lpy_envy_00001"] and pilot["registered_frames"] == 6
    assert pilot["arrays"] == [{"frames_per_tree": 6, "trees": ["lpy_envy_00001"], "minutes_per_task": 20}]
    full = q.training_plan("full", manifest=manifest, hash_assets=False)
    assert full["registered_frames"] == 14 and full["frames_to_render"] == 14
    assert [a["frames_per_tree"] for a in full["arrays"]] == [4, 6]
    assert full["maximum_gpu_minutes"] == sum(a["minutes_per_task"] * len(a["trees"]) for a in full["arrays"])
    assert q.task_minutes(90, 50.0, 2) == 90 and q.task_minutes(60, 50.0) == 60
    # A finished pilot is reused, and its measured rate sets the full array's reservation.
    pilot_manifest = tmp_path / "pilot_render_manifest.json"
    pilot_manifest.write_text(
        json.dumps({"smoke": False, "frames": [{"render_seconds": 20.0}, {"render_seconds": 30.0}]})
    )
    reused = q.training_plan("full", manifest=manifest, pilot_manifest=pilot_manifest, hash_assets=False)
    assert reused["trees"]["lpy_envy_00001"]["status"] == "rendered_pilot"
    assert reused["frames_to_render"] == 8 and reused["seconds_per_frame_reserved"] == 50.0
    # Chained behind a pilot that has not run yet: the tree is still excluded, at the default rate.
    pending = q.training_plan("full", manifest=manifest, pilot_manifest=tmp_path / "later.json", hash_assets=False)
    assert pending["trees"]["lpy_envy_00001"]["render_manifest_pending"] is True
    assert pending["frames_to_render"] == 8 and pending["seconds_per_frame_reserved"] == q.DEFAULT_SECONDS_PER_FRAME
    with pytest.raises(ValueError):
        q.training_plan("everything", manifest=manifest, hash_assets=False)


def test_manifest_builder_keeps_verified_frames_and_accounts_for_the_rest(monkeypatch, tmp_path):
    b = _load(monkeypatch, "build_training_lighting_manifest")
    surviving = _surviving(tmp_path, {"lpy_envy_00002": 4})
    tl = _load(monkeypatch, "training_lighting")
    frames = tl.frames_by_tree(tl.surviving_rows(surviving))["lpy_envy_00002"]
    batch = tmp_path / "batch"
    (batch / "render/lpy_envy_00002").mkdir(parents=True)
    (batch / "plan.json").write_text(
        json.dumps({"trees": {"lpy_envy_00002": {"status": "to_render", "frames": frames}}})
    )
    rendered = []
    for i, frame in enumerate(frames[:3]):
        rendered.append(
            {
                **frame,
                "rgb": f"/render/{i}.png",
                "lighting_label": f"rl1_{i:016x}",
                "mean_luma": 60.0 + 30 * i,
                "ok": i != 1,
            }
        )
    (batch / "render/lpy_envy_00002/render_manifest.json").write_text(json.dumps({"smoke": False, "frames": rendered}))
    summary = b.build([batch], surviving, tmp_path / "out")
    assert summary["registered_frames"] == 4 and summary["rendered_frames"] == 3
    assert summary["kept_frames"] == 2 and summary["failed_geometry_check"] == 1 and summary["not_rendered"] == 1
    assert summary["kept_luma"]["fraction_at_or_below_90"] == pytest.approx(0.5)
    combined = list(csv.DictReader((tmp_path / "out/combined.csv").open()))
    assert len(combined) == 4 + 2
    kept = [r for r in combined if r["lighting"].startswith("rl1_")]
    assert all(r["rgb_path"].startswith("/render/") for r in kept)
    assert {r["depth_path"] for r in kept} <= {str(f["depth_path"]) for f in frames}
    # A smoke render never enters a training manifest.
    (batch / "render/lpy_envy_00002/render_manifest.json").write_text(json.dumps({"smoke": True, "frames": rendered}))
    with pytest.raises(ValueError, match="Smoke"):
        b.build([batch], surviving, tmp_path / "out2")
