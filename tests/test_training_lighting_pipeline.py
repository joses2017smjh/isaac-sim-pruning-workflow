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


def _registered(tree, count):
    return [
        {
            "bark": "bark_brown_02",
            "tree": tree,
            "set_id": "box",
            "shot": f"shot{i:02d}",
            "view": "l",
            "rgb_path": f"/x/{i}.png",
            "depth_path": f"/x/{i}.npy",
            "mask_path": f"/x/{i}_mask.png",
        }
        for i in range(count)
    ]


def test_resume_keeps_every_earlier_record_and_renders_only_the_missing_frames(monkeypatch):
    tl = _load(monkeypatch, "training_lighting")
    frames = _registered("lpy_envy_00014", 6)
    previous = {"frames": [{**frames[0], "ok": True}, {**frames[2], "ok": False}], "smoke": False}
    kept, remaining = tl.resume_state(previous, frames)
    # A failed check stays on the record and is not rendered again; plan indices survive the resume.
    assert [r["shot"] for r in kept] == ["shot00", "shot02"]
    assert [index for index, _ in remaining] == [1, 3, 4, 5]
    with pytest.raises(ValueError, match="not registered"):
        tl.resume_state({"frames": [{**frames[0], "shot": "shot99"}]}, frames)
    with pytest.raises(ValueError, match="twice"):
        tl.resume_state({"frames": [frames[0], frames[0]]}, frames)
    with pytest.raises(ValueError, match="smoke"):
        tl.resume_state({"frames": [], "smoke": True}, frames)
    with pytest.raises(ValueError, match="smoke"):
        tl.resume_state({"frames": [], "smoke": False}, frames, smoke=True)
    assert tl.resume_state({"frames": [frames[0]], "smoke": True}, frames, smoke=True)[1][0][0] == 1


def test_resume_plan_registers_the_missing_frames_and_the_fine_tunes_that_read_the_batch(monkeypatch, tmp_path):
    q = _load(monkeypatch, "queue_training_lighting")
    target = tmp_path / "generalization/training-lighting-full"
    tree = "lpy_envy_00014"
    frames = _registered(tree, 60)
    (target / "render" / tree).mkdir(parents=True)
    (target / "plan.json").write_text(
        json.dumps(
            {
                "code_revision": "715b811",
                "seconds_per_frame_reserved": 50.0,
                "trees": {tree: {"status": "to_render", "frames": frames}, "done": {"status": "rendered_pilot"}},
            }
        )
    )
    manifest = {"smoke": False, "frames": [{**f, "ok": True} for f in frames[:9]]}
    (target / "render" / tree / "render_manifest.json").write_text(json.dumps(manifest))
    reader = tmp_path / "generalization/finetune-rendered-jitter"
    reader.mkdir(parents=True)
    (reader / "plan.json").write_text(json.dumps({"rendered_batches": [str(target)]}))
    (tmp_path / "generalization/finetune-other").mkdir()
    (tmp_path / "generalization/finetune-other/plan.json").write_text(json.dumps({"rendered_batches": None}))

    consumers = q.consumers_of(target, tmp_path / "generalization")
    assert [c["batch"] for c in consumers] == [str(reader)]
    plan = q.resume_plan(target, [tree], consumers)
    assert plan["trees"][tree]["recorded"] == 9 and plan["trees"][tree]["to_render"] == 51
    # 51 frames at the batch's reserved 50 s plus setup, rounded up to 10 minutes.
    assert plan["arrays"] == [{"trees": [tree], "minutes_per_task": 50}]
    assert plan["maximum_gpu_minutes"] == 50 and plan["frames_to_render"] == 51
    with pytest.raises(ValueError, match="not a tree this batch rendered"):
        q.resume_plan(target, ["done"], consumers)
    manifest["frames"] = [{**f, "ok": True} for f in frames]
    (target / "render" / tree / "render_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nothing to resume"):
        q.resume_plan(target, [tree], consumers)
