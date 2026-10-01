"""Protect the depth-aware appearance replay: it acts only on a grey failure and accepts only a static surface."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]
K = np.array([[160.0, 0, 120], [0, 160.0, 80], [0, 0, 1]])
POSE = np.eye(4)
#: Image rows and columns around the tracked pixel (120, 80) that cover its 13 x 13 patch with a margin.
PATCH_ROWS, PATCH_COLS = slice(72, 89), slice(112, 129)


@pytest.fixture(scope="module")
def depth_replay():
    spec = importlib.util.spec_from_file_location("replay_depth_appearance", ROOT / "tools/replay_depth_appearance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene(dx=0, dy=0, depth_value=0.6):
    """A textured rectangle at ``depth_value`` in front of a 2 m background (the tracker tests' scene)."""
    image = np.full((160, 240, 3), 30, dtype=np.uint8)
    patch = np.random.default_rng(7).integers(45, 240, (52, 28, 3), dtype=np.uint8)
    x, y = 120 + dx, 80 + dy
    image[y - 26 : y + 26, x - 14 : x + 14] = patch
    depth = np.full((160, 240), 2.0, dtype=np.float32)
    depth[y - 26 : y + 26, x - 14 : x + 14] = depth_value
    return image, depth


def _repainted(weight, seed=11):
    """The scene with the patch's own surface repainted: ``weight`` of the old texture is kept, depth unchanged."""
    image, depth = _scene()
    old = image[PATCH_ROWS, PATCH_COLS].astype(float)
    new = np.random.default_rng(seed).integers(45, 240, old.shape).astype(float)
    image[PATCH_ROWS, PATCH_COLS] = np.clip(weight * old + (1 - weight) * new, 0, 255).astype(np.uint8)
    return image, depth


def _started(module, arm):
    """A base tracker and a variant, both seeded and updated once on the unchanged scene (the preview update)."""
    image, depth = _scene()
    base, variant = module.VisualServoTracker(), module.DepthAppearanceTracker(arm=arm)
    for tracker in (base, variant):
        assert tracker.initialize(image, [120, 80], depth)["state"] == "initialized"
    assert base.update(image, depth, K, POSE)["state"] == "tracking"
    assert variant.update(image, depth, K, POSE, jaw_boxes=[])["state"] == "tracking"
    return base, variant


@pytest.mark.parametrize("arm", ["strict", "agreement"])
def test_the_variant_equals_the_repository_tracker_while_the_grey_check_passes(depth_replay, arm):
    base, variant = _started(depth_replay, arm)
    for dx, dy in [(2, 1), (4, 2), (6, 3)]:
        image, depth = _scene(dx, dy)
        expected = base.update(image, depth, K, POSE)
        assert variant.update(image, depth, K, POSE, jaw_boxes=[]) == expected
        assert variant.last_event is None


def test_a_repainted_static_surface_is_accepted_and_strict_continues_iff_its_confidence_reaches_the_gate(depth_replay):
    decisions = set()
    for weight in (0.25, 0.0):
        base, strict = _started(depth_replay, "strict")
        _, agreement = _started(depth_replay, "agreement")
        image, depth = _repainted(weight)
        recorded = base.update(image, depth, K, POSE)
        assert recorded["reason"] == "appearance_changed_or_occluded", recorded
        result = strict.update(image, depth, K, POSE, jaw_boxes=[])
        event = strict.last_event
        assert event["ncc"] < 0.35 and event["accept"], event
        assert result["patch_correlation"] == event["ncc"]
        assert (event["strict_confidence"] >= 0.15) == (result["state"] == "tracking")
        assert event["decision"] == ("continues" if result["state"] == "tracking" else "stops_downstream")
        decisions.add(event["decision"])
        assert agreement.update(image, depth, K, POSE, jaw_boxes=[])["state"] == "tracking"
        assert agreement.last_event["accept"]
    assert decisions == {"continues", "stops_downstream"}


def test_a_nearer_occluder_entering_the_patch_is_rejected(depth_replay):
    base, variant = _started(depth_replay, "strict")
    image, depth = _scene()
    image[60:100, 114:123] = np.random.default_rng(5).integers(0, 256, (40, 9, 3), dtype=np.uint8)
    depth[60:100, 114:123] = 0.5
    recorded = base.update(image, depth, K, POSE)
    result = variant.update(image, depth, K, POSE, jaw_boxes=[])
    event = variant.last_event
    assert recorded["reason"] == "appearance_changed_or_occluded"
    assert not event["accept"]
    assert "3_near_fraction" in event["failed_conditions"]
    assert event["decision"] == "loss_stands"
    assert {key: result[key] for key in ("state", "reason")} == {key: recorded[key] for key in ("state", "reason")}


def test_a_patch_touched_by_the_jaw_silhouette_is_rejected_and_an_unknown_jaw_fails_closed(depth_replay):
    jaw = [(np.array([0.0, 0.0, 0.5]), np.eye(3), np.array([0.003, 0.0125, 0.015]))]
    for boxes, reason in ((jaw, "jaw_silhouette_touches_patch"), (None, "jaw_silhouette_unavailable")):
        _, variant = _started(depth_replay, "strict")
        image, depth = _repainted(0.25)
        variant.update(image, depth, K, POSE, jaw_boxes=boxes)
        event = variant.last_event
        assert not event["accept"]
        assert event["failed_conditions"] == ["J_jaw_silhouette"]
        assert reason in event["reasons"]


def test_the_tracker_exclusion_mask_is_not_part_of_the_variant(depth_replay):
    image, depth = _scene()
    with pytest.raises(ValueError):
        depth_replay.DepthAppearanceTracker().initialize(image, [120, 80], depth, exclusion_mask=np.zeros((160, 240)))


def test_output_inside_the_repository_or_artifacts_or_existing_is_refused(depth_replay, tmp_path):
    assert depth_replay.output_refusal(ROOT / "out.json")
    assert depth_replay.output_refusal(ROOT / "artifacts/out.json")
    existing = tmp_path / "exists.json"
    existing.write_text("{}")
    assert depth_replay.output_refusal(existing)
    assert depth_replay.output_refusal(tmp_path / "new.json") is None


def test_a_continuing_strict_result_reports_the_strict_confidence(depth_replay):
    _, strict = _started(depth_replay, "strict")
    image, depth = _repainted(0.25)
    result = strict.update(image, depth, K, POSE, jaw_boxes=[])
    event = strict.last_event
    assert result["state"] == "tracking", event
    assert result["confidence"] == event["strict_confidence"] >= 0.15
    assert result["confidence_with_gate_value"] > result["confidence"]
    assert event["ended_by"] is None and event["reached_confidence_gate"]


def test_conditions_without_a_value_are_not_evaluated_and_give_no_reason(depth_replay):
    _, strict = _started(depth_replay, "strict")
    image, depth = _repainted(0.25)
    strict.update(image, np.full_like(depth, np.nan), K, POSE, jaw_boxes=[])
    event = strict.last_event
    assert not event["accept"]
    assert event["failed_conditions"] == ["1_verified_fraction"]
    assert set(event["not_evaluated_conditions"]) == {"3_near_fraction", "4_median_abs"}
    assert event["reasons"] == ["unverifiable_depth"]


def test_discovery_is_pinned_to_the_registered_batches(depth_replay, tmp_path):
    for batch in ("jaw-shadow-eve-a-r1-20260930", "tree1-listed-baseline-20260923", "jaw-hold-a-20261001"):
        run = tmp_path / batch / "run_00_x"
        (run / "frames").mkdir(parents=True)
        (run / "frames/wrist_00000.png").write_bytes(b"")
        (run / "report.json").write_text("{}")
        (run / "frames.json").write_text("{}")
    assert [p.parent.name for p in depth_replay.discover_heldout(tmp_path)] == ["jaw-shadow-eve-a-r1-20260930"]
    assert [p.parent.name for p in depth_replay.discover_regression(tmp_path)] == ["tree1-listed-baseline-20260923"]
    assert depth_replay.EXPECTED_RUNS == {"heldout": 12, "regression": 129}


@pytest.mark.parametrize(
    "report",
    [{"stage": "record"}, {"stage": "complete", "jaw_self_mask": {"enabled": True}}],
)
def test_runs_still_recording_or_recorded_with_the_jaw_arms_are_never_replayed(depth_replay, tmp_path, report):
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        depth_replay.replay_run(tmp_path)
