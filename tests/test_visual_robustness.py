from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")

SPEC = importlib.util.spec_from_file_location(
    "benchmark_visual_robustness", Path(__file__).resolve().parents[1] / "tools/benchmark_visual_robustness.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def _capture(tmp_path, count=16):
    (tmp_path / "frames").mkdir()
    image = np.full((160, 240, 3), 30, dtype=np.uint8)
    image[54:106, 106:134] = np.random.default_rng(7).integers(45, 240, (52, 28, 3), dtype=np.uint8)
    depth = np.full((160, 240), 2.0, dtype=np.float32)
    depth[54:106, 106:134] = 0.6
    Image.fromarray(image).save(tmp_path / "preview_wrist.png")
    np.save(tmp_path / "preview_depth.npy", depth)
    report = {
        "frame_count": count,
        "job_id": "synthetic",
        "task_outcome": "recorded_fixture_only",
        "initial_tool_pose_wxyz": [[0, 0, 0, 1, 0, 0, 0]],
        "camera": {
            "wrist_rotation_in_tool_ros": np.eye(3).tolist(),
            "wrist_position_in_tool_m": [0, 0, 0],
            "wrist_intrinsics": [[160, 0, 120], [0, 160, 80], [0, 0, 1]],
        },
        "vision_initialization": {"pixel_xy": [120, 80], "tracker": {}},
    }
    frames = []
    for index in range(count):
        Image.fromarray(image).save(tmp_path / f"frames/wrist_{index:05d}.png")
        np.save(tmp_path / f"frames/depth_{index:05d}.npy", depth)
        frames.append(
            {
                "index": index,
                "wrist_rotation_w_ros": np.eye(3).tolist(),
                "wrist_position_w_m": [0, 0, 0],
                "target_position_m": [0, 0, 0.6],
                "live_vision": {"cut": {"detached": index >= 13}},
                "detachment_requested_after_capture": index == 13,
            }
        )
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "frames.json").write_text(json.dumps({"frames": frames}))
    return tmp_path


@pytest.mark.parametrize("scenario", benchmark.SCENARIOS)
def test_stress_inputs_are_deterministic_and_do_not_mutate_capture(scenario):
    rgb = np.full((16, 16, 3), 100, dtype=np.uint8)
    depth = np.full((16, 16), 0.6, dtype=np.float32)
    before_rgb, before_depth = rgb.copy(), depth.copy()
    first = benchmark.stress_inputs(rgb, depth, 10, scenario)
    second = benchmark.stress_inputs(rgb, depth, 10, scenario)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(rgb, before_rgb)
    np.testing.assert_array_equal(depth, before_depth)
    assert not np.shares_memory(first[0], rgb)
    assert not np.shares_memory(first[1], depth)
    if scenario == "exposure_transition":
        assert np.all(first[0] == 35)
        assert np.all(benchmark.stress_inputs(rgb, depth, 20, scenario)[0] == 180)


def test_replay_counts_loss_frames_and_handles_release_after_capture(tmp_path):
    source = _capture(tmp_path)
    original_rgb = (source / "frames/wrist_00010.png").read_bytes()
    original_depth = (source / "frames/depth_00010.npy").read_bytes()
    result = benchmark.benchmark_capture(source, ("baseline", "rgb_dropout", "depth_dropout"))
    assert result["first_capture_after_recorded_release"] == 14
    for variant in result["variants"]:
        assert variant["summary"]["attempted_frames"] == 16
        assert len(variant["frames"]) == 16
        assert variant["by_release_period"]["pre_release"]["attempted_frames"] == 14
        assert variant["by_release_period"]["after_release"]["attempted_frames"] == 2
        assert variant["frames"][13]["period"] == "pre_release"
        assert variant["frames"][14]["target_centroid_error_m"] is None
        if variant["scenario"] == "rgb_dropout":
            assert variant["summary"]["tracking_frames"] == 10
            assert variant["summary"]["first_nontracking_frame_index"] == 10
            assert variant["frames"][-1]["reason"] == "explicit_initialization_required"
        elif variant["scenario"] == "depth_dropout":
            assert variant["summary"]["tracking_frames"] == 13
            assert all(row["state"] == "invalid_depth" for row in variant["frames"][10:13])
            assert all(row["state"] == "tracking" for row in variant["frames"][13:])
        else:
            assert variant["summary"]["tracking_frames"] == 16
    assert (source / "frames/wrist_00010.png").read_bytes() == original_rgb
    assert (source / "frames/depth_00010.npy").read_bytes() == original_depth
    json.dumps(result, allow_nan=False)


def test_truth_only_changes_scoring_not_tracking_or_corruptions(tmp_path):
    source = _capture(tmp_path, count=12)
    first = benchmark.benchmark_capture(source, ("baseline", "exposure_transition"))
    document = json.loads((source / "frames.json").read_text())
    for frame in document["frames"]:
        frame["target_position_m"] = [1000, -2000, 3000]
    (source / "frames.json").write_text(json.dumps(document))
    second = benchmark.benchmark_capture(source, ("baseline", "exposure_transition"))
    for before, after in zip(first["variants"], second["variants"]):
        assert before["summary"]["tracking_frames"] == after["summary"]["tracking_frames"]
        assert before["summary"]["centroid_error_median_m"] < after["summary"]["centroid_error_median_m"]
        for old_row, new_row in zip(before["frames"], after["frames"]):
            assert {k: v for k, v in old_row.items() if k not in ("target_centroid_error_m", "update_ms")} == {
                k: v for k, v in new_row.items() if k not in ("target_centroid_error_m", "update_ms")
            }


def test_initialization_failure_keeps_full_attempted_denominator(tmp_path):
    source = _capture(tmp_path, count=3)
    Image.fromarray(np.zeros((160, 240, 3), dtype=np.uint8)).save(source / "preview_wrist.png")
    result = benchmark.benchmark_capture(source, ("baseline",))
    for variant in result["variants"]:
        assert variant["initialization_state"] == "initialization_failed"
        assert variant["summary"]["attempted_frames"] == 3
        assert variant["summary"]["tracking_frames"] == 0


def test_incomplete_capture_is_not_silently_accepted(tmp_path):
    source = _capture(tmp_path, count=3)
    document = json.loads((source / "frames.json").read_text())
    document["frames"].pop()
    (source / "frames.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="denominator"):
        benchmark.benchmark_capture(source, ("baseline",))


def test_recorded_stop_is_separated_from_counterfactual_motion(tmp_path):
    source = _capture(tmp_path, count=5)
    document = json.loads((source / "frames.json").read_text())
    for frame in document["frames"][2:]:
        frame["live_vision"]["external_stop_reason"] = "vision_invalid"
    (source / "frames.json").write_text(json.dumps(document))
    result = benchmark.benchmark_capture(source, ("baseline",))
    assert result["first_recorded_stop_report_frame_index"] == 2
    for variant in result["variants"]:
        assert variant["by_recorded_stop"]["through_first_stop_report"]["attempted_frames"] == 3
        assert variant["by_recorded_stop"]["after_first_stop_report"]["attempted_frames"] == 2
        assert not variant["frames"][2]["after_recorded_stop"]
        assert variant["frames"][3]["after_recorded_stop"]


def test_full_recorded_configuration_is_preserved():
    config = benchmark.VisualServoConfig(max_lk_error=25.0, depth_radius_px=2, replenish_features=True)
    restored = benchmark._source_config({"tracker_config": benchmark.asdict(config)})
    assert restored == config


def test_current_reference_keeps_recorded_configuration_as_separate_provenance(tmp_path):
    source = _capture(tmp_path, count=3)
    report = json.loads((source / "report.json").read_text())
    historical = benchmark.VisualServoConfig(
        feature_quality_level=0.02, replenish_features=False, depth_radius_px=2, max_lk_error=25.0
    )
    report["tracker_config"] = benchmark.asdict(historical)
    (source / "report.json").write_text(json.dumps(report))
    current = benchmark.benchmark_capture(source, ("baseline",))
    recorded = benchmark.benchmark_capture(source, ("baseline",), config_policy="recorded")
    serialized_historical = json.loads(json.dumps(benchmark.asdict(historical)))
    assert current["source_tracker_config"] == recorded["source_tracker_config"] == serialized_historical
    assert current["reference_config_policy"] == "current"
    assert recorded["reference_config_policy"] == "recorded"
    expected = benchmark.asdict(
        benchmark.VisualServoConfig(depth_radius_px=1, feature_quality_level=0.005, replenish_features=True)
    )
    assert current["reference_tracker_config"] == expected
    assert recorded["reference_tracker_config"] == serialized_historical
    for variant in current["variants"]:
        assert variant["config"] == {**expected, "photometric_normalization": variant["mode"]}
