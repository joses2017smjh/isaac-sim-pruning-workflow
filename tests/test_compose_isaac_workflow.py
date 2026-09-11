from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from tools import compose_isaac_workflow as composer


def _capture(tmp_path, count=3):
    source = tmp_path / "capture"
    frames = source / "frames"
    frames.mkdir(parents=True)
    records = []
    for index in range(count):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[10:36, 15 + index : 39 + index] = (170, 100, 50)
        for camera in ("overview", "wrist"):
            Image.fromarray(rgb).save(frames / f"{camera}_{index:05d}.png")
        depth = np.full((48, 64), 0.7, dtype=np.float32)
        depth[0, 0] = np.inf
        np.save(frames / f"depth_{index:05d}.npy", depth)
        left = [[0.7] * 8 for _ in range(8)]
        left[0][0] = None
        records.append(
            {
                "index": index,
                "time_s": index / 15,
                "phase": "approach" if index < 2 else "hold",
                "tof_left_m": left,
                "tof_right_m": [[None] * 8 for _ in range(8)],
                "tool_position_m": [0.0, 0.0, index * 0.001],
                "target_position_m": [0.0, 0.0, 0.1],
            }
        )
    (source / "frames.json").write_text(json.dumps({"frames": records}), encoding="utf-8")
    (source / "report.json").write_text(
        json.dumps({"outcome": "test fixture, not Isaac evidence", "task_outcome": "stopped_failure"}),
        encoding="utf-8",
    )
    return source, records


def test_capture_contract_rejects_missing_frames_before_output(tmp_path):
    source, _ = _capture(tmp_path)
    (source / "frames" / "wrist_00001.png").unlink()
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        composer.compose_capture(source, output)
    assert not output.exists()


def test_capture_rejects_duplicate_indices(tmp_path):
    source, records = _capture(tmp_path)
    records[1]["index"] = 0
    (source / "frames.json").write_text(json.dumps({"frames": records}), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        composer.load_capture(source)


def test_tof_missing_pixels_are_not_valid_zero_ranges():
    record = {"tof_left_m": [[None] * 8 for _ in range(8)]}
    record["tof_left_m"][2][3] = 0.5
    record["tof_left_m"][1][1] = 0.0
    result = composer.tof_array(record, "left")
    assert np.isfinite(result).sum() == 1
    assert result[2, 3] == 0.5
    assert composer.range_image(result).getpixel((0, 0)) == (9, 13, 20)


def test_tof_honors_supplied_validity_mask():
    record = {"tof_left_m": [[0.5] * 8 for _ in range(8)], "tof_left_valid": [[False] * 8 for _ in range(8)]}
    assert not np.isfinite(composer.tof_array(record, "left")).any()


def test_flow_explicitly_unavailable_without_opencv(monkeypatch):
    monkeypatch.setattr(composer, "cv2", None)
    image = Image.new("RGB", (64, 48))
    assert composer.optical_flow(image, image) is None


@pytest.mark.skipif(composer.cv2 is None, reason="OpenCV optional")
def test_flow_measures_actual_image_translation():
    rng = np.random.default_rng(6)
    first = (rng.random((80, 80, 3)) * 255).astype(np.uint8)
    second = np.roll(first, 2, axis=1)
    flow = composer.optical_flow(Image.fromarray(first), Image.fromarray(second))
    assert np.median(flow[15:-15, 15:-15, 0]) == pytest.approx(2.0, abs=0.15)
    assert abs(np.median(flow[15:-15, 15:-15, 1])) < 0.15


def test_dashboard_preserves_real_overview_content(tmp_path):
    _, records = _capture(tmp_path)
    overview = Image.new("RGB", (928, 522), (101, 22, 203))
    wrist = Image.new("RGB", (64, 48), (14, 87, 30))
    canvas = composer.compose_frame(records[0], overview, wrist, np.full((48, 64), 0.7), None, 3)
    assert canvas.size == (1440, 960)
    assert canvas.getpixel((400, 400)) == (101, 22, 203)


def test_brown_candidate_measures_image_pixels_not_oracle():
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[10:30, 20:40] = (170, 100, 50)
    mask, centroid = composer.brown_candidate(Image.fromarray(rgb))
    assert int(mask.sum()) == 400
    assert centroid == [29.5, 19.5]
    _, missing = composer.brown_candidate(Image.new("RGB", (64, 48), (100, 100, 100)))
    assert missing is None


def test_display_denoise_does_not_change_raw_rgb_or_cv_measurements():
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[10:30, 20:40] = (170, 100, 50)
    rgb[4, 4] = (255, 255, 255)
    raw = Image.fromarray(rgb)
    before = composer.brown_candidate(raw)
    display = composer.display_rgb(raw, denoise=True)
    assert display.getpixel((4, 4)) == (0, 0, 0)
    assert raw.getpixel((4, 4)) == (255, 255, 255)
    assert np.array_equal(np.asarray(raw), rgb)
    after = composer.brown_candidate(raw)
    assert np.array_equal(before[0], after[0])
    assert before[1] == after[1]


@pytest.mark.skipif(composer._ffmpeg_executable() is None, reason="No system or bundled ffmpeg encoder")
def test_media_outputs_include_strict_measured_evidence(tmp_path):
    source, _ = _capture(tmp_path)
    outputs = composer.compose_capture(source, tmp_path / "output", display_denoise=True)
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    evidence = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert evidence["recorded_frames"] == 3
    assert evidence["distinct_wrist_images"] == 3
    assert evidence["frames"][0]["tof_left_valid_count"] == 63
    assert evidence["frames"][0]["tof_right_median_m"] is None
    assert evidence["frames"][0]["flow_mean_px_per_frame"] is None
    assert evidence["source_report"]["outcome"] == "test fixture, not Isaac evidence"
    assert evidence["task_outcome"] == "stopped_failure"
    assert evidence["display_processing"]["rgb"] == "3 x 3 spatial median"
    assert evidence["video_duration_s"] == pytest.approx(0.2)
    assert outputs["gif"].stat().st_size <= 1_900_000
    assert evidence["gif_preview"]["status"] == "available"
    assert evidence["gif_preview"]["source_frame_indices"] == [0, 1, 2]


@pytest.mark.skipif(composer._ffmpeg_executable() is None, reason="No system or bundled ffmpeg encoder")
def test_gif_failure_preserves_video_poster_and_evidence(tmp_path, monkeypatch):
    source, _ = _capture(tmp_path)

    def fail_gif(*args):
        raise ValueError("preview budget exhausted")

    monkeypatch.setattr(composer, "_write_gif", fail_gif)
    with pytest.warns(UserWarning, match="MP4, poster and evidence preserved"):
        outputs = composer.compose_capture(source, tmp_path / "output")
    assert set(outputs) == {"mp4", "png", "json"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    evidence = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert evidence["task_outcome"] == "stopped_failure"
    assert evidence["gif_preview"]["status"] == "unavailable"
    assert "preview budget exhausted" in evidence["gif_preview"]["reason"]


def test_noisy_gif_adapts_sampling_without_hiding_capture_endpoints(tmp_path):
    rng = np.random.default_rng(19)
    frames = [Image.fromarray(rng.integers(0, 256, (120, 180, 3), dtype=np.uint8)) for _ in range(24)]
    output = tmp_path / "noise.gif"
    result = composer._write_gif(frames, output, duration_s=14, max_bytes=1_900_000)
    assert output.stat().st_size <= 1_900_000
    assert result["sample_positions"][0] == 0
    assert result["sample_positions"][-1] == len(frames) - 1
    assert result["width_px"] <= 864
    assert result["duration_s"] == pytest.approx(14, abs=0.15)


def _live_evidence(*, stopped=False):
    return {
        "measurement": {
            "state": "tracking",
            "confidence": 0.81,
            "pixel_xy": [32.0, 24.0],
            "feature_pixels_xy": [[12.0, 12.0], [-100.0, 2.0], [200.0, 20.0]],
            "target_position_world_m": [0.1, 0.2, 0.3],
        },
        "measurement_time_s": 0.2,
        "cut": {
            "phase": "stopped" if stopped else "closing",
            "closure_progress": 0.5,
            "detach_event": False,
            "detached": False,
            "stopped_reason": "hazard_contact" if stopped else None,
            "certificate": {"mouth_distance_m": 0.006, "reasons": [], "ready_to_close": not stopped},
        },
        "vision_command_count": 7,
    }


@pytest.mark.parametrize("stopped,color", [(False, composer.ACCENT), (True, (255, 112, 112))])
def test_live_tracking_overlay_draws_recorded_pixels_and_stop_color(stopped, color):
    raw = Image.new("RGB", (64, 48), (20, 30, 40))
    overlay = composer.live_tracking_overlay(raw, _live_evidence(stopped=stopped))
    assert overlay.getpixel((32, 24)) == color
    assert overlay.getpixel((12, 10)) == color
    assert overlay.getpixel((60, 46)) == (20, 30, 40)
    assert raw.getpixel((32, 24)) == (20, 30, 40)


def test_live_tracking_does_not_invent_missing_target():
    raw = Image.new("RGB", (64, 48), (20, 30, 40))
    overlay = composer.live_tracking_overlay(raw, {"measurement": None, "cut": None})
    assert np.array_equal(np.asarray(raw), np.asarray(overlay))
    lines = composer._live_state_lines({"live_vision": {"measurement": None, "cut": None}})
    assert "awaiting image" in lines[0]
    assert "Gate: not reported" in lines
    assert "source frame: none" in lines[-1]


def test_live_dashboard_labels_causal_vision_and_keeps_offline_flow_separate(tmp_path, monkeypatch):
    _, records = _capture(tmp_path)
    record = {
        **records[1],
        "live_vision": _live_evidence(stopped=True),
        "controller_source_frame_index": 0,
        "visual_servo_decision": {"state": "hold", "reason": "hazard_contact"},
    }
    drawn_text = []
    original_text = composer._text

    def record_text(draw, xy, value, *args, **kwargs):
        drawn_text.append(str(value))
        return original_text(draw, xy, value, *args, **kwargs)

    def reject_brown_candidate(*args):
        pytest.fail("Live display must not substitute the offline brown-pixel heuristic")

    monkeypatch.setattr(composer, "_text", record_text)
    monkeypatch.setattr(composer, "brown_candidate", reject_brown_candidate)
    rgb = Image.new("RGB", (64, 48), (20, 30, 40))
    depth = np.full((48, 64), 0.7)
    canvas = composer.compose_frame(record, rgb, rgb, depth, None, 3, task_outcome="vision_stopped_failure")
    assert canvas.size == composer.SIZE
    assert any("LIVE RGB-D TRACKING" in line for line in drawn_text)
    assert any("OFFLINE DIAGNOSTIC" in line for line in drawn_text)
    assert any("VISION STOPPED / FAILURE" in line for line in drawn_text)
    assert any("Gate: hazard contact" in line for line in drawn_text)
    assert any("scene-selected branch" in line for line in drawn_text)
    assert not any("CV computed offline" in line or "brown-pixel" in line for line in drawn_text)
    measured = composer._frame_metrics(record, depth, None, rgb)
    assert measured["live_vision"] == record["live_vision"]
    assert measured["controller_source_frame_index"] == 0
    assert measured["visual_servo_decision"] == record["visual_servo_decision"]
    assert "brown_candidate_centroid_px" not in measured


@pytest.mark.skipif(composer._ffmpeg_executable() is None, reason="No system or bundled ffmpeg encoder")
def test_live_media_preserves_failure_report_and_controller_evidence(tmp_path):
    source, records = _capture(tmp_path)
    for record in records:
        record["live_vision"] = _live_evidence(stopped=True)
        record["controller_source_frame_index"] = record["index"] - 1 if record["index"] else None
        record["visual_servo_decision"] = {"state": "hold", "reason": "hazard_contact"}
    report = {"ok": True, "task_outcome": "vision_stopped_failure", "checks": {"detached": False}}
    (source / "frames.json").write_text(json.dumps({"frames": records}), encoding="utf-8")
    (source / "report.json").write_text(json.dumps(report), encoding="utf-8")
    outputs = composer.compose_capture(source, tmp_path / "live_media")
    evidence = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert evidence["source_report"] == report
    assert evidence["task_outcome"] == "vision_stopped_failure"
    assert evidence["live_vision_frames"] == 3
    assert "online RGB-D tracker" in evidence["channels"]["vision"]
    assert "Offline" in evidence["channels"]["offline_optical_flow"]
    assert "brown_candidate" not in evidence["channels"]
    assert evidence["frames"][1]["controller_source_frame_index"] == 0
    assert evidence["frames"][1]["live_vision"] == records[1]["live_vision"]
    assert any("not actuated CAD blades or wood fracture" in line for line in evidence["limitations"])
    assert json.loads((source / "report.json").read_text(encoding="utf-8")) == report
