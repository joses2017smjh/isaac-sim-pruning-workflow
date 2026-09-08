#!/usr/bin/env python3
"""Compose recorded Isaac cameras and sensor data into a labelled video dashboard.

Requires NumPy, Pillow and ffmpeg with libx264 (or imageio-ffmpeg). Install opencv-python-headless
for Farneback optical flow; without it, the panel explicitly reports unavailable.
This tool does not simulate images, interpolate robot poses, or infer task success.

Input: frames.json containing {"frames": [{"index": 0, "time_s": 0, ...}]} and
frames/{overview,wrist}_00000.png plus frames/depth_00000.npy. Sensor fields are
tof_left_m / tof_right_m (8 x 8, null for misses), tool_position_m and phase.
The renderer's report.json is carried into the evidence report without reinterpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    import cv2
except ImportError:
    cv2 = None


SIZE = (1440, 960)
BACKGROUND = (12, 19, 29)
PANEL = (22, 33, 46)
INK = (232, 240, 249)
MUTED = (162, 181, 199)
ACCENT = (92, 211, 183)
RANGE_M = (0.03, 3.4)


def _font(size):
    for path in (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _strict_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value {value} in {path}; use null for missing observations")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _vector(value, size=3):
    array = np.asarray(value, dtype=float)
    return array if array.shape == (size,) and np.isfinite(array).all() else None


def tof_array(record, side):
    """Convert JSON nulls to missing pixels, never into zero-range returns."""
    values = np.asarray(record.get(f"tof_{side}_m", [[None] * 8] * 8), dtype=float)
    if values.shape != (8, 8):
        raise ValueError(f"tof_{side}_m must be 8 x 8; got {values.shape}")
    valid = np.isfinite(values) & (values >= RANGE_M[0]) & (values <= RANGE_M[1])
    if f"tof_{side}_valid" in record:
        supplied = np.asarray(record[f"tof_{side}_valid"], dtype=bool)
        if supplied.shape != (8, 8):
            raise ValueError(f"tof_{side}_valid must be 8 x 8")
        valid &= supplied
    return np.where(valid, values, np.nan)


def load_capture(source):
    """Fail before writing output if the capture has missing or mismatched frames."""
    source = Path(source)
    document = _strict_json(source / "frames.json")
    records = document["frames"] if isinstance(document, dict) else document
    if not records:
        raise ValueError("Capture contains no frames")
    previous_time = -math.inf
    seen = set()
    for record in records:
        index = record["index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index in seen:
            raise ValueError("Frame indices must be unique non-negative integers")
        seen.add(index)
        stamp = float(record["time_s"])
        if not math.isfinite(stamp) or stamp < previous_time:
            raise ValueError("Frame time_s must be finite and nondecreasing")
        previous_time = stamp
        for name in ("overview", "wrist"):
            with Image.open(source / "frames" / f"{name}_{index:05d}.png") as image:
                image.verify()
        depth = np.load(source / "frames" / f"depth_{index:05d}.npy", allow_pickle=False)
        if depth.ndim not in (2, 3) or (depth.ndim == 3 and depth.shape[-1] != 1):
            raise ValueError(f"Frame {index}: depth must be H x W or H x W x 1")
        with Image.open(source / "frames" / f"wrist_{index:05d}.png") as wrist:
            if depth.shape[:2] != (wrist.height, wrist.width):
                raise ValueError(f"Frame {index}: wrist RGB and depth dimensions disagree")
        for side in ("left", "right"):
            tof_array(record, side)
    report = _strict_json(source / "report.json") if (source / "report.json").exists() else {}
    return records, report


def range_image(values, limits=RANGE_M):
    """Fixed-range heatmap; dark pixels indicate missing/out-of-range observations."""
    values = np.asarray(values, dtype=float).squeeze()
    valid = np.isfinite(values) & (values >= limits[0]) & (values <= limits[1])
    normalized = np.clip((np.where(valid, values, limits[0]) - limits[0]) / (limits[1] - limits[0]), 0, 1)
    anchors = np.asarray([[250, 195, 86], [81, 198, 179], [55, 112, 183], [48, 51, 106]], dtype=float)
    channels = [np.interp(normalized, np.linspace(0, 1, len(anchors)), anchors[:, axis]) for axis in range(3)]
    rgb = np.stack(channels, axis=-1).astype(np.uint8)
    rgb[~valid] = (9, 13, 20)
    return Image.fromarray(rgb)


def optical_flow(previous, current):
    """Classical vision on recorded RGB; first frame / absent OpenCV is unavailable."""
    if cv2 is None or previous is None:
        return None
    previous = np.asarray(previous.convert("RGB"))
    current = np.asarray(current.convert("RGB"))
    if previous.shape != current.shape:
        raise ValueError("Wrist image dimensions changed during capture")
    return cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(current, cv2.COLOR_RGB2GRAY),
        None,
        0.5,
        3,
        21,
        3,
        5,
        1.2,
        0,
    )


def flow_image(flow, wrist):
    """Display measured flow vectors over the original wrist frame, not a proxy image."""
    output = wrist.convert("RGB").copy()
    if flow is None:
        return output
    draw = ImageDraw.Draw(output)
    stride = max(12, min(output.size) // 14)
    for y in range(stride // 2, output.height, stride):
        for x in range(stride // 2, output.width, stride):
            dx, dy = flow[y, x]
            if np.isfinite((dx, dy)).all() and math.hypot(dx, dy) >= 0.1:
                draw.line((x, y, x + float(dx) * 4, y + float(dy) * 4), fill=ACCENT, width=2)
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 255, 255))
    return output


def brown_candidate(wrist):
    """Color-only bark candidate, not semantic segmentation or target identity."""
    rgb = np.asarray(wrist.convert("RGB"), dtype=float)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (red > 40) & (red > green * 1.15) & (green > blue * 1.05) & (red - blue > 20)
    if cv2 is not None and mask.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if count > 1:
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = labels == component
    y, x = np.where(mask)
    centroid = [float(x.mean()), float(y.mean())] if len(x) >= 20 else None
    return mask, centroid


def candidate_overlay(wrist, mask, centroid):
    output = wrist.convert("RGB").copy()
    if centroid is None:
        return output
    rgb = np.asarray(output).copy()
    rgb[mask] = (rgb[mask] * 0.75 + np.asarray((236, 110, 200)) * 0.25).astype(np.uint8)
    output = Image.fromarray(rgb)
    draw = ImageDraw.Draw(output)
    x, y = centroid
    draw.line((x - 10, y, x + 10, y), fill=(255, 230, 248), width=2)
    draw.line((x, y - 10, x, y + 10), fill=(255, 230, 248), width=2)
    return output


def _paste_fit(canvas, image, box, *, nearest=False):
    x, y, width, height = box
    scale = min(width / image.width, height / image.height)
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    fitted = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), resample)
    ImageDraw.Draw(canvas).rectangle((x, y, x + width, y + height), fill=PANEL)
    canvas.paste(fitted, (x + (width - fitted.width) // 2, y + (height - fitted.height) // 2))


def _text(draw, xy, value, size=18, fill=INK):
    draw.text(xy, str(value), font=_font(size), fill=fill)


def _tof_panel(canvas, record, side, x):
    draw = ImageDraw.Draw(canvas)
    values = tof_array(record, side)
    valid = values[np.isfinite(values)]
    _text(draw, (x, 673), f"ToF {side} | 8 x 8", 19)
    _paste_fit(canvas, range_image(values), (x, 705, 144, 144), nearest=True)
    _text(draw, (x + 151, 710), f"{valid.size}/64", 18, ACCENT)
    _text(draw, (x + 151, 735), "valid", 15, MUTED)
    median = f"{np.median(valid):.2f} m" if valid.size else "no hit"
    _text(draw, (x + 151, 780), median, 16)
    _text(draw, (x + 151, 805), "median", 14, MUTED)


def display_rgb(image, denoise=False):
    """Optional display-only speckle reduction; never used for sensor/CV metrics."""
    return image.filter(ImageFilter.MedianFilter(3)) if denoise else image


def compose_frame(record, overview, wrist, depth, flow, frame_count, *, task_outcome=None, display_denoise=False):
    """Render one dashboard while keeping the actual robot overview prominent."""
    canvas = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    _text(draw, (24, 16), "ROBOTIC PRUNING / ISAAC SIM", 29)
    _text(draw, (24, 55), "Recorded robot + scene + synchronized camera and range measurements", 18, MUTED)
    outcome = str(task_outcome or "not reported").replace("_", " ").upper()
    if task_outcome == "approach_inspect_retreat_no_cut":
        outcome = "INSPECTION COMPLETE / NO CUT"
    outcome_color = (255, 162, 133) if "FAIL" in outcome else MUTED
    _text(draw, (850, 24), f"OUTCOME: {outcome}", 17, outcome_color)
    _text(draw, (1040, 55), f"t = {record['time_s']:.2f} s", 20, ACCENT)
    _text(draw, (1200, 60), f"frame {record['index'] + 1}/{frame_count}", 15, MUTED)
    display_note = " | RGB display: 3 x 3 median" if display_denoise else ""
    _text(draw, (24, 90), "ROBOT OVERVIEW | RTX camera" + display_note, 17, MUTED)
    _paste_fit(canvas, display_rgb(overview, display_denoise), (24, 114, 928, 522))
    phase = str(record.get("phase", "not recorded")).replace("_", " ")
    _text(draw, (24, 642), f"PHASE: {phase.upper()}", 20, ACCENT)
    mask, centroid = brown_candidate(wrist)
    _text(draw, (976, 90), "WRIST RGB | brown-pixel CV candidate", 17, MUTED)
    wrist_display = display_rgb(wrist, display_denoise)
    _paste_fit(canvas, candidate_overlay(wrist_display, mask, centroid), (976, 114, 440, 248))
    _text(draw, (976, 373), "DEPTH | renderer ground truth, not learned", 16, MUTED)
    _paste_fit(canvas, range_image(depth), (976, 398, 440, 244))
    for x, side in ((24, "left"), (266, "right")):
        _tof_panel(canvas, record, side, x)
    _text(draw, (512, 673), "MEASURED STATE", 19)
    tool = _vector(record.get("tool_position_m"))
    target = _vector(record.get("target_position_m"))
    lines = []
    if tool is not None:
        lines.append("Tool xyz [m]: " + ", ".join(f"{v:+.3f}" for v in tool))
    if tool is not None and target is not None:
        lines.append(f"Tool-to-target: {np.linalg.norm(target - tool) * 1000:.1f} mm")
    contact = record.get("contact_force_n")
    if contact is not None:
        values = np.asarray(contact, dtype=float)
        if np.isfinite(values).all():
            lines.append(f"Recorded contact norm: {np.linalg.norm(values):.3f} N")
    if flow is not None:
        lines.append(f"RGB flow mean: {np.linalg.norm(flow, axis=-1).mean():.2f} px/frame")
    lines.append(f"Brown-pixel candidate: {mask.mean() * 100:.1f}% RGB area")
    for index, line in enumerate(lines):
        _text(draw, (512, 706 + index * 28), line, 16, MUTED)
    _text(draw, (976, 673), "COMPUTER VISION | Farneback optical flow", 16, MUTED)
    _paste_fit(canvas, flow_image(flow, wrist_display), (976, 700, 440, 149))
    label = (
        "Vectors x4; pixels per captured frame" if flow is not None else "Flow unavailable: first frame or no OpenCV"
    )
    _text(draw, (976, 852), label, 14, MUTED)
    draw.line((24, 883, 1416, 883), fill=(53, 72, 91))
    _text(draw, (24, 894), "Range colors: near 0.03 m -> far 3.40 m. Dark = missing / out of range.", 16, MUTED)
    _text(
        draw,
        (24, 923),
        "Scripted motion + live ToF stop gate. CV computed offline on raw RGB. No physical wood severing.",
        16,
        MUTED,
    )
    return canvas


def _frame_metrics(record, depth, flow, wrist):
    measured = {"index": record["index"], "time_s": record["time_s"], "phase": record.get("phase")}
    for side in ("left", "right"):
        values = tof_array(record, side)
        finite = values[np.isfinite(values)]
        measured[f"tof_{side}_valid_count"] = int(finite.size)
        measured[f"tof_{side}_median_m"] = float(np.median(finite)) if finite.size else None
    tool, target = _vector(record.get("tool_position_m")), _vector(record.get("target_position_m"))
    measured["tool_target_distance_m"] = (
        float(np.linalg.norm(tool - target)) if tool is not None and target is not None else None
    )
    measured["depth_finite_positive_fraction"] = float(np.mean(np.isfinite(depth) & (depth > 0)))
    measured["flow_mean_px_per_frame"] = float(np.linalg.norm(flow, axis=-1).mean()) if flow is not None else None
    mask, centroid = brown_candidate(wrist)
    measured["brown_candidate_fraction"] = float(mask.mean())
    measured["brown_candidate_centroid_px"] = centroid
    return measured


def _write_gif(frames, output, duration_s, max_bytes):
    """Bound a full-timeline preview; return the sampling and actual encoded duration.

    Temporal and palette reduction matter more than width alone for noisy RTX
    captures. The MP4 remains the full-resolution, every-frame deliverable.
    """
    attempts = (
        (1008, 96, 96),
        (864, 64, 72),
        (720, 48, 64),
        (576, 32, 48),
        (480, 24, 32),
        (480, 12, 24),
        (480, 6, 16),
    )
    for width, frame_limit, colors in attempts:
        positions = np.linspace(0, len(frames) - 1, min(frame_limit, len(frames))).astype(int)
        selected = [frames[index] for index in positions]
        resized = [
            image.resize((width, int(width * SIZE[1] / SIZE[0])), Image.Resampling.LANCZOS) for image in selected
        ]
        palette = resized[len(resized) // 2].quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        indexed = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in resized]
        duration_ms = max(20, round(duration_s * 1000 / len(indexed) / 10) * 10)
        indexed[0].save(output, save_all=True, append_images=indexed[1:], duration=duration_ms, loop=0, optimize=True)
        if output.stat().st_size <= max_bytes:
            with Image.open(output) as saved:
                encoded_duration_ms = 0
                for frame_index in range(saved.n_frames):
                    saved.seek(frame_index)
                    encoded_duration_ms += saved.info.get("duration", 0)
                encoded_frames = saved.n_frames
            return {
                "status": "available",
                "bytes": output.stat().st_size,
                "width_px": width,
                "palette_colors": colors,
                "sample_positions": positions.tolist(),
                "encoded_frames": encoded_frames,
                "duration_s": encoded_duration_ms / 1000,
                "timeline": "Uniform samples across the full capture, with held frames; not real-time frame density",
            }
    raise ValueError(f"Could not fit GIF within {max_bytes} bytes after spatial, temporal and palette reduction")


def _ffmpeg_executable():
    # Distribution ffmpeg builds may omit libx264; the optional wheel bundles it.
    try:
        import imageio_ffmpeg
    except ImportError:
        return shutil.which("ffmpeg")
    return imageio_ffmpeg.get_ffmpeg_exe()


def compose_capture(source, output_dir, *, name="isaac_workflow", fps=None, max_gif_mb=1.9, display_denoise=False):
    """Publish full video/evidence independently of the optional size-bounded GIF."""
    source, output_dir = Path(source), Path(output_dir)
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("name must be a filename stem")
    records, source_report = load_capture(source)
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError("An ffmpeg executable with libx264 is required; install imageio-ffmpeg or system ffmpeg")
    elapsed = float(records[-1]["time_s"] - records[0]["time_s"])
    if fps is None:
        fps = (len(records) - 1) / elapsed if len(records) > 1 and elapsed > 0 else 15.0
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(max_gif_mb) or max_gif_mb <= 0:
        raise ValueError("fps and max_gif_mb must be finite positive values")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_metrics, sampled, sampled_record_indices = [], [], []
    sample_indices = set(np.linspace(0, len(records) - 1, min(96, len(records))).astype(int))
    overview_hashes, wrist_hashes = set(), set()
    digest = hashlib.sha256()
    previous = None
    with tempfile.TemporaryDirectory(prefix="pruning-media-", dir=output_dir) as temporary:
        stage = Path(temporary)
        mp4 = stage / f"{name}.mp4"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{SIZE[0]}x{SIZE[1]}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "medium",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4),
        ]
        with (
            tempfile.TemporaryFile() as errors,
            subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors) as encoder,
        ):
            try:
                for offset, record in enumerate(records):
                    index = record["index"]
                    images = {}
                    for key, hashes in (("overview", overview_hashes), ("wrist", wrist_hashes)):
                        path = source / "frames" / f"{key}_{index:05d}.png"
                        content = path.read_bytes()
                        digest.update(content)
                        hashes.add(hashlib.sha256(content).hexdigest())
                        with Image.open(path) as image:
                            images[key] = image.convert("RGB")
                    depth_path = source / "frames" / f"depth_{index:05d}.npy"
                    digest.update(depth_path.read_bytes())
                    depth = np.load(depth_path, allow_pickle=False).squeeze()
                    flow = optical_flow(previous, images["wrist"])
                    canvas = compose_frame(
                        record,
                        images["overview"],
                        images["wrist"],
                        depth,
                        flow,
                        len(records),
                        task_outcome=source_report.get("task_outcome"),
                        display_denoise=display_denoise,
                    )
                    encoder.stdin.write(canvas.tobytes())
                    if offset in sample_indices:
                        sampled.append(canvas)
                        sampled_record_indices.append(index)
                    frame_metrics.append(_frame_metrics(record, depth, flow, images["wrist"]))
                    previous = images["wrist"]
                encoder.stdin.close()
                return_code = encoder.wait()
            except BrokenPipeError as exc:
                encoder.kill()
                encoder.wait()
                errors.seek(0)
                detail = errors.read().decode(errors="replace")
                raise RuntimeError(f"ffmpeg rejected video encoding: {detail}") from exc
            except BaseException:
                encoder.kill()
                encoder.wait()
                raise
            if return_code:
                errors.seek(0)
                raise RuntimeError(f"ffmpeg failed: {errors.read().decode(errors='replace')}")
        duration_s = len(records) / fps
        sampled[len(sampled) // 2].save(stage / f"{name}.png")
        evidence = {
            "schema_version": 2,
            "source_directory": str(source.resolve()),
            "source_frames_sha256": digest.hexdigest(),
            "source_metadata_sha256": hashlib.sha256((source / "frames.json").read_bytes()).hexdigest(),
            "source_report": source_report,
            "task_outcome": source_report.get("task_outcome"),
            "recorded_frames": len(records),
            "distinct_overview_images": len(overview_hashes),
            "distinct_wrist_images": len(wrist_hashes),
            "simulation_elapsed_s": elapsed,
            "video_duration_s": duration_s,
            "video_fps": fps,
            "display_processing": {
                "rgb": "3 x 3 spatial median" if display_denoise else "none",
                "scope": "Display only; source images, CV inputs, depth and ToF measurements are unchanged",
            },
            "gif_preview": {"status": "pending", "max_bytes": int(max_gif_mb * 1_000_000)},
            "phases": list(dict.fromkeys(record.get("phase", "unknown") for record in records)),
            "channels": {
                "overview": "Recorded Isaac RTX RGB camera",
                "wrist_rgb": "Recorded Isaac RTX wrist camera",
                "depth": "Recorded renderer distance_to_image_plane ground truth; not learned depth",
                "tof": "Recorded dual 8 x 8 ray-cast range observations; missing pixels are not filled",
                "vision": "OpenCV Farneback optical flow from consecutive recorded wrist RGB"
                if cv2 is not None
                else "unavailable: install opencv-python-headless",
                "brown_candidate": "RGB color threshold and largest connected component when OpenCV is available",
            },
            "limitations": [
                "The compositor does not determine task success; inspect source_report and numerical evidence.",
                "Scripted motion and cut phase labels do not establish wood severing or trained policy performance.",
                "Optical flow is pixels per captured frame, not 3-D velocity or a learned depth estimate.",
                "Brown-pixel segmentation can select the wrong object; it does not identify the target.",
                "Range color limits are fixed at 0.03 to 3.40 m; black pixels may also be beyond display range.",
            ],
            "frames": frame_metrics,
        }
        (stage / f"{name}.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        outputs = {}
        for suffix in ("mp4", "png", "json"):
            filename = f"{name}.{suffix}"
            destination = output_dir / filename
            (stage / filename).replace(destination)
            outputs[suffix] = destination
        # A publication preview must never discard the complete recording. Core
        # outputs already exist if encoding is interrupted or the GIF budget is
        # too small. Only include a GIF path when this invocation produced it.
        try:
            gif = _write_gif(sampled, stage / f"{name}.gif", duration_s, int(max_gif_mb * 1_000_000))
            gif["source_frame_indices"] = [sampled_record_indices[index] for index in gif.pop("sample_positions")]
            destination = output_dir / f"{name}.gif"
            (stage / f"{name}.gif").replace(destination)
            outputs["gif"] = destination
        except Exception as exc:
            gif = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
            if (output_dir / f"{name}.gif").exists():
                gif["stale_previous_file"] = f"{name}.gif; not produced by this invocation, do not publish"
            warnings.warn(f"GIF preview unavailable; MP4, poster and evidence preserved: {exc}", stacklevel=2)
        evidence["gif_preview"] = {**gif, "max_bytes": int(max_gif_mb * 1_000_000)}
        refreshed = stage / f"{name}.json"
        refreshed.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        refreshed.replace(outputs["json"])
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=os.environ.get("PRUNING_RENDER_DIR"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="isaac_workflow")
    parser.add_argument("--fps", type=float, help="Playback rate; default derives from captured timestamps")
    parser.add_argument(
        "--display-denoise", action="store_true", help="Median-filter RGB display only; CV uses raw images"
    )
    parser.add_argument(
        "--max-gif-mb", type=float, default=1.9, help="Decimal MB publication limit (default below 2 MB)"
    )
    args = parser.parse_args(argv)
    if args.input_dir is None:
        parser.error("--input-dir or PRUNING_RENDER_DIR is required")
    for kind, path in compose_capture(
        args.input_dir,
        args.output_dir,
        name=args.name,
        fps=args.fps,
        max_gif_mb=args.max_gif_mb,
        display_denoise=args.display_denoise,
    ).items():
        print(f"{kind}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
