"""Render measured analytic episodes into a GIF and an offline replay page."""

from __future__ import annotations

import html
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .simulation import rotation_matrix

BG = "#0b1220"
PANEL = "#111e30"
INK = "#e9f1fc"
MUTED = "#9bacc2"
GREEN = "#68e0b1"
AMBER = "#ffd080"
RED = "#ff8d97"
BLUE = "#70baff"
SIZE = (1100, 650)


def _font(size: int):
    path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    return ImageFont.truetype(str(path), size)


def _project(points):
    points = np.asarray(points)
    x, y, z = np.moveaxis(points, -1, 0)
    return np.stack((330 + 660 * (0.90 * x + 0.55 * z), 352 - 350 * (y + 0.24 * z)), axis=-1)


def _line(draw, points, color, width=2):
    projected = _project(points)
    draw.line([tuple(point) for point in projected], fill=color, width=width)


def _box(draw, pose, offset, extents, color):
    rotation = rotation_matrix(pose)
    corners = np.array(list(itertools.product((-1, 1), repeat=3))) * extents + offset
    corners = corners @ rotation.T + pose[:3]
    for a, b in itertools.combinations(range(8), 2):
        if (a ^ b) in (1, 2, 4):
            _line(draw, corners[[a, b]], color, 2)


def _heatmap(draw, values, x, y, label):
    draw.text((x, y), label, font=_font(15), fill=INK)
    array = np.asarray(values, dtype=float)
    for row in range(8):
        for col in range(8):
            depth = array[row, col]
            if np.isfinite(depth):
                t = np.clip(depth / 0.40, 0, 1)
                color = tuple(int(a + t * (b - a)) for a, b in zip((104, 224, 177), (54, 86, 164), strict=True))
            else:
                color = (31, 42, 58)
            left, top = x + 22 * col, y + 28 + 22 * row
            draw.rounded_rectangle((left, top, left + 19, top + 19), radius=3, fill=color)


def render_frame(report, episode, frame_index):
    frame = episode["frames"][frame_index]
    pose = np.array(frame["pose_w"])
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)
    draw.text((30, 24), "PRUNING / FROM RETURNS TO MOTION", font=_font(25), fill=INK)
    draw.text(
        (31, 62),
        "CPU ANALYTIC SIMULATION   /   IDEAL TOOL MOTION   /   SEED " + str(episode["seed"]),
        font=_font(12),
        fill=MUTED,
    )
    draw.rounded_rectangle((24, 102, 650, 510), radius=14, fill=PANEL)
    draw.rounded_rectangle((671, 102, 1076, 510), radius=14, fill=PANEL)
    draw.text((42, 120), episode["title"], font=_font(20), fill=INK)
    phase_color = GREEN if "ACCEPTED" in frame["phase"] else RED if "STOP" in frame["phase"] else AMBER
    draw.text((42, 152), frame["phase"], font=_font(15), fill=phase_color)

    for y in np.linspace(-0.3, 0.3, 7):
        _line(draw, [[-0.40, y, 0.42], [0.13, y, 0.42]], "#203149", 1)
    for cylinder in episode["scene"]:
        center, axis = np.array(cylinder["centroid"]), np.array(cylinder["orientation"])
        endpoints = center + np.array([-1, 1])[:, None] * axis * cylinder["length"] / 2
        color = (
            GREEN if cylinder["record_id"] == "target" else RED if cylinder["record_id"] == "obstacle" else "#8e7768"
        )
        _line(draw, endpoints, color, max(3, int(cylinder["radius"] * 950)))
        if cylinder["record_id"] == "target":
            target_pixel = _project(center)
            draw.text((target_pixel[0] + 15, target_pixel[1] - 24), "target spur", font=_font(12), fill=GREEN)

    trail = [item["pose_w"][:3] for item in episode["frames"][: frame_index + 1]]
    if len(trail) > 1:
        _line(draw, trail, BLUE, 3)
    rotation = rotation_matrix(pose)
    hits = np.array(frame["hit_points_w"])
    for index, offset in enumerate(report["sensor"]["offsets_in_tool_m"]):
        origin = pose[:3] + rotation @ np.array(offset)
        if hits.size:
            for hit in hits[index :: max(1, len(hits) // 6)]:
                _line(draw, [origin, hit], "#32536a", 1)
        u, v = _project(origin)
        draw.ellipse((u - 5, v - 5, u + 5, v + 5), fill=BLUE)
    cutter = report["cutter"]
    _box(draw, pose, np.array(cutter["failure_offset_m"]), np.array(cutter["failure_half_extents_m"]), RED)
    _box(draw, pose, np.array(cutter["mouth_offset_m"]), np.array(cutter["mouth_half_extents_m"]), GREEN)
    tool_pixel = _project(pose[:3])
    draw.ellipse((*tuple(tool_pixel - 4), *tuple(tool_pixel + 4)), fill=INK)
    draw.text((42, 478), "Blue: tool path     Green: mouth     Red: failure volume", font=_font(12), fill=MUTED)

    _heatmap(draw, frame["tof_m"][0], 687, 124, "ToF 0 / 8 x 8")
    _heatmap(draw, frame["tof_m"][1], 886, 124, "ToF 1 / 8 x 8")
    draw.text((687, 343), f"{frame['valid_returns']} / 128 valid returns", font=_font(18), fill=BLUE)
    draw.text((687, 375), "Near: green  /  Far: blue  /  Miss: dark", font=_font(12), fill=MUTED)
    draw.text((687, 404), f"Sensor time  {frame['time_s']:.2f} s  @ 15 Hz", font=_font(15), fill=INK)
    checks = frame["checks"]
    for i, (key, label) in enumerate((("mouth_hit", "Mouth"), ("failure_clear", "Clear"), ("perpendicular", "Angle"))):
        x = 687 + i * 125
        draw.rounded_rectangle((x, 448, x + 113, 484), radius=8, fill="#1c3043")
        draw.text(
            (x + 10, 457),
            label + (" OK" if checks[key] else " --"),
            font=_font(12),
            fill=GREEN if checks[key] else AMBER,
        )

    draw.text((30, 531), f"Target distance  {frame['target_distance_m'] * 1000:.1f} mm", font=_font(18), fill=INK)
    draw.text(
        (687, 531), f"Closing-angle error  {checks['perpendicularity_error_deg']:.2f} deg", font=_font(16), fill=INK
    )
    distances = np.array([item["target_distance_m"] for item in episode["frames"]])
    points = [
        (31 + 610 * j / max(1, len(distances) - 1), 606 - 42 * value / max(distances))
        for j, value in enumerate(distances)
    ]
    draw.line(points, fill="#304359", width=2)
    if frame_index > 0:
        draw.line(points[: frame_index + 1], fill=BLUE, width=3)
    draw.text((687, 564), "Geometry gate only. No arm dynamics or cutting.", font=_font(12), fill=MUTED)
    draw.text((30, 624), "PROCEDURAL WOOD  >  RAY CAST  >  NOISE + SERVO  >  CUT / STOP", font=_font(11), fill=MUTED)
    return image


def write_outputs(report: dict, output_dir: str | Path) -> dict[str, Path]:
    """Save one measured replay, an 18-second GIF, a poster, and offline HTML."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        key: output / f"pruning_demo.{suffix}"
        for key, suffix in (("json", "json"), ("gif", "gif"), ("poster", "png"), ("html", "html"))
    }
    report["playback"] = {"fps": 10, "frames_per_episode": 60, "duration_s": 6 * len(report["episodes"])}
    paths["json"].write_text(json.dumps(report, allow_nan=False) + "\n", encoding="utf-8")
    images = []
    for episode in report["episodes"]:
        indices = np.linspace(0, len(episode["frames"]) - 1, 48).round().astype(int).tolist()
        indices += [len(episode["frames"]) - 1] * 12
        images.extend(render_frame(report, episode, index) for index in indices)
    images[0].save(paths["gif"], save_all=True, append_images=images[1:], duration=100, loop=0, optimize=True)
    render_frame(report, report["episodes"][0], len(report["episodes"][0]["frames"]) - 1).save(paths["poster"])
    payload = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    rows = "".join(
        f"<tr><td>{html.escape(e['title'])}</td><td>{html.escape(e['metrics']['outcome'])}</td>"
        f"<td>{e['metrics']['steps']}</td><td>{e['metrics']['final_target_distance_m'] * 1000:.1f} mm</td></tr>"
        for e in report["episodes"]
    )
    scope = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in report["scope"].items())
    page = Path(__file__).with_name("replay.html").read_text(encoding="utf-8")
    paths["html"].write_text(
        page.replace("ROWS", rows).replace("SCOPE", scope).replace("PAYLOAD", payload), encoding="utf-8"
    )
    return paths
