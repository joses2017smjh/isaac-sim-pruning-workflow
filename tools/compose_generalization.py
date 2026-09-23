"""Summarize paired frozen-model results and encode full-frame evidence MP4s."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from depth_generalization import depth_metrics, sha256, target_metrics
from PIL import Image, ImageDraw, ImageFont

LIGHTS = ["source", "morning", "noon", "evening"]


def colorize(values, limit, mask=None):
    scaled = np.nan_to_num(values, nan=0, posinf=limit, neginf=0)
    gray = np.uint8(np.clip(scaled / limit, 0, 1) * 255)
    rgb = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    if mask is not None:
        rgb[~mask] = 0
    return Image.fromarray(rgb)


def render_frame(row_a, row_d, ordinal, total, summary):
    f = row_a["frame"]
    rgb = Image.open(f["rgb"]).convert("RGB")
    gt = np.load(f["depth"])
    a = np.load(row_a["prediction"])
    d = np.load(row_d["prediction"])
    mask = np.asarray(Image.open(f["mask"])) > 0
    if mask.ndim == 3:
        mask = mask.any(axis=2)
    valid = np.isfinite(gt) & (gt > 0) & (gt < 1e6)
    panels = [
        rgb,
        colorize(gt, 5, valid),
        colorize(a, 5),
        colorize(abs(a - gt), 1, mask & valid),
        colorize(d, 5),
        colorize(abs(d - gt), 1, mask & valid),
    ]
    titles = [
        "Rendered RGB (complete frame)",
        "Cycles optical-Z GT | 0-5m",
        "Frozen DA2 | 0-5m",
        "DA2 absolute error on tree | 0-1m",
        "Frozen DINO RGB+D | 0-5m",
        "DINO absolute error on tree | 0-1m",
    ]
    canvas = Image.new("RGB", (1536, 896), (16, 20, 29))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    for i, (panel, title) in enumerate(zip(panels, titles)):
        x, y = (i % 3) * 512, (i // 3) * 320
        draw.text((x + 8, y + 5), title, font=font, fill="white")
        canvas.paste(panel, (x, y + 30))
    pixel = f.get("target_pixel_xy")
    if pixel is not None:
        x, y = pixel
        draw.ellipse((x - 5, y + 25, x + 5, y + 35), outline="yellow", width=2)
    t = target_metrics(a, gt, pixel, radius=0)
    local = (
        f"DA2 target pixel error {t['mae_m'] * 1000:.1f} mm"
        if t and t.get("valid")
        else "Target visibility FAILED / target error unavailable"
    )
    lines = [
        f"{f['family'].upper()} | {f['tree_id']} | {f['lighting']} | {f['view_id']} | view {ordinal}/{total}",
        "STATIC PAIRED-VIEW EVALUATION | Task not executed: learned-control depth gates failed",
        f"{local} | 3x3 patches may mix tiny branches and background; see quantitative JSON",
        "DA2 SHA256 5ecc5182a2717e14... | DINO 2750c4d8f5db7c0d... | same frozen weights in every clip",
        "Tracker: N/A | ToF/contact: N/A | Release: NOT EXECUTED | Grade: OFFLINE ONLY; control not qualified",
        f"Tree-mask pooled RMSE: DA2 {summary['da2']['rmse_m']:.3f}m / DINO {summary['dino']['rmse_m']:.3f}m",
        "Fixed color scales; depth >5m and error >1m saturate for display. Original arrays remain authoritative.",
        "Envy: DA2 validation-tree lighting shift. UFO: absent from task fine-tuning; backbone exposure unknown.",
    ]
    for i, line in enumerate(lines):
        draw.text((12, 650 + i * 29), line, font=font, fill=(240, 240, 240))
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluations", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {}
    sources = []
    for folder in args.evaluations:
        da = json.loads((folder / "da2/evaluation.json").read_text())
        di = json.loads((folder / "dino/evaluation.json").read_text())
        assert da["ok"] and di["ok"]
        mapping = {r["frame"]["rgb"]: r for r in di["rows"]}
        sources.append(
            {
                "evaluation": str(folder),
                "da2_sha256": sha256(folder / "da2/evaluation.json"),
                "dino_sha256": sha256(folder / "dino/evaluation.json"),
            }
        )
        for row in da["rows"]:
            f = row["frame"]
            key = (f["family"], f["lighting"])
            groups.setdefault(key, []).append((row, mapping[f["rgb"]]))
    result = {
        "schema_version": 1,
        "sources": sources,
        "scope": "paired static-view depth evaluation; no population pruning success rate",
        "checkpoint_sha256": {
            "da2": "5ecc5182a2717e14f65dbb9d2d400aac6927435f6e92e87b7e891fb8f7a82834",
            "dino": "2750c4d8f5db7c0d5d40138f08ee9d4bc39a1b184063d33f3dfe9386438ac8ca",
        },
        "groups": {},
        "clips": [],
    }
    for family in ["envy", "ufo"]:
        for light in LIGHTS:
            key = (family, light)
            rows = sorted(groups[key], key=lambda pair: (pair[0]["frame"]["tree_id"], pair[0]["frame"]["view_id"]))
            if args.smoke:
                rows = rows[:2]
            ids = [(r[0]["frame"]["tree_id"], r[0]["frame"]["view_id"]) for r in rows]
            assert len(set(ids)) == len(ids), "Duplicate observations"
            summary = {
                "family": family,
                "lighting": light,
                "tree_ids": sorted({k[0] for k in ids}),
                "views": len(rows),
                "visible_target_views": sum(r[0]["frame"].get("target_visible", False) for r in rows),
                "pixel_metrics_scope": (
                    "Pooled valid GT pixels on whole-tree mask; target pixel metrics separate; no GT calibration"
                ),
            }
            per_tree = {}
            for mi, model in enumerate(["da2", "dino"]):
                ps = []
                gs = []
                targets = []
                latencies = []
                for pair in rows:
                    row = pair[mi]
                    f = row["frame"]
                    gt = np.load(f["depth"])
                    pred = np.load(row["prediction"])
                    mask = np.asarray(Image.open(f["mask"])) > 0
                    if mask.ndim == 3:
                        mask = mask.any(axis=2)
                    ps.append(pred[mask])
                    gs.append(gt[mask])
                    per_tree.setdefault(f["tree_id"], {}).setdefault(model, []).append(
                        depth_metrics(pred, gt, mask)["rmse_m"]
                    )
                    if f.get("target_visible"):
                        target = target_metrics(pred, gt, f.get("target_pixel_xy"), radius=0)
                        if target and target.get("valid"):
                            targets.append(target["mae_m"])
                    latencies.append(row["inference_seconds"] if mi == 0 else row["six_view_total_inference_seconds"])
                summary[model] = depth_metrics(np.concatenate(ps)[None], np.concatenate(gs)[None])
                summary[model]["target_pixel_mae_m"] = float(np.mean(targets)) if targets else None
                summary[model]["target_pixel_p95_abs_m"] = float(np.quantile(targets, 0.95)) if targets else None
                summary[model]["latency_p95_seconds"] = float(np.quantile(latencies, 0.95))
                summary[model]["latency_scope"] = (
                    "single-view DA2" if mi == 0 else "six-view DA2 + DINO group; excludes acquisition"
                )
            summary["per_tree_mean_frame_rmse_m"] = {
                tree: {m: float(np.mean(v)) for m, v in models.items()} for tree, models in per_tree.items()
            }
            result["groups"][family + "_" + light] = summary
            video = args.output / f"{family}_{light}.mp4"
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "1536x896",
                "-r",
                "10",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libopenh264",
                "-b:v",
                "8M",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ]
            proc = subprocess.Popen(command, stdin=subprocess.PIPE)
            for i, pair in enumerate(rows):
                frame = render_frame(*pair, i + 1, len(rows), summary).tobytes()
                for _ in range(10):
                    proc.stdin.write(frame)
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("Video encoding failed")
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,width,height,nb_frames,duration",
                        "-of",
                        "json",
                        str(video),
                    ]
                )
            )
            assert int(probe["streams"][0]["nb_frames"]) == len(rows) * 10
            clip = {
                "clip": len(result["clips"]) + 1,
                "tree_family": family,
                "tree_ids": summary["tree_ids"],
                "lighting": light,
                "model": "frozen DA2 + six-view DINO RGB+D",
                "checkpoint_sha256": result["checkpoint_sha256"],
                "depth_source": ["Cycles optical-Z GT", "learned DA2", "learned DINO"],
                "job_ids": sorted(
                    {json.loads((p / "da2/evaluation.json").read_text())["job_id"] for p in args.evaluations}
                ),
                "task_outcome": "not executed: learned-control depth gates failed",
                "grade": "offline only; control not qualified",
                "video": str(video),
                "video_sha256": sha256(video),
                "probe": probe,
                "static_views": True,
            }
            result["clips"].append(clip)
            (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"clip": str(video), "summary": summary}), flush=True)
            if args.smoke:
                return


if __name__ == "__main__":
    main()
