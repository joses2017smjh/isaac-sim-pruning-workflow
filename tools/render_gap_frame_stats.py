#!/usr/bin/env python3
"""Measure how the render-gap Matched frames differ from the published matrix frames they replace.

For each of the 192 Matched frames (training-renderer RGB, area-downsampled to 512x288) and the published
matrix frame of the same tree, preset and view (rendered at 512x288, 16 samples, OIDN), both scored against
the same depth and mask: the mean grey level on the tree mask, the variance of the grey Laplacian over the
image (sharpness), and the correlation of the two grey images on the tree mask. Measurement only; not part of
G0-G3 and not pre-registered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATCHED_PLAN = "artifacts/generalization/render-gap-20260930/eval/matched/plan.json"
PUBLISHED_PLAN = "artifacts/generalization/family-matrix-20260923/eval/plan.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(frame):
    return frame["tree_id"], frame["lighting"], frame["view_id"]


def gray(path):
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).astype(np.float64)


def tree_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return mask.any(axis=2) if mask.ndim == 3 else mask > 0


def frame_stats(new_frame, published_frame):
    new, old = gray(new_frame["rgb"]), gray(published_frame["rgb"])
    mask = tree_mask(new_frame["mask"])
    return {
        "same_ground_truth": new_frame["depth"] == published_frame["depth"]
        and new_frame["mask"] == published_frame["mask"],
        "tree_pixels": int(mask.sum()),
        "tree_grey_new": float(new[mask].mean()),
        "tree_grey_published": float(old[mask].mean()),
        "laplacian_variance_new": float(cv2.Laplacian(new, cv2.CV_64F).var()),
        "laplacian_variance_published": float(cv2.Laplacian(old, cv2.CV_64F).var()),
        "tree_grey_correlation": float(np.corrcoef(new[mask], old[mask])[0, 1]),
    }


def summarize(rows):
    def median(field):
        return statistics.median(r[field] for r in rows)

    return {
        "frames": len(rows),
        "median_tree_grey_new": median("tree_grey_new"),
        "median_tree_grey_published": median("tree_grey_published"),
        "frames_with_darker_tree": sum(r["tree_grey_new"] < r["tree_grey_published"] for r in rows),
        "median_sharpness_ratio_new_over_published": statistics.median(
            r["laplacian_variance_new"] / r["laplacian_variance_published"] for r in rows
        ),
        "frames_sharper": sum(r["laplacian_variance_new"] > r["laplacian_variance_published"] for r in rows),
        "median_tree_grey_correlation": median("tree_grey_correlation"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    cv2.setNumThreads(1)
    matched = json.loads((args.root / MATCHED_PLAN).read_text())["frames"]
    published = {key(f): f for f in json.loads((args.root / PUBLISHED_PLAN).read_text())["frames"]}
    rows = [
        {
            "tree_id": f["tree_id"],
            "lighting": f["lighting"],
            "view_id": f["view_id"],
            **frame_stats(f, published[key(f)]),
        }
        for f in matched
    ]
    head = subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "-C", str(args.root), "status", "--porcelain", "--untracked-files=no"])
    )
    document = {
        "schema_version": 1,
        "scope": (
            "Measurement, not pre-registered: how the 192 render-gap Matched frames (training renderer, 1920x1080, "
            "area-downsampled to 512x288) differ in the image from the published matrix frames of the same views "
            "(512x288, 16 samples, OIDN). Both are scored against the same depth and mask."
        ),
        "protocol": "docs/EVAL_PROTOCOL_RENDER_GAP_2026-09-30.md",
        "code_revision": head,
        "code_tree_dirty": dirty,
        "inputs_sha256": {path: sha256(args.root / path) for path in (MATCHED_PLAN, PUBLISHED_PLAN)},
        "all_frames_share_ground_truth": all(r["same_ground_truth"] for r in rows),
        "by_lighting": {
            lighting: summarize([r for r in rows if r["lighting"] == lighting])
            for lighting in sorted({r["lighting"] for r in rows})
        },
        "all": summarize(rows),
        "frames": rows,
    }
    serialized = json.dumps(document, indent=1, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({"all": document["all"], "by_lighting": document["by_lighting"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
