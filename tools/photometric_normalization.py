#!/usr/bin/env python3
"""Write photometrically normalized copies of rendered frames, for a test-time input study.

The learned depth model collapses under the evening preset, whose frames are
about 2.6x darker than source light (mean luma 55 against 142). This tool asks
the cheapest question first: is that an exposure problem the input pipeline can
undo, or a shading-structure problem it cannot? It produces one normalized copy
of each selected frame per variant, with fixed parameters stated up front, and a
plan manifest per variant that ``depth_generalization.py`` can score unchanged.

Nothing here touches the model or the ground truth. Depth and mask paths are
passed through untouched, so a scored variant differs from the original in the
RGB bytes only. CPU only; scoring the variants is a separate, frozen GPU job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

#: Mean luma of the source-light frames in the registered matrix (see the
#: manifest this tool writes for the measured value used). Exposure and gamma
#: variants target this value; it is a fixed reference, not a per-frame fit
#: against anything the model would not have at run time.
DEFAULT_TARGET_LUMA = 142.0

#: Fixed parameters. The CLAHE settings are the ones the tracker experiment used
#: and rejected on September 19, chosen before either experiment ran.
VARIANTS = {
    "exposure": {"description": "linear gain so mean luma equals the target; clipped at 255"},
    "gamma": {"description": "power-law gamma solved so mean luma equals the target"},
    "hist_eq": {"description": "global histogram equalization of the luma channel"},
    "clahe": {"description": "CLAHE on luma, clipLimit 2.0, tileGridSize 8x8", "clip_limit": 2.0, "tile": 8},
}


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def luma(bgr) -> float:
    import cv2

    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].mean())


def normalize(bgr, variant, target_luma=DEFAULT_TARGET_LUMA):
    """Return the normalized uint8 BGR image and the parameter actually applied."""
    import cv2

    image = np.asarray(bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an 8-bit three-channel BGR image")
    current = luma(image)
    if variant == "exposure":
        gain = target_luma / max(current, 1e-6)
        out = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return out, {"gain": float(gain)}
    if variant == "gamma":
        # Bisect for the exponent that moves the mean luma to the target.
        low, high = 0.05, 20.0
        table = None
        for _ in range(60):
            mid = (low + high) / 2.0
            table = np.clip(255.0 * (np.arange(256) / 255.0) ** mid, 0, 255).astype(np.uint8)
            if luma(table[image]) < target_luma:
                high = mid  # brighter needed: smaller exponent
            else:
                low = mid
        gamma = (low + high) / 2.0
        table = np.clip(255.0 * (np.arange(256) / 255.0) ** gamma, 0, 255).astype(np.uint8)
        return table[image], {"gamma": float(gamma)}
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    if variant == "hist_eq":
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), {}
    if variant == "clahe":
        clahe = cv2.createCLAHE(clipLimit=VARIANTS["clahe"]["clip_limit"], tileGridSize=(8, 8))
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), {"clip_limit": 2.0, "tile": 8}
    raise ValueError(f"Unknown variant: {variant}")


def is_plain_cell(frame, lighting):
    """A frame rendered under ``lighting`` with nothing else changed.

    Plans that carry a ``condition`` (the generalization controls) label every
    changed axis in it; the plain lighting cell is the one whose condition is
    the lighting name itself, so camera and pose controls under the same light
    are never normalized or used as the luma reference.
    """
    return frame.get("lighting") == lighting and frame.get("condition", lighting) == lighting


def select_frames(plan, lighting):
    frames = [f for f in plan["frames"] if is_plain_cell(f, lighting)]
    if not frames:
        raise ValueError(f"No plain frames with lighting={lighting!r} in the plan")
    return frames


def measured_reference(plan, reference_lighting="source"):
    """Mean luma of the reference frames in this plan, recorded for the manifest."""
    import cv2

    values = []
    for frame in plan["frames"]:
        if is_plain_cell(frame, reference_lighting):
            image = cv2.imread(frame["rgb"])
            if image is not None:
                values.append(luma(image))
    return float(np.mean(values)) if values else None


def export(plan_path, output, *, lighting="evening", variants=tuple(VARIANTS), target_luma=None):
    import cv2

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    frames = select_frames(plan, lighting)
    reference = measured_reference(plan)
    if target_luma is None:
        target_luma = reference if reference is not None else DEFAULT_TARGET_LUMA

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "Test-time photometric normalization of rendered RGB only. Depth, masks and targets are passed "
            "through untouched. Parameters are fixed in advance, not tuned to outcomes."
        ),
        "source_plan": str(plan_path),
        "source_plan_sha256": sha256(plan_path),
        "lighting": lighting,
        "frames": len(frames),
        "target_luma": float(target_luma),
        "reference_luma_measured_on_source_frames": reference,
        "variants": {},
    }
    for variant in variants:
        folder = output / variant
        folder.mkdir(parents=True)
        new_frames, params = [], []
        for index, frame in enumerate(frames):
            image = cv2.imread(frame["rgb"])
            if image is None:
                raise ValueError(f"RGB missing: {frame['rgb']}")
            out, applied = normalize(image, variant, target_luma)
            path = folder / f"{index:05d}_{Path(frame['rgb']).name}"
            cv2.imwrite(str(path), out)
            params.append({"index": index, **applied, "luma_before": luma(image), "luma_after": luma(out)})
            new_frame = {**frame, "rgb": str(path), "photometric_normalization": variant, "original_rgb": frame["rgb"]}
            if "condition" in frame:
                new_frame["condition"] = f"{frame['condition']}/{variant}"
                new_frame["condition_group"] = "photometric"
            new_frames.append(new_frame)
        variant_plan = {
            **{k: v for k, v in plan.items() if k != "frames"},
            "frames": new_frames,
            "photometric_normalization": variant,
            "photometric_parameters": VARIANTS[variant],
            "normalization_manifest": str(output / "manifest.json"),
        }
        (folder / "plan.json").write_text(json.dumps(variant_plan, indent=2) + "\n", encoding="utf-8")
        manifest["variants"][variant] = {
            **VARIANTS[variant],
            "plan": str(folder / "plan.json"),
            "mean_luma_after": float(np.mean([p["luma_after"] for p in params])),
            "per_frame": params,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="An evaluation plan.json (prepare_family_evaluation)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lighting", default="evening")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--target-luma", type=float, default=None, help="Default: mean of the plan's source frames")
    args = parser.parse_args(argv)
    manifest = export(
        args.plan, args.output, lighting=args.lighting, variants=tuple(args.variants), target_luma=args.target_luma
    )
    summary = {k: v for k, v in manifest.items() if k != "variants"}
    summary["variants"] = {
        name: {"mean_luma_after": v["mean_luma_after"], "plan": v["plan"]} for name, v in manifest["variants"].items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
