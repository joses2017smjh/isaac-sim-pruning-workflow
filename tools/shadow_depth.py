"""File IPC for frozen DA2 shadow inference; outputs never enter the controller."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from depth_generalization import FrozenDA2, depth_metrics, sanity_gates, target_metrics


class ShadowClient:
    def __init__(self, folder, timeout=180):
        self.folder = Path(folder)
        self.timeout = timeout
        self.rows = []
        self._wait(self.folder / "ready.json")
        self.model = json.loads((self.folder / "ready.json").read_text())
        if not self.model.get("ok"):
            raise RuntimeError(self.model)

    def _wait(self, path):
        deadline = time.monotonic() + self.timeout
        while not path.exists():
            if (self.folder / "error.json").exists():
                raise RuntimeError((self.folder / "error.json").read_text())
            if time.monotonic() > deadline:
                raise TimeoutError(str(path))
            time.sleep(0.01)

    def observe(self, index, rgb, gt, pixel):
        from PIL import Image

        stem = self.folder / f"frame_{index:05d}"
        start = time.perf_counter()
        Image.fromarray(rgb.astype(np.uint8)).save(str(stem) + ".png")
        Path(str(stem) + ".request").touch(exist_ok=False)
        result_path = Path(str(stem) + ".json")
        self._wait(result_path)
        meta = json.loads(result_path.read_text())
        pred = np.load(str(stem) + ".npy", allow_pickle=False)
        row = {
            "index": index,
            "inference_seconds": meta["inference_seconds"],
            "roundtrip_seconds": time.perf_counter() - start,
            "target": target_metrics(pred, gt, pixel),
            "all_valid_gt": depth_metrics(pred, gt),
            "prediction": str(stem) + ".npy",
            "control_depth_source": "RTX distance_to_image_plane",
            "learned_depth_used_for_commands": False,
        }
        self.rows.append(row)
        (self.folder / "evaluation.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "model": self.model,
                    "rows": self.rows,
                    "sanity_gates": sanity_gates(self.rows),
                    "scope": (
                        "Live RGB inference beside RTX-controlled commands; "
                        "simulation paused during synchronous inference"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        return row


def worker(args):
    import cv2

    args.folder.mkdir(parents=True, exist_ok=False)
    try:
        model = FrozenDA2(args.companion, args.checkpoint, args.checkpoint_sha256)
        ready = args.folder / "ready.json.tmp"
        ready.write_text(json.dumps(dict(model.metadata, ok=True)))
        ready.rename(args.folder / "ready.json")
        processed = set()
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and not (args.folder / "stop").exists():
            pending = sorted(set(args.folder.glob("*.request")) - processed)
            if not pending:
                time.sleep(0.01)
                continue
            for request in pending:
                stem = request.with_suffix("")
                rgb = cv2.imread(str(stem) + ".png")
                pred, latency = model.predict(rgb)
                np.save(str(stem) + ".npy", pred)
                temp = Path(str(stem) + ".json.tmp")
                temp.write_text(json.dumps({"inference_seconds": latency}))
                temp.rename(str(stem) + ".json")
                processed.add(request)
        if not processed:
            raise RuntimeError("No live frame received before worker deadline")
    except Exception:
        import traceback

        (args.folder / "error.json").write_text(json.dumps({"error": traceback.format_exc()}))
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--companion", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--timeout", type=int, default=1200)
    worker(p.parse_args())


if __name__ == "__main__":
    main()
