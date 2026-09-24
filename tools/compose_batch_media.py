#!/usr/bin/env python3
"""Turn every recorded run of one or more batches into a labelled GIF, MP4 and poster.

CPU only, one run at a time, through ``compose_isaac_workflow.py``. Pass and
fail alike are composed; a run with no capture is listed as skipped, never
invented. Existing media is kept unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

COMPOSER = Path(__file__).resolve().parent / "compose_isaac_workflow.py"


def media_name(run_dir: Path) -> str:
    return run_dir.name.replace("_source", "")


def runs_of(batch: Path):
    return sorted(p for p in batch.glob("run_*") if p.is_dir())


def compose(run_dir: Path, python: str, force: bool = False, dry_run: bool = False):
    name = media_name(run_dir)
    media = run_dir / "media"
    if not (run_dir / "frames.json").is_file() or not (run_dir / "experiment_result.json").is_file():
        return {"run": run_dir.name, "status": "skipped_no_capture"}
    if not force and (media / f"{name}.gif").is_file():
        return {"run": run_dir.name, "status": "kept", "gif": str(media / f"{name}.gif")}
    command = [python, str(COMPOSER), "--input-dir", str(run_dir), "--output-dir", str(media), "--name", name]
    if dry_run:
        return {"run": run_dir.name, "status": "would_compose", "command": command}
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (run_dir / "media_compose.log").write_text(completed.stdout + completed.stderr)
    status = "composed" if completed.returncode == 0 and (media / f"{name}.gif").is_file() else "failed"
    return {
        "run": run_dir.name,
        "status": status,
        "gif": str(media / f"{name}.gif"),
        "returncode": completed.returncode,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, action="append", required=True)
    parser.add_argument("--python", default=sys.executable, help="Interpreter with numpy, PIL and ffmpeg")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    results = [compose(run, args.python, args.force, args.dry_run) for batch in args.batch for run in runs_of(batch)]
    print(json.dumps(results, indent=2))
    return 0 if all(r["status"] != "failed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
