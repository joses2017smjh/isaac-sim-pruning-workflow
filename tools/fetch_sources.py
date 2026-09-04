#!/usr/bin/env python3
"""Fetch external robot and orchard sources at reviewed immutable revisions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "third_party" / "sources.yaml"
DEFAULT_DESTINATION = REPOSITORY_ROOT / "third_party" / "src"


def _run(*command: str, capture: bool = False) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def _load_sources(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(f"{path} does not contain a sources mapping.")
    return sources


def _head(path: Path) -> str:
    return _run("git", "-C", str(path), "rev-parse", "HEAD", capture=True)


def _is_dirty(path: Path) -> bool:
    return bool(_run("git", "-C", str(path), "status", "--porcelain", capture=True))


def _fetch(name: str, source: dict, destination: Path) -> None:
    if source.get("integration") == "fork_history":
        archive = source.get("archive_repository", "the current fork")
        raise RuntimeError(
            f"{name} is preserved as fork history, not a fetchable dependency. "
            f"Inspect revision {source['revision']} in {archive}."
        )
    revision = str(source["revision"])
    target = destination / name
    if target.exists():
        if not (target / ".git").is_dir():
            raise RuntimeError(f"{target} exists but is not a git checkout.")
        if _is_dirty(target):
            raise RuntimeError(f"{target} has local changes; refusing to replace them.")
        _run("git", "-C", str(target), "fetch", "--depth=1", "origin", revision)
        _run("git", "-C", str(target), "checkout", "--detach", revision)
    else:
        _run(
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            str(source["repository"]),
            str(target),
        )
        _run("git", "-C", str(target), "checkout", "--detach", revision)
    print(f"{name}: {_head(target)}")


def _check(name: str, source: dict, destination: Path) -> bool:
    target = destination / name
    expected = str(source["revision"])
    if not (target / ".git").is_dir():
        print(f"{name}: missing", file=sys.stderr)
        return False
    actual = _head(target)
    matched = actual == expected
    print(f"{name}: {actual} ({'ok' if matched else f'expected {expected}'})")
    return matched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--source", action="append", help="Fetch only this named source; repeatable")
    parser.add_argument("--check", action="store_true", help="Only verify existing checkout SHAs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = _load_sources(args.manifest)
    selected = args.source or [name for name, source in sources.items() if source.get("integration") == "fetch_only"]
    unknown = sorted(set(selected) - sources.keys())
    if unknown:
        raise ValueError(f"Unknown source names: {', '.join(unknown)}")

    args.destination.mkdir(parents=True, exist_ok=True)
    if args.check:
        return 0 if all(_check(name, sources[name], args.destination) for name in selected) else 1

    for name in selected:
        source = sources[name]
        if source.get("license") == "NOASSERTION":
            print(
                f"warning: {name} has no declared license; keep it fetch-only until permission is documented",
                file=sys.stderr,
            )
        _fetch(name, source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
