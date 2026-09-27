"""Pin the storage policy: the warning line is ours, the refusal is real, and negatives are refused."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def storage(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("research_storage", tools / "research_storage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_lines_are_the_documented_ones(storage):
    assert storage.SOFT_QUOTA_BYTES == 1_649_267_441_664  # Lustre project 30762 soft quota, 1.5 TiB
    assert storage.HARD_LIMIT_BYTES == 2_199_023_255_552  # Lustre project 30762 hard limit, 2 TiB
    # The user's 2026-09-27 choice: refuse at the hard limit, the reserve added inside each projection.
    assert storage.WARNING_BYTES == storage.HARD_LIMIT_BYTES
    report = storage.assess(1_640_000_000_000, 15_000_000_000)
    assert report["ok"] is True and report["over_soft_quota"] is True


def test_the_measurement_that_was_refused_now_passes_and_the_reserve_still_counts(storage):
    # 1.548 TB measured on 2026-09-23 plus the ~0.4 GB matrix and the 20 GB reserve.
    report = storage.assess(1_547_671_741_440, 400 * 1024 * 1024)
    assert report["ok"] is True
    assert report["projected_with_reserve_bytes"] == 1_547_671_741_440 + 400 * 1024 * 1024 + 20_000_000_000
    # Under the previous 1.5 TB line the same projection was refused; that is the change.
    assert storage.assess(1_547_671_741_440, 400 * 1024 * 1024, warning=1_500_000_000_000)["ok"] is False


def test_refusal_happens_at_the_warning_line_not_only_the_hard_limit(storage):
    just_under = storage.WARNING_BYTES - 20_000_000_000 - 1
    assert storage.assess(just_under, 0)["ok"] is True
    assert storage.assess(just_under + 1, 0)["ok"] is False
    assert storage.assess(storage.HARD_LIMIT_BYTES, 0)["ok"] is False


def test_negative_byte_counts_are_refused(storage):
    with pytest.raises(ValueError, match="nonnegative"):
        storage.assess(-1, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        storage.assess(0, -1)
