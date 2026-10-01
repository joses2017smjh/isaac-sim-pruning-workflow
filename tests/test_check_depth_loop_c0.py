"""Protect the C0 comparator: floats within the tolerance, every discrete field exact."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def c0(monkeypatch):
    tools = ROOT / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("check_depth_loop_c0", tools / "check_depth_loop_c0.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_floats_agree_within_the_tolerance_and_the_largest_difference_is_kept(c0):
    compare = c0.Comparison()
    assert compare.equal({"ncc": 0.22001418, "pixel": [1.0, 2.0]}, {"ncc": 0.22001424, "pixel": [1.0, 2.0]})
    assert compare.max_float_difference == pytest.approx(6e-8)
    assert not compare.equal(0.2, 0.2 + 2e-6)
    assert not compare.equal([1.0, 2.0], [1.0])
    assert not compare.equal({"a": 1.0}, {"a": 1.0, "b": 2.0})


def test_discrete_fields_must_match_exactly(c0):
    compare = c0.Comparison()
    assert compare.equal({"accept": True, "n": 20, "reason": None}, {"accept": True, "n": 20, "reason": None})
    assert not compare.equal(True, 1) and not compare.equal(None, 0.0) and not compare.equal(False, None)
    assert not compare.equal(20, 21) and not compare.equal("continues", "stops_downstream")
    assert not compare.equal(["3_near_fraction"], ["4_median_abs"])


def test_a_live_event_is_flattened_like_the_published_record_and_context_fields_are_ignored(c0):
    live = c0.flat_event({"accept": True, "ncc": 0.2, "stats": {"median_abs_m": 0.0001, "n_verified": 160}})
    published = {
        "frame": 67,
        "accept": True,
        "ncc": 0.2000001,
        "median_abs_m": 0.0001,
        "n_verified": 160,
        "recorded": {},
    }
    assert c0.event_problems(published, live) == []
    assert c0.event_problems({**published, "n_verified": 159}, live) == ["n_verified"]
    assert c0.event_problems(published, None) == ["no live event"]
    assert c0.flat_event(None) is None
