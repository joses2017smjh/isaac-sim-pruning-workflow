"""Protect the jaw-in-view scorer (P1-P8 of docs/EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md).

Covered: each prediction at its registered support/refute boundary, the untested and reported-only cases, failed,
incomplete and refused runs kept in every table, the registered run set, the frozen-code import, the disclosure of
what was seen before scoring, and the refusal to overwrite an output. Every record here is synthetic; no recording
is read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
A, B, C = "jaw-hold-a-20261001", "jaw-hold-b-20261001", "jaw-hold-c-20261001"
K = 74  # closure start of every synthetic 530 run
TOOL = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
PIECE = (0.0, 0.0, 1.07, 1.0, 0.0, 0.0, 0.0)
WAIVED = [
    "vision_invalid",
    "vision_stale_or_future",
    "vision_timestamp_regressed",
    "vision_frame_reused_during_closure",
]
#: The baseline target's shift (m) from its own value at k, frame by frame: exactly 3 mm at k+3.
MOVES = {K + offset: 0.001 * offset for offset in range(1, 7)}


@pytest.fixture
def scorer(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("score_jaw_in_view", TOOLS / "score_jaw_in_view.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------- synthetic runs


def _progress(step, k, stop, detach):
    """The cut's closure progress after its update at ``step``: what the runner renders into frame step + 1."""
    if k is None or step < k:
        return 0.0
    if stop is not None and step >= stop:
        return _progress(stop - 1, k, None, detach)
    if step >= k + 6:
        return 1.0 if detach else 5 / 6
    return (step - k) / 6


def raw_run(k=K, loss=K + 2, attributable=True, hold=True, hold_until=K + 6, detach=True, cut_stop=None, length=200):
    """frames.json records of a synthetic run: approach, align from k-4, closure at k, the first honest loss at
    ``loss`` (None: tracking throughout), held from it through ``hold_until`` when ``hold``, detach at k+6 unless
    the cut stops first (``cut_stop`` = (frame, reason)); ``k=None`` never closes."""
    if k is not None and not detach and cut_stop is None:
        cut_stop = (k + 6, "gate_lost_during_closure")
    stop, stop_reason = cut_stop or (None, None)

    def cut_phase(j):
        if stop is not None and j >= stop:
            return "stopped"
        if k is None or j < k - 4:
            return "approach"
        if j < k:
            return "align"
        return "closing" if j < k + 6 else "retreat"

    frames = []
    for i in range(length):
        cut, before = cut_phase(i), cut_phase(i - 1) if i else "approach"
        lost = loss is not None and i >= loss
        if lost and i == loss and attributable:
            state, reason, kept = "jaw_mask_occluded", "too_few_unmasked_patch_pixels", 130
        elif lost and i == loss:
            state, reason, kept = "tracking_lost", "appearance_changed_or_occluded", 169
        elif lost:
            state, reason, kept = "tracking_lost", "explicit_initialization_required", None
        else:
            state, reason, kept = "tracking", None, 169
        eligible = k is not None and k + 1 <= i <= k + 6 and (stop is None or stop >= i)
        held = hold and eligible and lost and attributable and i <= hold_until
        explanation = None
        if lost and attributable and i == loss:
            explanation = "too_few_unmasked_patch_pixels"
        elif held:
            explanation = "latched_by_held_jaw_loss"
        age = (i - k) / 10 if held else None
        shadow = "stopped" if lost or cut == "stopped" else cut
        detaching = cut == "retreat" and i == k + 6
        frames.append(
            {
                "index": i,
                "phase": "vision_approach",
                "visual_servo_decision": {"state": "tracking", "approach_phase": "final" if i >= 50 else "standoff"}
                if before == "approach"
                else {"state": "hold"},
                "visual_jaw_closure_progress": _progress(i - 1, k, stop, detach),
                "detachment_requested_after_capture": detaching,
                "tool_pose_wxyz": list(TOOL),
                "selected_piece_pose_wxyz": list(PIECE)
                if k is None or i <= k + 6 or not detach
                else [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                "live_vision": {
                    "approach_phase": "final" if i >= 50 else "standoff",
                    "measurement": {
                        "state": state,
                        "reason": reason,
                        "pixel_xy": [240.0, 160.0],
                        "target_position_world_m": [0.0, 0.0, 1.07] if state == "tracking" else None,
                        "jaw_mask_patch_unmasked": kept,
                        "jaw_mask_patch_masked_prev": 0,
                        "jaw_mask_patch_masked_cur": 0 if kept in (None, 169) else 169 - kept,
                        "jaw_mask_dropped_valid": 0,
                        "jaw_mask_in_mask_any": 0,
                    },
                    "cut": {
                        "phase": cut,
                        "stopped_reason": stop_reason if cut == "stopped" else None,
                        "detach_event": detaching,
                        "closure_started_s": None if k is None or i < k else (k + 1) / 10,
                        "certificate": {
                            "mouth_distance_m": 0.003,
                            "vision_source": "closure_hold" if held else "live",
                            "waived_checks": list(WAIVED) if held else [],
                            "held_target_age_s": age,
                        },
                    },
                    "closure_hold": {
                        "eligible": eligible,
                        "stationary": eligible,
                        "dt_mm": 0.0,
                        "dr_deg": 0.0,
                        "explained": explanation is not None,
                        "explanation": explanation,
                        "held": held,
                        "held_target_age_s": age,
                        "latched_by_jaw": held,
                        "measurement_state": state,
                    },
                    "cut_without_hold": {
                        "phase": shadow,
                        "stopped_reason": ("vision_invalid" if lost else stop_reason) if shadow == "stopped" else None,
                    },
                    "jaw_self_mask": {"mask_pixel_count": 5000},
                    "mouth_offset_tool_m": [0.0, 0.0, 0.07],
                    "closing_axis_tool": [1.0, 0.0, 0.0],
                },
            }
        )
    return frames


def raw_19444(stop=67, touch=66, attributable=True, length=200):
    """19444: the jaw first masks patch elements at ``touch``; the cut stops at ``stop`` on the final leg."""
    frames = raw_run(
        k=None, loss=stop, attributable=attributable, hold=False, cut_stop=(stop, "vision_invalid"), length=length
    )
    if touch is not None:
        frames[touch]["live_vision"]["measurement"].update(jaw_mask_patch_unmasked=150, jaw_mask_patch_masked_cur=19)
    return frames


def p1_rows(n=200):
    """P1 rows as p1_frame returns them for frames whose mask and depth agree with the reconstruction."""
    return [
        {
            "index": i,
            "rendered_progress": 0.0,
            "recorded_count": 5000,
            "reconstructed_count": 5000,
            "count_equal": True,
            "progress_echo_equal": True,
            "roll_difference_rad": 0.0,
            "roll_equal": True,
            "bbox_equal": True,
            "corner_max_abs_px": 0.0,
            "depth_checked": True,
            "hit_pixels": 4800,
            "violations": 0,
            "nonfinite_hits": 0,
            "max_excess_m": 1e-5,
            "edge_band_pixels": 10,
            "violations_outside_edge_band": 0,
            "edge_band_beyond": [],
            "nearer_pixels": 0,
            "hit_outside_mask": 0,
        }
        for i in range(n)
    ]


def baseline(moves=None, states=None, drop=()):
    """P7's replayed baseline rows: tracking, its target ``moves[i]`` metres from the origin (its value at k)."""
    moves = MOVES if moves is None else moves
    states = states or {}
    rows = []
    for i in range(200):
        if i in drop:
            continue
        state = states.get(i, "tracking")
        target = [moves.get(i, 0.0), 0.0, 0.0] if state == "tracking" else None
        rows.append({"index": i, "state": state, "reason": None, "target": target})
    return {
        "flags": {"jaw_self_mask": False, "closure_hold": False},
        "rows": rows,
        "initialization": {"matches": True},
        "recorded_closure_start_frame": K,
        "reproduces_recording_through_recorded_stop": False,
        "tracker_config_matches_recording": True,
    }


def record(scorer, batch, index, target, raw, **fields):
    """A scorer record as run_record builds it, from synthetic frames.json records (``raw=None``: not recorded)."""
    facts = None if raw is None else [scorer.frame_facts(frame) for frame in raw]
    for fact in facts or []:
        fact.update(sd_px=5.0, sd_px_minus_half=5.0)  # the tracked pixel lies outside the jaw silhouette
    out = {
        "batch": batch,
        "run_index": index,
        "run_directory": f"run_{index:02d}_source_tree{int(target != 530)}_v{target}_planned_pose_jaw_hold",
        "run_path": f"/nonexistent/{batch}",
        "target": target,
        "target_tree_index": int(target != 530),
        "strategy": "planned_pose_jaw_hold",
        "outcome": "pass",
        "status": "graded",
        "checks_passed": 17,
        "checks_total": 17,
        "failed_checks": [],
        "stop_reason": None,
        "stop_frame": None,
        "node": "cn-gpu5",
        "gpu_model": "RTX 8000",
        "configuration_matches": True,
        "refused": False,
        "refused_reasons": [],
        "configuration_problems": [],
        "fps": 10,
        "report": None,
        "frames": facts,
        "not_recorded_reason": None if raw is not None else "no report.json and no frames.json",
        "p1": None if raw is None else scorer.p1_summary(p1_rows(len(raw)), len(raw)),
        "baseline": None,
        "inputs": {},
    }
    out.update(fields)
    return out


def incomplete(scorer, batch, index, target):
    return record(
        scorer, batch, index, target, None, outcome="incomplete", status="incomplete", checks_passed=0, checks_total=0
    )


def run_530(scorer, batch, raw=None, **fields):
    fields.setdefault("baseline", baseline())
    return record(scorer, batch, 0, 530, raw_run() if raw is None else raw, **fields)


def six(scorer, **replace):
    """The six registered runs, each behaving as predicted, unless ``replace[key]`` (a530 ... c15004) is given."""
    build = {
        "a530": lambda: run_530(scorer, A),
        "a19444": lambda: record(
            scorer, A, 1, 19444, raw_19444(), outcome="stopped_vision_invalid", checks_passed=12, stop_frame=68
        ),
        "b530": lambda: run_530(scorer, B),
        "b14944": lambda: record(scorer, B, 1, 14944, raw_run(loss=None, hold=False)),
        "c530": lambda: run_530(scorer, C),
        "c15004": lambda: record(scorer, C, 1, 15004, raw_run(loss=None, hold=False)),
    }
    return [replace[key] if key in replace else make() for key, make in build.items()]


def all_530(scorer, **kwargs):
    """The three 530 runs replaced by the same synthetic run."""
    fields = {key: kwargs.pop(key) for key in ("baseline",) if key in kwargs}
    batches = (("a", A), ("b", B), ("c", C))
    return six(scorer, **{f"{key}530": run_530(scorer, batch, raw_run(**kwargs), **fields) for key, batch in batches})


def verdict(scorer, records, pid):
    return next(p for p in scorer.score(records) if p["id"] == pid)


def verdicts(scorer, records):
    return {p["id"]: p["verdict"] for p in scorer.score(records)}


# ---------------------------------------------------------------------------------------------- all predictions


def test_the_predicted_pattern_supports_every_prediction(scorer):
    assert verdicts(scorer, six(scorer)) == {f"P{i}": "supported" for i in range(1, 9)}


# ---------------------------------------------------------------------------------------------- P1


def test_box_depth_matches_the_analytic_box(scorer):
    camera = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    shape = (100, 100)
    rays = scorer.pixel_rays(camera, shape)
    box = (np.array([0.0, 0.0, 0.2]), np.eye(3), np.array([0.01, 0.01, 0.01]))
    # A pixel-centre ray has x/z = (u + 0.5 - 50) / 100; it enters the front face z = 0.19 exactly when
    # |x/z| <= 0.01 / 0.19 in both axes, i.e. u, v in 45..54, and its optical Z there is 0.19.
    depth = scorer.box_depth([box], np.eye(4), rays, shape)
    expected = np.zeros(shape, bool)
    expected[45:55, 45:55] = True
    assert (np.isfinite(depth) == expected).all()
    assert np.abs(depth[expected] - 0.19).max() < 1e-12
    # Moving the camera back 0.1 m puts the front face at optical Z 0.29: |u + 0.5 - 50| <= 100 * 0.01 / 0.29 = 3.45,
    # so u, v in 47..52.
    transform = np.eye(4)
    transform[2, 3] = -0.1
    depth = scorer.box_depth([box], transform, rays, shape)
    expected = np.zeros(shape, bool)
    expected[47:53, 47:53] = True
    assert (np.isfinite(depth) == expected).all()
    assert np.abs(depth[expected] - 0.29).max() < 1e-12


def test_depth_test_counts_every_hit_beyond_1mm_and_reports_the_edge_band_apart(scorer):
    camera = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    shape = (100, 100)
    rays = scorer.pixel_rays(camera, shape)
    # Front face at z = 0.19; its x edges pass through the centres of columns 45 and 54 (|x/z| = 0.045), so those
    # columns are hits within the 0.01 px band; rows 45..54 lie at least 0.44 px inside the y edges.
    box = (np.array([0.0, 0.0, 0.2]), np.eye(3), np.array([0.045 * 0.19 * (1 + 1e-9), 0.0094, 0.01]))
    analytic = scorer.box_depth([box], np.eye(4), rays, shape)
    hit = np.isfinite(analytic)
    assert hit.sum() == 100 and hit[45:55, 45:55].all()
    recorded = np.where(hit, analytic, 5.0)
    recorded[50, 50] += 0.0009  # within 1 mm: not a violation
    recorded[51, 51] += 0.0011  # more than 1 mm beyond: a violation
    recorded[45:55, 54] = 5.0  # the background seen through every edge-band pixel of column 54
    result = scorer.depth_test([box], camera, np.eye(4), rays, shape, np.ones(shape, bool), recorded)
    assert result["violations"] == 11  # every hit counts: the registered reading
    assert result["violations_outside_edge_band"] == 1  # the sensitivity
    assert result["edge_band_pixels"] == 20
    assert len(result["edge_band_beyond"]) == 10
    assert all(abs(pixel["signed_distance_px"]) < scorer.EDGE_BAND_PX for pixel in result["edge_band_beyond"])
    recorded[51, 51] = analytic[51, 51]
    result = scorer.depth_test([box], camera, np.eye(4), rays, shape, np.ones(shape, bool), recorded)
    assert result["violations"] == 10 and result["violations_outside_edge_band"] == 0


def test_depth_test_counts_only_strictly_more_than_1mm(scorer, monkeypatch):
    camera = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    shape = (100, 100)
    rays = scorer.pixel_rays(camera, shape)
    box = (np.array([0.0, 0.0, 0.2]), np.eye(3), np.array([0.01, 0.01, 0.01]))
    hit = np.isfinite(scorer.box_depth([box], np.eye(4), rays, shape))
    # A stand-in box depth of 0.001 m makes the excess exact: 0.002 - 0.001 == 0.001 in binary floating point.
    monkeypatch.setattr(scorer, "box_depth", lambda *args: np.where(hit, 0.001, np.inf))
    assert scorer.DEPTH_TOLERANCE_M == 0.002 - 0.001
    recorded = np.where(hit, 0.002, 5.0)
    result = scorer.depth_test([box], camera, np.eye(4), rays, shape, np.ones(shape, bool), recorded)
    assert result["hit_pixels"] == 100 and result["violations"] == 0  # exactly 1 mm beyond is not more than 1 mm
    recorded[hit] = np.nextafter(0.002, 1.0)
    result = scorer.depth_test([box], camera, np.eye(4), rays, shape, np.ones(shape, bool), recorded)
    assert result["violations"] == 100


def _jaw_frame(scorer, count_delta=0, roll_delta_ulps=1):
    """A frame whose jaw lies in front of a wrist camera at the tool origin, with its controller mask record."""
    geo = {
        "camera_matrix": np.array([[320.0, 0.0, 240.0], [0.0, 320.0, 160.0], [0.0, 0.0, 1.0]]),
        "shape": (320, 480),
        "roll_rad": 0.3,
        "radius_m": 0.005,
    }
    pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    boxes = scorer.jaw_boxes(pose, geo["roll_rad"], 0.0, geo["radius_m"])
    mask = scorer.jaw_mask(boxes, geo["camera_matrix"], np.eye(4), geo["shape"])
    summary = scorer.jaw_mask_summary(boxes, mask, geo["camera_matrix"], np.eye(4))
    roll = geo["roll_rad"]
    for _ in range(roll_delta_ulps):
        roll = float(np.nextafter(roll, 1.0))
    frame = {
        "index": 0,
        "visual_jaw_closure_progress": 0.0,
        "tool_pose_wxyz": pose.tolist(),
        "wrist_position_w_m": [0.0, 0.0, 0.0],
        "wrist_rotation_w_ros": np.eye(3).tolist(),
        "live_vision": {
            "jaw_self_mask": {
                "mask_pixel_count": summary["mask_pixel_count"] + count_delta,
                "closure_progress_used": 0.0,
                "roll_rad": roll,
                "mask_bbox_xyxy": summary["mask_bbox_xyxy"],
                "corners_px": summary["corners_px"],
            }
        },
    }
    return geo, frame


def test_p1_frame_reconstructs_the_count_and_allows_a_one_ulp_roll(scorer):
    geo, frame = _jaw_frame(scorer)
    rays = scorer.pixel_rays(geo["camera_matrix"], geo["shape"])
    row, jaw = scorer.p1_frame(geo, rays, frame, None)
    assert jaw is not None and row["recorded_count"] > 0
    assert row["count_equal"] and row["roll_equal"] and row["progress_echo_equal"] and row["bbox_equal"]
    assert 0 < abs(row["roll_difference_rad"]) <= scorer.ROLL_TOLERANCE_RAD
    geo, frame = _jaw_frame(scorer, count_delta=1, roll_delta_ulps=0)
    row, _ = scorer.p1_frame(geo, rays, frame, None)
    assert not row["count_equal"] and "margin_slack_px" in row
    frame["live_vision"]["jaw_self_mask"]["mask_pixel_count"] -= 1
    frame["live_vision"]["jaw_self_mask"]["roll_rad"] = geo["roll_rad"] + 1e-9
    row, _ = scorer.p1_frame(geo, rays, frame, None)
    assert row["count_equal"] and not row["roll_equal"]


def test_p1_refuted_by_one_count_mismatch_or_one_depth_violation(scorer):
    records = six(scorer)
    rows = p1_rows()
    rows[50] = {**rows[50], "recorded_count": 5001, "count_equal": False}
    records[0]["p1"] = scorer.p1_summary(rows, 200)
    assert verdict(scorer, records, "P1")["verdict"] == "refuted"
    records = six(scorer)
    rows = p1_rows()
    rows[60] = {**rows[60], "violations": 1, "violations_outside_edge_band": 1, "max_excess_m": 0.0011}
    records[2]["p1"] = scorer.p1_summary(rows, 200)
    assert verdict(scorer, records, "P1")["verdict"] == "refuted"


def test_p1_a_violation_in_the_edge_band_refutes_and_the_sensitivity_is_only_reported(scorer):
    records = six(scorer)
    rows = p1_rows()
    edge = {"v": 242, "u": 155, "signed_distance_px": -3e-5, "box_z_m": 0.1396, "recorded_m": None, "excess_m": None}
    rows[13] = {**rows[13], "violations": 1, "violations_outside_edge_band": 0, "edge_band_beyond": [edge]}
    records[1]["p1"] = scorer.p1_summary(rows, 200)
    p1 = verdict(scorer, records, "P1")
    assert p1["verdict"] == "refuted" and p1["clauses"][1]["refutes"]
    depth = p1["clauses"][1]["observed"]
    assert depth["runs_with_violations"] == [f"{A} 19444"] and depth["violations"] == 1
    sensitivity = depth["edge_band_sensitivity"]
    assert sensitivity["runs_with_violations_outside_edge_band"] == [] and sensitivity["would_refute"] is False
    run = depth["runs"][f"{A} 19444"]
    assert run["violation_frames"] == [
        {"index": 13, "violations": 1, "nonfinite_hits": 0, "within_edge_band": 1, "max_excess_mm": 0.01}
    ]
    assert run["edge_band_sensitivity"]["edge_band_beyond"] == [{"index": 13, **edge}]


def test_p1_roll_mismatch_and_missing_frames_keep_p1_from_support_without_refuting(scorer):
    records = six(scorer)
    rows = p1_rows()
    rows[5] = {**rows[5], "roll_equal": False, "roll_difference_rad": 1e-9}
    records[0]["p1"] = scorer.p1_summary(rows, 200)
    assert verdict(scorer, records, "P1")["verdict"] == "partly supported"
    records = six(scorer)
    records[0]["p1"] = scorer.p1_summary(p1_rows(150), 150)  # 150 of the planned 200 frames
    assert verdict(scorer, records, "P1")["verdict"] == "partly supported"


# ---------------------------------------------------------------------------------------------- P2


def _control(scorer, raw, **fields):
    return record(scorer, B, 1, 14944, raw, **fields)


def test_p2_loss_through_the_detach_frame_refutes_and_after_it_does_not(scorer):
    raw = raw_run(loss=None, hold=False)
    raw[K + 6]["live_vision"]["measurement"]["state"] = "invalid_depth"
    assert verdict(scorer, six(scorer, b14944=_control(scorer, raw)), "P2")["verdict"] == "refuted"
    raw = raw_run(loss=None, hold=False)
    raw[K + 7]["live_vision"]["measurement"]["state"] = "tracking_lost"
    assert verdict(scorer, six(scorer, b14944=_control(scorer, raw)), "P2")["verdict"] == "supported"


def test_p2_one_masked_patch_element_or_a_hold_refutes(scorer):
    raw = raw_run(loss=None, hold=False)
    raw[10]["live_vision"]["measurement"]["jaw_mask_patch_unmasked"] = 168
    assert verdict(scorer, six(scorer, b14944=_control(scorer, raw)), "P2")["verdict"] == "refuted"
    raw = raw_run(loss=None, hold=False)
    raw[K + 3]["live_vision"]["closure_hold"]["held"] = True
    assert verdict(scorer, six(scorer, c15004=_control(scorer, raw)), "P2")["verdict"] == "refuted"
    raw = raw_run(loss=None, hold=False)
    raw[K + 9]["live_vision"]["closure_hold"]["held"] = True  # after detach: "the hold never applies" has no window
    assert verdict(scorer, six(scorer, c15004=_control(scorer, raw)), "P2")["verdict"] == "refuted"


def test_p2_a_failed_check_refutes_but_an_ungraded_or_refused_control_only_withholds_support(scorer):
    failed = _control(scorer, raw_run(loss=None, hold=False), outcome="stopped_x", checks_passed=16)
    assert verdict(scorer, six(scorer, b14944=failed), "P2")["verdict"] == "refuted"
    p2 = verdict(scorer, six(scorer, b14944=incomplete(scorer, B, 1, 14944)), "P2")
    assert p2["verdict"] == "partly supported"
    assert f"{B} 14944" in p2["clauses"][0]["observed"]
    refused = _control(scorer, raw_run(loss=None, hold=False), refused=True, refused_reasons=["closure_hold off"])
    assert verdict(scorer, six(scorer, b14944=refused), "P2")["verdict"] == "partly supported"


# ---------------------------------------------------------------------------------------------- P3


def test_p3_tracking_inside_the_silhouette_refutes(scorer):
    records = six(scorer)
    records[1]["frames"][30]["sd_px"] = -0.1
    assert verdict(scorer, records, "P3")["verdict"] == "refuted"
    records = six(scorer)
    records[1]["frames"][30]["sd_px"] = 0.0  # on the undilated outline counts as inside
    assert verdict(scorer, records, "P3")["verdict"] == "refuted"


def test_p3_an_unexplained_stop_or_align_after_the_touch_refutes(scorer):
    unexplained = record(scorer, A, 1, 19444, raw_19444(attributable=False))
    assert verdict(scorer, six(scorer, a19444=unexplained), "P3")["verdict"] == "refuted"
    records = six(scorer)
    records[1]["frames"][66]["cut_phase"] = "align"
    assert verdict(scorer, records, "P3")["verdict"] == "refuted"


def test_p3_is_untested_when_the_jaw_never_reaches_the_patch(scorer):
    early = record(scorer, A, 1, 19444, raw_19444(touch=None, attributable=False))
    p3 = verdict(scorer, six(scorer, a19444=early), "P3")
    assert p3["verdict"] == "untested" and "before the jaw reached the patch" in p3["untested_reason"]
    neither = record(scorer, A, 1, 19444, raw_run(k=None, loss=None, hold=False))
    p3 = verdict(scorer, six(scorer, a19444=neither), "P3")
    assert p3["verdict"] == "untested" and "no registered refutation condition applies" in p3["untested_reason"]
    p3 = verdict(scorer, six(scorer, a19444=incomplete(scorer, A, 1, 19444)), "P3")
    assert p3["verdict"] == "untested" and p3["untested_reason"] == "19444 was not recorded"


def test_p3_a_stop_off_the_final_leg_withholds_support_without_refuting(scorer):
    records = six(scorer)
    records[1]["frames"][67]["leg"] = "standoff"
    p3 = verdict(scorer, records, "P3")
    assert p3["verdict"] == "partly supported" and not p3["clauses"][0]["holds"]
    assert not any(c["refutes"] for c in p3["clauses"])


def test_p3_final_leg_falls_back_to_the_previous_live_leg(scorer):
    records = six(scorer)
    records[1]["frames"][67]["leg"] = None  # the decision carries no approach_phase
    p3 = verdict(scorer, records, "P3")
    assert p3["verdict"] == "supported"
    assert p3["clauses"][0]["observed"]["leg_source"].startswith("live approach_phase of the previous frame")


# ---------------------------------------------------------------------------------------------- P4


def test_p4_first_loss_at_another_frame_or_unattributable_refutes(scorer):
    assert verdict(scorer, six(scorer, a530=run_530(scorer, A, raw_run(loss=K + 3))), "P4")["verdict"] == "refuted"
    unexplained = run_530(scorer, A, raw_run(attributable=False, hold=False))
    assert verdict(scorer, six(scorer, a530=unexplained), "P4")["verdict"] == "refuted"
    early = run_530(scorer, A, raw_run(loss=K + 1, hold_until=K + 6))
    assert verdict(scorer, six(scorer, a530=early), "P4")["verdict"] == "refuted"


def test_p4_four_millimetres_is_predicted_and_above_is_reported_only(scorer):
    def at(mouth):
        records = six(scorer)
        for r in records[0::2]:
            k_fact = next(f for f in r["frames"] if f["index"] == K)
            k_fact["mouth_distance_m"] = mouth
        return verdict(scorer, records, "P4")

    assert at(0.004)["verdict"] == "supported"
    above = at(0.0040001)
    assert above["verdict"] == "untested"
    assert above["untested_reason"] == "no 530 run started closure with a gate mouth distance of at most 4 mm"
    assert len(above["clauses"][6]["observed"]) == 3  # every run reported, not predicted


def test_p4_a_closure_that_never_rendered_the_jaw_closing_is_untested_not_refuted(scorer):
    unstable = run_530(scorer, A, raw_run(loss=None, hold=False, cut_stop=(K + 1, "unstable_during_closure")))
    p4 = verdict(scorer, six(scorer, a530=unstable), "P4")
    assert p4["verdict"] == "partly supported"
    p4 = verdict(scorer, all_530(scorer, loss=None, hold=False, cut_stop=(K + 1, "unstable_during_closure")), "P4")
    assert p4["verdict"] == "untested" and "never rendered closing" in p4["untested_reason"]


def test_p4_truncated_recordings_are_untested_unless_tracking_reaches_k_plus_3(scorer):
    p4 = verdict(scorer, all_530(scorer, length=K + 2), "P4")
    assert p4["verdict"] == "untested" and "end before P4 is settled" in p4["untested_reason"]
    assert verdict(scorer, all_530(scorer, loss=None, hold=False, length=K + 4), "P4")["verdict"] == "refuted"


# ---------------------------------------------------------------------------------------------- P5


def test_p5_a_held_frame_that_stops_or_a_missing_detachment_refutes(scorer):
    stops = run_530(scorer, A, raw_run(cut_stop=(K + 4, "gate_lost_during_closure")))
    p5 = verdict(scorer, six(scorer, a530=stops), "P5")
    assert p5["verdict"] == "refuted" and p5["clauses"][4]["refutes"]
    uncovered = run_530(scorer, A, raw_run(hold_until=K + 5, cut_stop=(K + 6, "vision_invalid")))
    p5 = verdict(scorer, six(scorer, a530=uncovered), "P5")
    assert p5["verdict"] == "refuted" and p5["clauses"][3]["refutes"] and not p5["clauses"][4]["refutes"]
    # Stopped on an unheld frame before the deadline, recording ending there: no detachment can follow, so it refutes.
    truncated = run_530(scorer, A, raw_run(hold_until=K + 4, cut_stop=(K + 5, "vision_invalid"), length=K + 6))
    p5 = verdict(scorer, six(scorer, a530=truncated), "P5")
    assert p5["verdict"] == "refuted" and p5["clauses"][3]["refutes"] and not p5["clauses"][4]["refutes"]
    assert p5["clauses"][3]["observed"][f"{A} 530"]["recorded_through_deadline"] is False
    # Recording ending before the deadline with the cut still closing and nothing settled: not evaluable, no verdict.
    open_end = run_530(scorer, A, raw_run(length=K + 5))
    p5 = verdict(scorer, six(scorer, a530=open_end), "P5")
    assert p5["verdict"] == "partly supported" and not any(c["refutes"] for c in p5["clauses"])


def test_p5_tool_and_mouth_limits_withhold_support_without_refuting(scorer):
    records = six(scorer)
    records[0]["frames"][K + 3]["tool_pose"] = [0.0006, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert verdict(scorer, records, "P5")["verdict"] == "partly supported"
    records = six(scorer)
    records[0]["frames"][K + 4]["mouth_distance_m"] = 0.003 + 0.00004
    assert verdict(scorer, records, "P5")["verdict"] == "supported"
    records[0]["frames"][K + 4]["mouth_distance_m"] = 0.003 + 0.00006
    assert verdict(scorer, records, "P5")["verdict"] == "partly supported"


def test_p5_tool_limits_are_strict_at_half_a_millimetre_and_a_quarter_degree(scorer):
    def tool_at(pose):
        records = six(scorer)
        records[0]["frames"][K + 3]["tool_pose"] = pose
        return verdict(scorer, records, "P5")["verdict"]

    def turned(degrees):
        half = np.radians(degrees) / 2
        return [0.0, 0.0, 1.0, float(np.cos(half)), float(np.sin(half)), 0.0, 0.0]

    assert tool_at([0.00049, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]) == "supported"
    assert tool_at([0.0005, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]) == "partly supported"  # strict: 0.5 mm is not within
    assert tool_at(turned(0.249)) == "supported"
    assert tool_at(turned(0.251)) == "partly supported"


def test_p5_the_mouth_limit_includes_exactly_five_hundredths_of_a_millimetre(scorer):
    records = six(scorer)
    for fact in records[0]["frames"][K : K + 7]:
        fact["mouth_distance_m"] = 0.0
    records[0]["frames"][K + 4]["mouth_distance_m"] = 0.00005  # exactly 0.05 mm from the closure-start value
    assert verdict(scorer, records, "P5")["verdict"] == "supported"


def test_p5_a_mislabelled_held_frame_withholds_support_without_refuting(scorer):
    records = six(scorer)
    records[0]["frames"][K + 3]["vision_source"] = "live"
    p5 = verdict(scorer, records, "P5")
    assert p5["verdict"] == "partly supported" and not any(c["refutes"] for c in p5["clauses"])
    assert p5["clauses"][5]["observed"][f"{A} 530"]["mislabelled_held_frames"] == [K + 3]


def test_p5_population_ignores_the_4mm_bound_but_needs_k_plus_2_rendered_closing(scorer):
    def wide(batch):
        run = run_530(scorer, batch)
        for fact in run["frames"][K : K + 7]:
            fact["mouth_distance_m"] = 0.0045  # P4 reports this run, but P4's loss still occurs in it
        return run

    records = six(scorer, a530=wide(A), b530=wide(B), c530=wide(C))
    assert verdict(scorer, records, "P4")["verdict"] == "untested"
    p5 = verdict(scorer, records, "P5")
    assert p5["verdict"] == "supported" and len(p5["clauses"][3]["observed"]) == 3
    # The cut left closing at k+1 (never rendered closing at k+2): the loss at k+2 is not P4's loss.
    left = run_530(scorer, A, raw_run(cut_stop=(K + 1, "unstable_during_closure")))
    p5 = verdict(scorer, six(scorer, a530=left), "P5")
    assert p5["verdict"] == "supported"
    reported = p5["clauses"][7]["observed"][f"{A} 530"]
    assert reported["why"] == "the cut left closing at k+1, so k+2 was not rendered closing"


def test_p5_applies_only_where_p4s_loss_occurs(scorer):
    later = run_530(scorer, A, raw_run(loss=K + 3))  # held from k+3 and detached at k+6, but P4's loss did not occur
    p5 = verdict(scorer, six(scorer, a530=later), "P5")
    assert p5["verdict"] == "supported"
    assert f"{A} 530" in p5["clauses"][7]["observed"]
    p5 = verdict(scorer, all_530(scorer, loss=K + 3), "P5")
    assert p5["verdict"] == "untested" and p5["untested_reason"].startswith("no 530 run had P4's loss")
    p5 = verdict(scorer, all_530(scorer, loss=None, hold=False), "P5")  # no closure loss, so no hold
    assert p5["verdict"] == "untested"
    assert len(p5["clauses"][7]["observed"]) == 3


# ---------------------------------------------------------------------------------------------- P6


def test_p6_needs_one_unrefused_pass(scorer):
    def stopped(batch):
        return run_530(scorer, batch, outcome="stopped_gate_lost_during_closure", checks_passed=15)

    assert verdict(scorer, six(scorer, a530=stopped(A), b530=stopped(B)), "P6")["verdict"] == "supported"
    assert verdict(scorer, six(scorer, a530=stopped(A), b530=stopped(B), c530=stopped(C)), "P6")["verdict"] == "refuted"
    refused = run_530(scorer, C, refused=True, refused_reasons=["configuration_matches is false on a complete capture"])
    p6 = verdict(scorer, six(scorer, a530=stopped(A), b530=stopped(B), c530=refused), "P6")
    assert p6["verdict"] == "refuted"
    assert p6["clauses"][0]["observed"]["refused_passes_not_counted"] == [f"{C} 530"]


# ---------------------------------------------------------------------------------------------- P7


def test_p7_three_millimetres_before_the_deadline_supports_and_less_refutes(scorer):
    p7 = verdict(scorer, six(scorer), "P7")
    assert p7["verdict"] == "supported"
    assert p7["clauses"][0]["observed"][f"{A} 530"]["status"] == "moved"
    within = {K + offset: 0.0029 for offset in range(1, 6)} | {K + 6: 0.01}  # the deadline frame does not count
    assert verdict(scorer, all_530(scorer, baseline=baseline(within)), "P7")["verdict"] == "refuted"
    exactly = {K + 1: 0.001, K + 2: 0.002, K + 3: 0.003, K + 4: 0.0029, K + 5: 0.0029}  # 3 mm exactly, once
    p7 = verdict(scorer, all_530(scorer, baseline=baseline(exactly)), "P7")
    assert p7["verdict"] == "supported" and p7["clauses"][0]["observed"][f"{A} 530"]["status"] == "moved"


def test_p7_invalid_depth_is_a_gap_and_a_latched_loss_ends_the_window(scorer):
    gap = {K + 2: "invalid_depth"}
    moved_after_gap = baseline({K + 1: 0.001, K + 3: 0.004}, states=gap)
    assert verdict(scorer, all_530(scorer, baseline=moved_after_gap), "P7")["verdict"] == "supported"
    latched = baseline({K + 1: 0.001}, states={i: "tracking_lost" for i in range(K + 2, 200)})
    p7 = verdict(scorer, six(scorer, a530=run_530(scorer, A, baseline=latched)), "P7")
    assert p7["verdict"] == "partly supported" and not p7["clauses"][0]["refutes"]
    view = p7["clauses"][0]["observed"][f"{A} 530"]
    assert view["status"] == "indeterminate" and "latched tracking_lost at frame 76" in view["why"]
    p7 = verdict(scorer, all_530(scorer, baseline=latched), "P7")
    assert p7["verdict"] == "untested" and "indeterminate" in p7["untested_reason"]


def test_p7_within_3mm_refutes_only_when_every_window_frame_measured_the_target(scorer):
    within = {K + offset: 0.002 for offset in range(1, 7)}
    for states in (
        {K + 2: "invalid_depth"},
        {K + 5: "invalid_depth"},  # a trailing gap
        {K + offset: "invalid_calibration" for offset in range(2, 7)},  # measured at k+1 only
    ):
        unmeasured = baseline(within, states=states)
        p7 = verdict(scorer, six(scorer, a530=run_530(scorer, A, baseline=unmeasured)), "P7")
        assert p7["verdict"] == "partly supported" and not p7["clauses"][0]["refutes"]
        view = p7["clauses"][0]["observed"][f"{A} 530"]
        assert view["status"] == "indeterminate" and view["gap_frames"][0]["frame"] == min(states)
        p7 = verdict(scorer, all_530(scorer, baseline=unmeasured), "P7")
        assert p7["verdict"] == "untested" and "indeterminate" in p7["untested_reason"]
    targetless = baseline(within)
    targetless["rows"][K + 3]["target"] = None  # tracking, but no target: a gap
    p7 = verdict(scorer, all_530(scorer, baseline=targetless), "P7")
    assert p7["verdict"] == "untested"
    assert p7["clauses"][0]["observed"][f"{A} 530"]["gap_frames"] == [
        {"frame": K + 3, "state": "tracking", "reason": None}
    ]
    assert verdict(scorer, all_530(scorer, baseline=baseline(within)), "P7")["verdict"] == "refuted"


def test_p7_untested_when_frames_are_missing_or_no_run_holds(scorer):
    missing = baseline({K + 1: 0.001, K + 2: 0.002}, drop={K + 3})
    p7 = verdict(scorer, all_530(scorer, baseline=missing), "P7")
    assert p7["verdict"] == "untested"
    no_hold = all_530(scorer, hold=False, cut_stop=(K + 2, "vision_invalid"))  # the honest loss stops the cut
    p7 = verdict(scorer, no_hold, "P7")
    assert p7["verdict"] == "untested" and p7["untested_reason"] == "no 530 run held"
    assert len(p7["clauses"][1]["observed"]) == 3


# ---------------------------------------------------------------------------------------------- P8


def test_p8_a_tenth_of_a_millimetre_refutes_and_less_supports(scorer):
    records = six(scorer)
    records[0]["frames"][K + 3]["piece_pose"] = [0.00009, 0.0, 1.07, 1.0, 0.0, 0.0, 0.0]
    assert verdict(scorer, records, "P8")["verdict"] == "supported"
    records[0]["frames"][K + 3]["piece_pose"] = [0.0001, 0.0, 1.07, 1.0, 0.0, 0.0, 0.0]
    assert verdict(scorer, records, "P8")["verdict"] == "refuted"


def test_p8_the_mouth_to_spur_distance_alone_refutes(scorer):
    records = six(scorer)
    records[0]["frames"][K + 3]["tool_pose"] = [0.0002, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # the piece stays still
    p8 = verdict(scorer, records, "P8")
    assert p8["verdict"] == "refuted"
    assert p8["clauses"][0]["holds"] and not p8["clauses"][0]["refutes"] and p8["clauses"][1]["refutes"]
    records[0]["frames"][K + 3]["tool_pose"] = [0.00009, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert verdict(scorer, records, "P8")["verdict"] == "supported"


def test_p8_runs_that_never_detach_are_reported_and_leave_p8_untested(scorer):
    p8 = verdict(scorer, all_530(scorer, hold=False, cut_stop=(K + 2, "vision_invalid")), "P8")
    assert p8["verdict"] == "untested" and p8["untested_reason"] == "no 530 run detached"
    assert len(p8["clauses"][2]["observed"]) == 3


# ---------------------------------------------------------------------------------------------- tables


def test_incomplete_and_refused_runs_stay_in_every_table_and_withhold_support(scorer):
    records = six(scorer, a530=incomplete(scorer, A, 0, 530))
    result = {p["id"]: p for p in scorer.score(records)}
    assert {pid: result[pid]["verdict"] for pid in ("P1", "P4", "P5", "P6", "P7", "P8")} == {
        "P1": "partly supported",
        "P4": "partly supported",
        "P5": "partly supported",
        "P6": "supported",
        "P7": "partly supported",
        "P8": "partly supported",
    }
    for pid in ("P4", "P5", "P7", "P8"):
        assert any(f"{A} 530" in json.dumps(c["observed"]) and not c["holds"] for c in result[pid]["clauses"])
    assert f"{A} 530" in result["P6"]["clauses"][0]["observed"]["runs"]
    assert result["P1"]["clauses"][3]["observed"][f"{A} 530"]["judged"] is False
    refused = run_530(scorer, A, refused=True, refused_reasons=["the report does not show closure_hold enabled"])
    result = {p["id"]: p for p in scorer.score(six(scorer, a530=refused))}
    assert result["P4"]["verdict"] == "partly supported" and result["P1"]["verdict"] == "partly supported"
    rows = [scorer.run_summary(r) for r in six(scorer, a530=refused, b14944=incomplete(scorer, B, 1, 14944))]
    assert [(row["batch"], row["target"]) for row in rows] == [
        (A, 530),
        (A, 19444),
        (B, 530),
        (B, 14944),
        (C, 530),
        (C, 15004),
    ]
    assert rows[0]["refused"] and rows[3]["frames_recorded"] is None and rows[3]["outcome"] == "incomplete"


def test_a_graded_row_without_17_checks_stops_the_scorer(scorer):
    records = six(scorer)
    records[0]["checks_total"] = 16
    with pytest.raises(scorer.EvidenceError):
        scorer.score(records)


# ---------------------------------------------------------------------------------------------- run records


def _plan_row(index=0, tree=0, target=530):
    return {
        "index": index,
        "daylight": "source",
        "photometric_normalization": "raw",
        "target_tree_index": tree,
        "component_first_vertex": target,
        "strategy": {
            "name": "planned_pose_jaw_hold",
            "mode": "planned_pose_standoff",
            "standoff_m": 0.06,
            "max_step_m": 0.004,
            "max_rotation_deg": 1.5,
            "jaw_self_mask": True,
            "closure_hold": True,
        },
    }


def _stub_report(scorer, **fields):
    """report.json as the runner flushes it before the camera block (an abort such as a layout rejection)."""
    report = {
        "stage": "settle_physical_joint_drives",
        "node": "cn-gpu5",
        "task_outcome": "runtime_failure",
        "frame_count": 0,
        "photometric_normalization": "raw",
        "approach_strategy": {"mode": "planned_pose_standoff", "standoff_m": 0.06, "max_step_m": 0.004},
        "jaw_self_mask": {"enabled": True, "constants": scorer.registered_jaw_self_mask()},
        "closure_hold": {
            "enabled": True,
            "thresholds": scorer.registered_closure_hold(),
            "waived_checks": list(scorer.CLOSURE_HOLD_WAIVED_CHECKS),
        },
        "blender_scene": {
            "target": {"id": "tree0_SPUR_component_530", "radius_m": 0.005},
            "daylight": {"preset": "source"},
            "visual_proxy_roll_rad": 0.9,
        },
        "error": "RuntimeError('Orchard layout rejected: startup robot contact 6.1 N > 5 N')",
    }
    report.update(fields)
    return report


def test_run_record_keeps_a_run_that_aborted_before_the_camera_block(scorer, tmp_path):
    plan_row = _plan_row()
    row = {
        "run_directory": scorer.run_label(0, plan_row),
        "component_first_vertex": 530,
        "target_tree_index": 0,
        "outcome": "incomplete",
        "status": "incomplete",
        "checks_passed": 0,
        "checks_total": 0,
        "failed_checks": None,
    }
    run = tmp_path / "batch" / row["run_directory"]
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps(_stub_report(scorer)))
    (run / "frames.json").write_text(json.dumps({"fps": 10, "frames": []}))
    (run / "experiment_result.json").write_text(json.dumps({"capture_ok": False, "task_outcome": "runtime_failure"}))
    kept = scorer.run_record(tmp_path, "batch", plan_row, row)
    assert kept["frames"] is None and kept["p1"] is None
    assert "camera.wrist_intrinsics" in kept["not_recorded_reason"]
    assert kept["refused"] is False and kept["configuration_matches"] is None and kept["node"] == "cn-gpu5"
    (run / "frames.json").write_text('{"fps": 10, "frames": [')  # killed mid-flush
    truncated = scorer.run_record(tmp_path, "batch", plan_row, row)
    assert truncated["frames"] is None and "unreadable" in truncated["not_recorded_reason"]
    records = six(scorer, a530={**kept, "batch": A, "run_index": 0})
    assert verdict(scorer, records, "P1")["clauses"][3]["observed"][f"{A} 530"]["judged"] is False
    assert scorer.run_summary(records[0])["not_recorded_reason"] == kept["not_recorded_reason"]


def test_refusal_follows_configuration_matches_and_otherwise_the_report(scorer):
    plan_row = _plan_row()
    complete = _stub_report(scorer, frame_count=200)
    assert scorer.refusal(True, complete, plan_row) == (False, [], [])
    refused, reasons, _ = scorer.refusal(False, complete, plan_row)
    assert refused and reasons == ["configuration_matches is false on a complete capture"]
    partial = _stub_report(scorer, frame_count=120)
    assert scorer.refusal(False, partial, plan_row)[0] is False  # incomplete, not refused
    hold_off = {**partial, "closure_hold": {**partial["closure_hold"], "enabled": False}}
    refused, reasons, _ = scorer.refusal(None, hold_off, plan_row)
    assert refused and reasons == ["the report does not show closure_hold enabled"]
    refused, reasons, problems = scorer.refusal(True, hold_off, plan_row)
    assert not refused and not reasons and problems == ["the report does not show closure_hold enabled"]
    other_target = _stub_report(scorer, blender_scene={**partial["blender_scene"], "target": {"id": "tree1_x"}})
    assert scorer.refusal(None, other_target, plan_row)[0] is True


# ---------------------------------------------------------------------------------------------- the run set


def _write_evidence(scorer, root, mutate=None):
    for batch, runs in scorer.REGISTERED_RUNS.items():
        plan_rows = [_plan_row(index, tree, target) for index, (tree, target) in enumerate(runs)]
        plan = {
            "protocol": scorer.PROTOCOL,
            "frames": 200,
            "fps": 10,
            "code_revision": "frozen",
            "strategy": {"name": "planned_pose_jaw_hold"},
            "runs": plan_rows,
            "source_sha256": {},
        }
        rows = [
            {
                "run_directory": scorer.run_label(row["index"], row),
                **{key: row[key] for key in ("target_tree_index", "component_first_vertex", "daylight")},
                "photometric_normalization": "raw",
                "strategy": "planned_pose_jaw_hold",
                "outcome": "pass",
                "status": "graded",
                "checks_passed": 17,
                "checks_total": 17,
            }
            for row in plan_rows
        ]
        document = {
            "protocol": scorer.PROTOCOL,
            "batches": [
                {"condition": batch, "batch_dir": f"artifacts/vision_robustness/{batch}", "planned_runs_kept": 2}
            ],
            "runs": rows,
        }
        if mutate is not None:
            mutate(batch, plan, document)
        (root / "artifacts/vision_robustness" / batch).mkdir(parents=True, exist_ok=True)
        (root / "artifacts/vision_robustness" / batch / "plan.json").write_text(json.dumps(plan))
        (root / scorer.EVIDENCE_DIR).mkdir(parents=True, exist_ok=True)
        (root / scorer.EVIDENCE_DIR / f"{batch}.json").write_text(json.dumps(document))


def test_the_evidence_must_be_exactly_the_registered_six_runs(scorer, tmp_path):
    _write_evidence(scorer, tmp_path)
    validated = scorer.validated_evidence(tmp_path)
    assert [[plan_row["component_first_vertex"] for plan_row, _ in v[4]] for v in validated] == [
        [530, 19444],
        [530, 14944],
        [530, 15004],
    ]

    def tree_filtered(batch, plan, document):  # aggregate_eval --tree 1 drops C's 530 row
        if batch == C:
            document["runs"] = document["runs"][1:]
            document["batches"][0]["planned_runs_kept"] = 1

    def wrong_batch_dir(batch, plan, document):
        if batch == A:
            document["batches"][0]["batch_dir"] = f"artifacts/vision_robustness/{B}"

    def duplicate_row(batch, plan, document):
        if batch == A:
            document["runs"][0] = dict(document["runs"][1])

    def other_target(batch, plan, document):
        if batch == A:
            plan["runs"][1]["component_first_vertex"] = 14944

    def arm_off(batch, plan, document):
        if batch == B:
            plan["runs"][0]["strategy"]["closure_hold"] = False

    def sixteen_checks(batch, plan, document):
        if batch == C:
            document["runs"][1]["checks_total"] = 16

    def other_protocol(batch, plan, document):  # the plan is right; the grades were aggregated under another protocol
        if batch == B:
            document["protocol"] = "docs/EVAL_PROTOCOL_PERCEPTION_2026-09-26.md"

    mutations = (tree_filtered, wrong_batch_dir, duplicate_row, other_target, arm_off, sixteen_checks, other_protocol)
    for mutate in mutations:
        root = tmp_path / mutate.__name__
        _write_evidence(scorer, root, mutate)
        with pytest.raises(scorer.EvidenceError):
            scorer.validated_evidence(root)


def test_load_keeps_every_planned_run_even_without_recordings(scorer, tmp_path):
    _write_evidence(scorer, tmp_path)
    records = scorer.load(tmp_path)
    assert [(r["batch"], r["target"]) for r in records] == [
        (A, 530),
        (A, 19444),
        (B, 530),
        (B, 14944),
        (C, 530),
        (C, 15004),
    ]
    assert all(r["frames"] is None and r["not_recorded_reason"] == "no report.json and no frames.json" for r in records)
    result = {p["id"]: p for p in scorer.score(records)}
    assert result["P1"]["verdict"] == "untested" and result["P3"]["verdict"] == "untested"
    assert result["P4"]["verdict"] == "untested" and "refused or not recorded" in result["P4"]["untested_reason"]
    assert all(result[pid]["verdict"] != "supported" for pid in ("P1", "P2", "P3", "P4", "P5", "P7", "P8"))


def test_main_refuses_to_score_when_a_snapshot_is_not_the_plans_code(scorer, tmp_path):
    _write_evidence(scorer, tmp_path)  # valid evidence, but no <batch>/code snapshot holding the frozen sources
    with pytest.raises(SystemExit) as raised:
        scorer.main(["--root", str(tmp_path)])
    assert raised.value.code == 2


def test_main_refuses_to_score_without_the_registered_evidence(scorer, tmp_path):
    with pytest.raises(SystemExit) as raised:
        scorer.main(["--root", str(tmp_path)])
    assert raised.value.code == 2


def test_main_refuses_to_overwrite_an_existing_output(scorer, tmp_path):
    output = tmp_path / "verdicts.json"
    output.write_text("keep\n")
    with pytest.raises(SystemExit) as raised:
        scorer.main(["--root", str(tmp_path), "--output", str(output)])
    assert raised.value.code == 2 and output.read_text() == "keep\n"


# ---------------------------------------------------------------------------------------------- frozen code


def test_snapshot_drift_names_every_changed_or_missing_scored_file(scorer, tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    frozen = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in ("a.py", "b.py")}
    assert scorer.snapshot_drift(tmp_path, frozen, names=("a.py", "b.py")) == {}
    (tmp_path / "b.py").write_text("y = 3\n")
    drift = scorer.snapshot_drift(tmp_path, frozen, names=("a.py", "b.py", "c.py"))
    assert sorted(drift) == ["b.py", "c.py"] and drift["c.py"]["snapshot"] is None


@pytest.fixture
def restore_imports(monkeypatch):
    """bind_frozen_code replaces the isaaclab_pruning modules and sys.path; put them back for the other tests."""

    def ours(name):
        return name == "replay_jaw_self_mask" or name == "isaaclab_pruning" or name.startswith("isaaclab_pruning.")

    saved = {name: module for name, module in sys.modules.items() if ours(name)}
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    yield
    for name in [name for name in sys.modules if ours(name)]:
        del sys.modules[name]
    sys.modules.update(saved)


def test_bind_frozen_code_imports_from_the_tree_and_checks_every_loaded_file(scorer, restore_imports):
    files = [*(ROOT / "source/isaaclab_pruning").rglob("*.py"), TOOLS / "replay_jaw_self_mask.py"]
    frozen = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    before = {name: getattr(scorer, name) for names in scorer.FROZEN_NAMES.values() for name in names}
    provenance = scorer.bind_frozen_code(ROOT, frozen)
    jaw = "source/isaaclab_pruning/isaaclab_pruning/perception/jaw_self_mask.py"
    assert jaw in provenance["loaded_files_sha256_equal_plan"]
    assert provenance["modules"]["replay_jaw_self_mask"] == str(TOOLS / "replay_jaw_self_mask.py")
    for module_name, names in scorer.FROZEN_NAMES.items():
        module = sys.modules[module_name]
        assert module.__file__ == str(ROOT / "source/isaaclab_pruning" / (module_name.replace(".", "/") + ".py"))
        for name in names:
            assert getattr(scorer, name) is getattr(module, name)  # rebound to the freshly imported module
    assert scorer.jaw_boxes is not before["jaw_boxes"] and scorer.pose_change is not before["pose_change"]
    with pytest.raises(scorer.FrozenCodeError, match="jaw_self_mask.py"):
        scorer.bind_frozen_code(ROOT, {**frozen, jaw: "0" * 64})


def test_frozen_code_needs_the_three_plans_to_freeze_the_same_code(scorer, tmp_path, restore_imports):
    def snapshot(root, batch, text):
        hashes = {}
        for name in scorer.SCORED_SOURCES:
            path = root / "artifacts/vision_robustness" / batch / "code" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            hashes[name] = hashlib.sha256(text.encode()).hexdigest()
        return hashes

    for differs in ("source_sha256", "code_revision"):
        root = tmp_path / differs

        def mutate(batch, plan, document, root=root, differs=differs):
            other = differs == "source_sha256" and batch == B
            plan["source_sha256"] = snapshot(root, batch, "# other code\n" if other else "# frozen\n")
            if differs == "code_revision" and batch == C:
                plan["code_revision"] = "another revision"

        _write_evidence(scorer, root, mutate)
        validated = scorer.validated_evidence(root)
        with pytest.raises(scorer.FrozenCodeError, match="froze other sources"):
            scorer.frozen_code(root, validated)


# ---------------------------------------------------------------------------------------------- the document


def test_the_document_discloses_the_exposure_and_lists_every_planned_run(scorer):
    records = six(scorer, a19444=incomplete(scorer, A, 1, 19444))
    document = scorer.build_document(records, {}, {}, {}, "revision", False)
    assert document["construction_exposure"] == scorer.CONSTRUCTION_EXPOSURE
    text = " ".join(document["construction_exposure"])
    for fact in (
        "registered before",
        "81 frames",
        "eligible true, stationary true, explained false, held false, measurement_state tracking",
        "closure_progress_used 0.0, mask_pixel_count 5793",
        "'phase=retreat' at frames 101-121",
        "A run 1 FAILED exit 1 24m45s",
        "C run 1 COMPLETED 26m19s",
        "cn-gpu5",
        "exits 0 only when the capture completed, the unchanged grader passed all 17 checks",
        "each passed 17/17 with configuration_matches true",
        "said 'a Slurm state is not a grade', which understated this",
        "the outcome P6 counts",
        "bear on P4, P5, P6, P7 and P8 for that run",
        "not opened until it is committed",
        "No clause reads or uses any of these items",
        "decides P1 on the registered literal reading",
    ):
        assert fact in text
    assert "construction_exposure" in document["scope"]
    assert len(document["runs"]) == 6 and document["runs"][1]["outcome"] == "incomplete"
    seen = [row["seen_before_scoring"] for row in document["runs"]]
    assert seen[0].startswith("records read mid-recording")
    assert all(seen[i].endswith(f"({scorer.PASSED_EXIT})") for i in (0, 2, 3, 4, 5))
    assert seen[1].startswith("Slurm FAILED 24m45s (exit 1:")
    json.dumps(scorer.json_safe(document), allow_nan=False)
