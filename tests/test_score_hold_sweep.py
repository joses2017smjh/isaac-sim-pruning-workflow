"""Protect the closure-hold sweep scorer (S1-S7 of docs/EVAL_PROTOCOL_HOLD_SWEEP_2026-10-06.md).

Covered: each prediction at its registered support/refute boundary, untested cases, unscored runs deciding nothing,
the registered run set and plans, the frozen scoring code and the output and root refusals. Every record is
synthetic, built with the jaw-in-view scorer tests' frame builders; no recording is read.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]
K = 74


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sw():
    sys.path.insert(0, str(ROOT / "tools"))
    return _load("score_hold_sweep", ROOT / "tools/score_hold_sweep.py")


@pytest.fixture(scope="module")
def build():
    return _load("jaw_in_view_tests_for_sweep", ROOT / "tests/test_score_jaw_in_view.py")


def _record(sw, build, batch, index, target, raw, **fields):
    record = build.record(sw.jiv, batch, index, target, raw, **fields)
    record["unscored_reason"] = sw.unscored_reason(record)
    return record


def _rejected(build):
    raw = build.raw_run(k=None, loss=0, hold=False, cut_stop=(0, "initial_target_not_visible"))
    return raw, {"outcome": "stopped_initial_target_not_visible", "checks_passed": 8}


def thirty(sw, build, behave=None):
    """The registered thirty runs behaving as predicted unless ``behave[target]`` gives (raw, fields)."""
    behave = behave or {}
    records = []
    for batch, (targets, _) in sw.REGISTRY.items():
        for index, (_, target) in enumerate(targets):
            if target in behave:
                raw, fields = behave[target]()
            elif target in sw.PREDICTED_REJECTED:
                raw, fields = _rejected(build)
            elif target == sw.HOLDLESS_CONTROL:
                raw, fields = build.raw_run(loss=None, hold=False), {}
            else:
                raw, fields = build.raw_run(), {}
            records.append(_record(sw, build, batch, index, target, raw, **fields))
    return records


def verdict(sw, records, pid):
    return next(p for p in sw.score(records) if p["id"] == pid)


def test_the_predicted_pattern_supports_every_prediction(sw, build):
    assert {p["id"]: p["verdict"] for p in sw.score(thirty(sw, build))} == {f"S{i}": "supported" for i in range(1, 8)}


def test_the_registry_is_the_protocols_thirty_runs(sw):
    register = json.loads((ROOT / sw.REGISTER_SWEEP).read_text())
    assert [(t["target_tree_index"], t["component_first_vertex"]) for t in register["targets"]] == list(sw.SWEEP)
    assert len(sw.SWEEP) == 28 and sum(sw.EXPECTED_RUNS.values()) == 30
    assert set(sw.PREDICTED_REJECTED) <= set(sw.SWEEP_TARGETS)
    evidence = json.loads((ROOT / "docs/evidence/home_visibility_check_2026-10-05.json").read_text())
    rejected = {int(k.split("_")[1]) for k, v in evidence["candidates"].items() if not v["visible"]}
    assert rejected & set(sw.SWEEP_TARGETS) == set(sw.PREDICTED_REJECTED)


# ------------------------------------------------------------------------------------------------ S1
def test_s1_refuted_when_the_visibility_check_is_wrong_either_way(sw, build):
    visible = sw.SWEEP_TARGETS[1]
    assert visible not in sw.PREDICTED_REJECTED
    wrong = thirty(sw, build, behave={visible: lambda: _rejected(build)})
    s1 = verdict(sw, wrong, "S1")
    assert s1["verdict"] == "refuted" and s1["clauses"][0]["observed"]["departures"]
    initialized = thirty(sw, build, behave={sw.PREDICTED_REJECTED[0]: lambda: (build.raw_run(), {})})
    assert verdict(sw, initialized, "S1")["verdict"] == "refuted"


def test_s1_only_a_frame_zero_visibility_stop_counts_as_rejected(sw, build):
    other = build.raw_run(k=None, loss=0, hold=False, cut_stop=(0, "vision_invalid"))
    records = thirty(
        sw, build, behave={sw.PREDICTED_REJECTED[0]: lambda: (other, {"outcome": "x", "checks_passed": 9})}
    )
    assert verdict(sw, records, "S1")["verdict"] == "refuted"


# ------------------------------------------------------------------------------------------------ S2-S3, S5-S7
def test_s2_refuted_by_a_first_loss_at_another_frame(sw, build):
    target = sw.SWEEP_TARGETS[1]
    assert (
        verdict(sw, thirty(sw, build, behave={target: lambda: (build.raw_run(loss=K + 3), {})}), "S2")["verdict"]
        == "refuted"
    )


def test_s3_refuted_by_a_held_frame_that_stops(sw, build):
    stopped = {"outcome": "stopped_gate_lost_during_closure", "checks_passed": 15}
    behave = {sw.SWEEP_TARGETS[2]: lambda: (build.raw_run(cut_stop=(K + 4, "gate_lost_during_closure")), stopped)}
    s3 = verdict(sw, thirty(sw, build, behave=behave), "S3")
    assert s3["verdict"] == "refuted" and s3["clauses"][4]["refutes"]


def test_s5_refuted_by_piece_motion(sw, build):
    records = thirty(sw, build)
    run = next(r for r in records if r["target"] == sw.SWEEP_TARGETS[3])
    next(f for f in run["frames"] if f["index"] == K + 3)["piece_pose"] = [0.0002, 0.0, 1.07, 1.0, 0.0, 0.0, 0.0]
    assert verdict(sw, records, "S5")["verdict"] == "refuted"


def test_s6_refuted_by_a_count_mismatch_and_not_by_a_band_pixel(sw, build):
    def with_rows(change):
        records = thirty(sw, build)
        rows = build.p1_rows()
        rows[40] = {**rows[40], **change}
        records[1]["p1"] = sw.jiv.p1_summary(rows, 200)
        return verdict(sw, records, "S6")["verdict"]

    assert with_rows({"recorded_count": 5001, "count_equal": False}) == "refuted"
    assert with_rows({"violations": 1, "violations_outside_edge_band": 0}) == "supported"


def test_s7_refuted_when_14944_holds(sw, build):
    held = build.raw_run(loss=None, hold=False)
    held[K + 3]["live_vision"]["closure_hold"]["held"] = True
    assert (
        verdict(sw, thirty(sw, build, behave={sw.HOLDLESS_CONTROL: lambda: (held, {})}), "S7")["verdict"] == "refuted"
    )


# ------------------------------------------------------------------------------------------------ S4
def _failing(build):
    return build.raw_run(), {"outcome": "stopped_piece_not_dropped", "checks_passed": 16}


def test_s4_needs_at_least_half_of_the_engaged_runs_to_pass(sw, build):
    engaged = [t for t in sw.SWEEP_TARGETS if t not in sw.PREDICTED_REJECTED]
    assert len(engaged) == 25
    half_fail = thirty(sw, build, behave={t: (lambda: _failing(build)) for t in engaged[:12]})  # 13 of 25 pass
    assert verdict(sw, half_fail, "S4")["verdict"] == "supported"
    most_fail = thirty(sw, build, behave={t: (lambda: _failing(build)) for t in engaged[:13]})  # 12 of 25 pass
    s4 = verdict(sw, most_fail, "S4")
    assert s4["verdict"] == "refuted" and s4["clauses"][0]["observed"]["passes"] == 12


def test_s4_untested_with_fewer_than_three_engaged_runs(sw, build):
    engaged = [t for t in sw.SWEEP_TARGETS if t not in sw.PREDICTED_REJECTED]
    never = build.raw_run(k=None, loss=None, hold=False, cut_stop=(60, "hazard_contact"))
    behave = {t: (lambda: (never, {"outcome": "stopped_hazard_contact", "checks_passed": 9})) for t in engaged[2:]}
    s4 = verdict(sw, thirty(sw, build, behave=behave), "S4")
    assert s4["verdict"] == "untested" and len(s4["clauses"][1]["observed"]) == 23 + 3


def test_unscored_runs_never_refute_and_withhold_support(sw, build):
    target = sw.SWEEP_TARGETS[1]
    records = thirty(sw, build, behave={target: lambda: (build.raw_run(loss=K + 3), {"configuration_matches": None})})
    result = {p["id"]: p for p in sw.score(records)}
    assert not any(c["refutes"] for p in result.values() for c in p["clauses"])
    assert result["S2"]["verdict"] == "partly supported" and result["S1"]["verdict"] == "partly supported"


# ------------------------------------------------------------------------------------------------ inputs
def real_plan(sw, name):
    launcher = importlib.import_module("queue_vision_robustness")
    _, strategy = sw.REGISTRY[name]
    relative = sw.register_of(name)
    loaded, provenance = launcher.load_targets(ROOT / relative)
    plan = launcher.experiment_plan(loaded, "source", "raw", strategy)
    plan["target_register"] = {**provenance, "targets_file": relative}
    return plan


def test_plans_built_by_the_queue_tool_pass_and_departures_refuse(sw):
    for name in sw.REGISTRY:
        sw.check_plan(name, real_plan(sw, name))
    plan = real_plan(sw, "hold-sweep-20261006")
    assert plan["protocol"] == sw.PROTOCOL and len(plan["runs"]) == 28
    for change in (
        lambda p: p["runs"][0]["strategy"].update(standoff_m=0.10),
        lambda p: p["runs"][3]["strategy"].update(planned_tool_quat_wxyz=[1.0, 0.0, 0.0, 0.0]),
        lambda p: p["runs"].pop(),
        lambda p: p["target_register"].update(targets_sha256="0" * 64),
        lambda p: p["runs"][0].update(daylight="evening"),
    ):
        bad = real_plan(sw, "hold-sweep-20261006")
        change(bad)
        with pytest.raises(sw.InputError):
            sw.check_plan("hold-sweep-20261006", bad)


def test_the_evidence_must_be_exactly_the_two_registered_batches(sw, tmp_path):
    with pytest.raises(sw.InputError, match="expected exactly"):
        sw.validated_evidence(tmp_path)


def test_every_plan_must_have_frozen_the_scoring_code(sw):
    files = {name: sw.sha256(ROOT / name) for name in sw.SCORING_FILES}
    good = {"source_sha256": dict(files)}
    record = sw.frozen_scorer([(Path("x.json"), {}, "artifacts/vision_robustness/b", good, [])], files[sw.SELF])
    assert not any(v["amended_since_freeze"] for v in record["batches"]["b"].values())
    stale = {"source_sha256": {**files, "tools/score_jaw_in_view.py": "0" * 64}}
    with pytest.raises(sw.InputError, match="differ from the copies frozen"):
        sw.frozen_scorer([(Path("x.json"), {}, "artifacts/vision_robustness/b", stale, [])], files[sw.SELF])
    missing = {"source_sha256": {k: v for k, v in files.items() if k != sw.SELF}}
    with pytest.raises(sw.InputError, match="did not freeze"):
        sw.frozen_scorer([(Path("x.json"), {}, "artifacts/vision_robustness/b", missing, [])], files[sw.SELF])


def test_main_refuses_another_root_and_an_existing_output(sw, tmp_path, capsys):
    with pytest.raises(SystemExit):
        sw.main(["--root", str(tmp_path)])
    assert "--root must be" in capsys.readouterr().err
    output = tmp_path / "out.json"
    output.write_text("keep\n")
    with pytest.raises(SystemExit):
        sw.main(["--output", str(output)])
    assert "Refusing to overwrite" in capsys.readouterr().err and output.read_text() == "keep\n"


def test_the_document_lists_every_run_and_serializes(sw, build):
    document = sw.build_document(thirty(sw, build), {}, None, {}, "revision", False)
    assert len(document["runs"]) == 30 and [p["id"] for p in document["predictions"]] == [f"S{i}" for i in range(1, 8)]
    json.dumps(sw.jiv.json_safe(document), allow_nan=False)


def test_s1_a_visibility_stop_after_frame_zero_is_not_a_rejection_at_home(sw, build):
    late = build.raw_run(k=None, loss=5, hold=False, cut_stop=(5, "initial_target_not_visible"))
    records = thirty(sw, build, behave={sw.PREDICTED_REJECTED[0]: lambda: (late, {"outcome": "x", "checks_passed": 8})})
    assert verdict(sw, records, "S1")["verdict"] == "refuted"


def test_s4_exactly_half_holds(sw, build):
    engaged = [t for t in sw.SWEEP_TARGETS if t not in sw.PREDICTED_REJECTED]
    never = build.raw_run(k=None, loss=None, hold=False, cut_stop=(60, "hazard_contact"))
    behave = {engaged[0]: lambda: (never, {"outcome": "stopped_hazard_contact", "checks_passed": 9})}  # 24 engaged
    behave.update({t: (lambda: _failing(build)) for t in engaged[1:13]})  # 12 of 24 pass
    s4 = verdict(sw, thirty(sw, build, behave=behave), "S4")
    assert s4["verdict"] == "supported" and s4["clauses"][0]["observed"] == {
        **s4["clauses"][0]["observed"],
        "passes": 12,
        "of": 24,
    }


def test_s4_counts_only_scored_runs(sw, build):
    engaged = [t for t in sw.SWEEP_TARGETS if t not in sw.PREDICTED_REJECTED]

    def unscored_fail():
        return build.raw_run(), {"outcome": "stopped_x", "checks_passed": 16, "configuration_matches": None}

    behave = {t: (lambda: _failing(build)) for t in engaged[:11]}
    behave.update({t: unscored_fail for t in engaged[11:13]})  # 23 scored engaged, 12 pass
    s4 = verdict(sw, thirty(sw, build, behave=behave), "S4")
    assert s4["clauses"][0]["observed"]["of"] == 23 and not s4["clauses"][0]["refutes"]
