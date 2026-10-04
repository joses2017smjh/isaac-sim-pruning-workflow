"""Protect the closure-hold generalization scorer (H1-H7 of docs/EVAL_PROTOCOL_HOLD_GENERALIZATION_2026-10-04.md).

Covered: each prediction at its registered support/refute boundary, untested cases, unscored runs deciding nothing,
the registered run set, the frozen scoring code and the output and root refusals. Every record is synthetic, built
with the jaw-in-view scorer tests' frame builders; no recording is read.
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
def hg():
    sys.path.insert(0, str(ROOT / "tools"))
    return _load("score_hold_generalization", ROOT / "tools/score_hold_generalization.py")


@pytest.fixture(scope="module")
def build():
    """The jaw-in-view scorer tests' synthetic builders (raw_run, record, p1_rows)."""
    return _load("jaw_in_view_tests", ROOT / "tests/test_score_jaw_in_view.py")


def _record(hg, build, batch, index, target, raw, **fields):
    record = build.record(hg.jiv, batch, index, target, raw, **fields)
    record["unscored_reason"] = hg.unscored_reason(record)
    return record


def eleven(hg, build, replace=None, behave=None):
    """The registered eleven runs, each behaving as predicted unless ``per_target[target]`` gives a raw-run factory
    (called with the repeat number) or ``replace[(batch, target)]`` gives a whole record."""
    replace, per_target = replace or {}, behave or {}
    records = []
    for batch, (targets, _) in hg.REGISTRY.items():
        repeat = int(batch.split("-r")[1][0]) if "-r" in batch else 1
        for index, (_, target) in enumerate(targets):
            if (batch, target) in replace:
                records.append(replace[(batch, target)])
                continue
            if target in per_target:
                raw, fields = per_target[target](repeat)
            elif target == hg.HOLDLESS_CONTROL:
                raw, fields = build.raw_run(loss=None, hold=False), {}
            else:
                raw, fields = build.raw_run(), {}
            records.append(_record(hg, build, batch, index, target, raw, **fields))
    return records


def verdicts(hg, records):
    return {p["id"]: p["verdict"] for p in hg.score(records)}


def verdict(hg, records, pid):
    return next(p for p in hg.score(records) if p["id"] == pid)


def test_the_predicted_pattern_supports_every_prediction(hg, build):
    assert verdicts(hg, eleven(hg, build)) == {f"H{i}": "supported" for i in range(1, 8)}


def test_the_registry_is_the_protocols_eleven_runs(hg):
    counts = {}
    for targets, strategy in hg.REGISTRY.values():
        assert strategy in hg.STRATEGY_FIELDS
        for _, target in targets:
            counts[target] = counts.get(target, 0) + 1
    assert counts == hg.EXPECTED_RUNS and sum(counts.values()) == 11
    assert hg.REGISTRY["hold-gen-18143-r1-20261004"][1] == "planned_pose_jaw_hold_gen_s100"
    assert hg.STRATEGY_FIELDS["planned_pose_jaw_hold_gen_s100"]["standoff_m"] == 0.10


# ------------------------------------------------------------------------------------------------ H1
def _never_closes(build):
    return build.raw_run(k=None, loss=None, hold=False, cut_stop=(60, "hazard_contact")), {
        "outcome": "stopped_hazard_contact",
        "checks_passed": 9,
    }


def test_h1_refuted_when_a_target_never_starts_closure_and_partly_with_one_start(hg, build):
    never = eleven(hg, build, behave={3721: lambda r: _never_closes(build)})
    h1 = verdict(hg, never, "H1")
    assert h1["verdict"] == "refuted" and h1["clauses"][0]["observed"]["refuting"] == [3721]
    assert len(h1["clauses"][0]["observed"]["targets"]["3721"]["stopped_before_closure"]) == 3
    one = eleven(hg, build, behave={3721: lambda r: (build.raw_run(), {}) if r == 1 else _never_closes(build)})
    assert verdict(hg, one, "H1")["verdict"] == "partly supported"
    two = eleven(hg, build, behave={3721: lambda r: (build.raw_run(), {}) if r != 3 else _never_closes(build)})
    assert verdict(hg, two, "H1")["verdict"] == "supported"


# ------------------------------------------------------------------------------------------------ H2, H3
def test_h2_refuted_by_a_first_loss_at_another_frame_or_an_unattributable_loss(hg, build):
    late = eleven(hg, build, behave={36196: lambda r: (build.raw_run(loss=K + 3), {})})
    assert verdict(hg, late, "H2")["verdict"] == "refuted"
    unexplained = eleven(hg, build, behave={18143: lambda r: (build.raw_run(attributable=False, hold=False), {})})
    assert verdict(hg, unexplained, "H2")["verdict"] == "refuted"


def test_h2_untested_when_no_selected_run_starts_closure(hg, build):
    never = eleven(hg, build, behave={t: (lambda r: _never_closes(build)) for t in hg.SELECTED})
    h2 = verdict(hg, never, "H2")
    assert h2["verdict"] == "untested" and "at most 4 mm" in h2["untested_reason"]


def test_h3_refuted_by_a_held_frame_that_stops_or_a_missing_detachment(hg, build):
    stops = eleven(
        hg, build, behave={3721: lambda r: (build.raw_run(cut_stop=(K + 4, "gate_lost_during_closure")), {})}
    )
    assert verdict(hg, stops, "H3")["verdict"] == "refuted"
    uncovered = eleven(
        hg, build, behave={3721: lambda r: (build.raw_run(hold_until=K + 5, cut_stop=(K + 6, "vision_invalid")), {})}
    )
    assert verdict(hg, uncovered, "H3")["verdict"] == "refuted"


def test_h3_tool_motion_withholds_support_without_refuting(hg, build):
    records = eleven(hg, build)
    target = next(r for r in records if r["target"] == 3721)
    next(f for f in target["frames"] if f["index"] == K + 3)["tool_pose"] = [0.0006, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    h3 = verdict(hg, records, "H3")
    assert h3["verdict"] == "partly supported" and not any(c["refutes"] for c in h3["clauses"])


# ------------------------------------------------------------------------------------------------ H4
def test_h4_refuted_when_a_target_that_reached_closure_never_passes(hg, build):
    failing = {"outcome": "stopped_gate_lost_during_closure", "checks_passed": 15}
    records = eleven(
        hg, build, behave={36196: lambda r: (build.raw_run(cut_stop=(K + 4, "gate_lost_during_closure")), failing)}
    )
    h4 = verdict(hg, records, "H4")
    assert h4["verdict"] == "refuted" and h4["clauses"][0]["observed"]["without_a_pass_after_closure"] == [36196]
    one_pass = eleven(
        hg,
        build,
        behave={36196: lambda r: (build.raw_run(), {}) if r == 2 else (build.raw_run(cut_stop=(K + 4, "x")), failing)},
    )
    assert verdict(hg, one_pass, "H4")["verdict"] == "supported"


def test_h4_a_target_that_never_reaches_closure_is_untested_not_refuted(hg, build):
    records = eleven(hg, build, behave={18143: lambda r: _never_closes(build)})
    h4 = verdict(hg, records, "H4")
    assert h4["verdict"] == "partly supported" and h4["clauses"][0]["observed"]["untested"] == [18143]
    everyone = eleven(hg, build, behave={t: (lambda r: _never_closes(build)) for t in hg.SELECTED})
    assert verdict(hg, everyone, "H4")["verdict"] == "untested"


# ------------------------------------------------------------------------------------------------ H5
def test_h5_a_tenth_of_a_millimetre_refutes(hg, build):
    records = eleven(hg, build)
    target = next(r for r in records if r["target"] == 18143)
    next(f for f in target["frames"] if f["index"] == K + 3)["piece_pose"] = [0.0001, 0.0, 1.07, 1.0, 0.0, 0.0, 0.0]
    assert verdict(hg, records, "H5")["verdict"] == "refuted"
    records = eleven(hg, build)
    target = next(r for r in records if r["target"] == 18143)
    next(f for f in target["frames"] if f["index"] == K + 3)["piece_pose"] = [0.00009, 0.0, 1.07, 1.0, 0.0, 0.0, 0.0]
    assert verdict(hg, records, "H5")["verdict"] == "supported"


# ------------------------------------------------------------------------------------------------ H6
def test_h6_refuted_by_a_count_mismatch_or_a_violation_outside_the_band_but_not_inside_it(hg, build):
    def with_rows(change):
        records = eleven(hg, build)
        rows = build.p1_rows()
        rows[40] = {**rows[40], **change}
        records[0]["p1"] = hg.jiv.p1_summary(rows, 200)
        return verdict(hg, records, "H6")

    assert with_rows({"recorded_count": 5001, "count_equal": False})["verdict"] == "refuted"
    assert with_rows({"violations": 1, "violations_outside_edge_band": 1})["verdict"] == "refuted"
    band = with_rows({"violations": 1, "violations_outside_edge_band": 0})
    assert band["verdict"] == "supported"
    assert band["clauses"][1]["observed"]["literal_reading_not_registered_here"]["violations"] == 1


# ------------------------------------------------------------------------------------------------ H7
def test_h7_refuted_when_14944_holds_or_530_fails_a_check(hg, build):
    held = build.raw_run(loss=None, hold=False)
    held[K + 3]["live_vision"]["closure_hold"]["held"] = True
    records = eleven(hg, build, behave={14944: lambda r: (held, {})})
    assert verdict(hg, records, "H7")["verdict"] == "refuted"
    failed = eleven(hg, build, behave={530: lambda r: (build.raw_run(), {"outcome": "stopped_x", "checks_passed": 16})})
    assert verdict(hg, failed, "H7")["verdict"] == "refuted"
    masked = build.raw_run(loss=None, hold=False)
    masked[10]["live_vision"]["measurement"]["jaw_mask_patch_unmasked"] = 168
    assert verdict(hg, eleven(hg, build, behave={14944: lambda r: (masked, {})}), "H7")["verdict"] == "refuted"


# ------------------------------------------------------------------------------------------------ unscored runs
def test_unscored_runs_decide_nothing(hg, build):
    unscored = {"configuration_matches": None}
    records = eleven(hg, build, behave={36196: lambda r: (build.raw_run(), unscored)})
    assert all(r["unscored_reason"] for r in records if r["target"] == 36196)
    result = {p["id"]: p for p in hg.score(records)}
    assert result["H4"]["verdict"] == "partly supported" and result["H4"]["clauses"][0]["observed"]["untested"] == [
        36196
    ]
    assert result["H1"]["verdict"] == "partly supported"
    assert not any(c["refutes"] for p in result.values() for c in p["clauses"])
    short = _record(hg, build, "hold-gen-a-r1-20261004", 0, 3721, build.raw_run(length=150))
    assert short["unscored_reason"] == "capture incomplete"
    problem = _record(hg, build, "hold-gen-a-r1-20261004", 0, 3721, build.raw_run(), configuration_problems=["x"])
    assert problem["unscored_reason"].startswith("the report departs")


def test_an_unscored_control_withholds_support_without_refuting(hg, build):
    records = eleven(
        hg, build, behave={530: lambda r: (build.raw_run(), {"status": "incomplete", "outcome": "incomplete"})}
    )
    h7 = verdict(hg, records, "H7")
    assert h7["verdict"] == "partly supported" and not h7["clauses"][0]["refutes"]


# ------------------------------------------------------------------------------------------------ inputs
def test_the_evidence_must_be_exactly_the_registered_batches(hg, tmp_path):
    with pytest.raises(hg.InputError, match="expected exactly"):
        hg.validated_evidence(tmp_path)
    (tmp_path / hg.EVIDENCE_DIR).mkdir(parents=True)
    for name in list(hg.REGISTRY)[:-1]:
        (tmp_path / hg.EVIDENCE_DIR / f"{name}.json").write_text("{}")
    with pytest.raises(hg.InputError, match="expected exactly"):
        hg.validated_evidence(tmp_path)


def real_plan(hg, name):
    """The plan the queue tool builds for a registered batch from its committed register."""
    launcher = hg.jiv.importlib.import_module("queue_vision_robustness")
    targets, strategy = hg.REGISTRY[name]
    relative = hg.register_of(name)
    loaded, provenance = launcher.load_targets(ROOT / relative)
    plan = launcher.experiment_plan(loaded, "source", "raw", strategy)
    plan["target_register"] = {**provenance, "targets_file": relative}
    return plan


def test_plans_built_by_the_queue_tool_from_the_committed_registers_pass(hg):
    for name in hg.REGISTRY:
        hg.check_plan(name, real_plan(hg, name))
    plan = real_plan(hg, "hold-gen-ctl-20261004")
    assert plan["runs"][1]["strategy"]["planned_tool_quat_wxyz"] is None  # 14944 keeps the home orientation


def test_a_plan_must_match_its_register_and_the_registered_strategy_exactly(hg):
    def refused(name, change):
        plan = real_plan(hg, name)
        change(plan)
        with pytest.raises(hg.InputError):
            hg.check_plan(name, plan)

    refused("hold-gen-18143-r1-20261004", lambda p: p["runs"][0]["strategy"].update(standoff_m=0.06))
    refused("hold-gen-ctl-20261004", lambda p: p["runs"][1]["strategy"].update(planned_tool_quat_wxyz=[1.0, 0, 0, 0]))
    refused("hold-gen-a-r1-20261004", lambda p: p["runs"][0]["strategy"].update(depth_appearance=True))
    refused("hold-gen-a-r1-20261004", lambda p: p["target_register"].update(targets_sha256="0" * 64))
    refused("hold-gen-a-r1-20261004", lambda p: p["runs"].reverse())
    refused("hold-gen-a-r2-20261004", lambda p: p.update(protocol="docs/EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md"))


def _write_evidence(hg, root, mutate=None):
    """A complete synthetic evidence tree: committed registers, a real plan per batch and a grade file each."""
    for relative in {hg.register_of(name) for name in hg.REGISTRY}:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes((ROOT / relative).read_bytes())
    for name in hg.REGISTRY:
        plan = real_plan(hg, name)
        rows = [
            {
                "run_directory": hg.jiv.run_label(row["index"], row),
                **{k: row[k] for k in ("target_tree_index", "component_first_vertex", "daylight")},
                "photometric_normalization": "raw",
                "strategy": row["strategy"]["name"],
                "outcome": "pass",
                "status": "graded",
                "checks_passed": 17,
                "checks_total": 17,
            }
            for row in plan["runs"]
        ]
        document = {
            "protocol": hg.PROTOCOL,
            "grader": hg.GRADER,
            "batches": [
                {"condition": name, "batch_dir": f"artifacts/vision_robustness/{name}", "planned_runs_kept": len(rows)}
            ],
            "runs": rows,
        }
        if mutate is not None:
            mutate(name, plan, document)
        (root / "artifacts/vision_robustness" / name).mkdir(parents=True, exist_ok=True)
        (root / "artifacts/vision_robustness" / name / "plan.json").write_text(json.dumps(plan))
        (root / hg.EVIDENCE_DIR).mkdir(parents=True, exist_ok=True)
        (root / hg.EVIDENCE_DIR / f"{name}.json").write_text(json.dumps(document))


def test_valid_evidence_passes_and_each_departure_refuses(hg, tmp_path):
    _write_evidence(hg, tmp_path / "ok")
    assert len(hg.validated_evidence(tmp_path / "ok")) == 7
    first = "hold-gen-a-r1-20261004"

    def only(change):
        return lambda name, plan, document: change(plan, document) if name == first else None

    mutations = {
        "protocol": only(lambda p, d: d.update(protocol="other")),
        "grader": only(lambda p, d: d.update(grader="other")),
        "batch_dir": only(lambda p, d: d["batches"][0].update(batch_dir="artifacts/vision_robustness/other")),
        "kept": only(lambda p, d: d["batches"][0].update(planned_runs_kept=1)),
        "missing_row": only(lambda p, d: d.update(runs=d["runs"][:1])),
        "label": only(lambda p, d: d["runs"][0].update(run_directory="run_00_wrong")),
        "pass_16": only(lambda p, d: d["runs"][0].update(checks_passed=16)),
        "total_16": only(lambda p, d: d["runs"][0].update(checks_total=16, checks_passed=16, outcome="stopped_x")),
        "strategy": only(lambda p, d: d["runs"][0].update(strategy="planned_pose_jaw_hold")),
        "target": only(lambda p, d: d["runs"][0].update(component_first_vertex=530)),
    }
    for label, mutate in mutations.items():
        root = tmp_path / label
        _write_evidence(hg, root, mutate)
        with pytest.raises(hg.InputError):
            hg.validated_evidence(root)
    extra = tmp_path / "extra"
    _write_evidence(hg, extra)
    (extra / hg.EVIDENCE_DIR / "hold-gen-a-r4-20261004.json").write_text("{}")
    with pytest.raises(hg.InputError, match="expected exactly"):
        hg.validated_evidence(extra)


def test_every_plan_must_have_frozen_the_scoring_code(hg, tmp_path):
    files = {name: hg.sha256(ROOT / name) for name in hg.SCORING_FILES}
    plan = {"source_sha256": dict(files)}
    validated = [(Path("x.json"), {}, "artifacts/vision_robustness/b", plan, [])]
    record = hg.frozen_scorer(validated, files[hg.SELF])
    assert not any(v["amended_since_freeze"] for v in record["batches"]["b"].values())
    stale = {"source_sha256": {**files, hg.SELF: "0" * 64}}
    record = hg.frozen_scorer([(Path("x.json"), {}, "artifacts/vision_robustness/b", stale, [])], files[hg.SELF])
    assert record["batches"]["b"][hg.SELF]["amended_since_freeze"] is True
    missing = {"source_sha256": {k: v for k, v in files.items() if k != "tools/score_jaw_in_view.py"}}
    with pytest.raises(hg.InputError, match="did not freeze"):
        hg.frozen_scorer([(Path("x.json"), {}, "artifacts/vision_robustness/b", missing, [])], files[hg.SELF])


def test_main_refuses_another_root_and_an_existing_output(hg, tmp_path, capsys):
    with pytest.raises(SystemExit):
        hg.main(["--root", str(tmp_path)])
    assert "--root must be" in capsys.readouterr().err
    output = tmp_path / "out.json"
    output.write_text("keep\n")
    with pytest.raises(SystemExit):
        hg.main(["--output", str(output)])
    assert "Refusing to overwrite" in capsys.readouterr().err and output.read_text() == "keep\n"


def test_the_document_lists_every_run_and_serializes(hg, build):
    records = eleven(hg, build)
    document = hg.build_document(records, {}, None, {}, "revision", False)
    assert len(document["runs"]) == 11 and document["label"] == hg.LABEL
    json.dumps(hg.jiv.json_safe(document), allow_nan=False)


def _unscored(build, raw):
    return raw, {"configuration_matches": None}


def test_h1_refutation_needs_two_scored_runs_without_a_start(hg, build):
    one_scored = eleven(
        hg, build, behave={3721: lambda r: _never_closes(build) if r == 1 else _unscored(build, build.raw_run())}
    )
    h1 = verdict(hg, one_scored, "H1")
    assert h1["verdict"] == "partly supported" and not h1["clauses"][0]["refutes"]
    two_scored = eleven(
        hg, build, behave={3721: lambda r: _never_closes(build) if r != 3 else _unscored(build, build.raw_run())}
    )
    assert verdict(hg, two_scored, "H1")["verdict"] == "refuted"


def test_h2_tracking_that_continues_at_k_plus_3_refutes(hg, build):
    records = eleven(hg, build, behave={3721: lambda r: (build.raw_run(loss=None, hold=False), {})})
    h2 = verdict(hg, records, "H2")
    assert h2["verdict"] == "refuted" and h2["clauses"][1]["refutes"]


def test_h2_shadow_clause_withholds_support_without_refuting(hg, build):
    records = eleven(hg, build)
    run = next(r for r in records if r["target"] == 3721)
    for fact in run["frames"]:
        fact["shadow"] = {"phase": "closing", "stopped_reason": None}
    h2 = verdict(hg, records, "H2")
    assert h2["verdict"] == "partly supported" and not h2["clauses"][3]["holds"]


def test_h3_a_held_frame_that_stops_refutes_its_own_clause(hg, build):
    records = eleven(
        hg, build, behave={3721: lambda r: (build.raw_run(cut_stop=(K + 4, "gate_lost_during_closure")), {})}
    )
    h3 = verdict(hg, records, "H3")
    assert h3["clauses"][4]["refutes"] and h3["clauses"][3]["refutes"]


def test_h3_coverage_and_mouth_withhold_support(hg, build):
    records = eleven(hg, build)
    run = next(r for r in records if r["target"] == 36196)
    next(f for f in run["frames"] if f["index"] == K + 4)["mouth_distance_m"] = 0.003 + 0.00006
    h3 = verdict(hg, records, "H3")
    assert h3["verdict"] == "partly supported" and not h3["clauses"][2]["holds"]
    records = eleven(hg, build, behave={36196: lambda r: (build.raw_run(hold_until=K + 5), {})})
    h3 = verdict(hg, records, "H3")
    assert not h3["clauses"][0]["holds"]


def test_h5_the_distance_clause_refutes_on_its_own(hg, build):
    records = eleven(hg, build)
    run = next(r for r in records if r["target"] == 18143)
    next(f for f in run["frames"] if f["index"] == K + 3)["tool_pose"] = [0.0002, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    h5 = verdict(hg, records, "H5")
    assert h5["verdict"] == "refuted" and h5["clauses"][0]["holds"] and h5["clauses"][1]["refutes"]


def test_h7_530s_held_stop_and_14944s_failed_check_each_refute(hg, build):
    stopped = {"outcome": "stopped_gate_lost_during_closure", "checks_passed": 15}
    held_stop = eleven(hg, build, behave={530: lambda r: (build.raw_run(cut_stop=(K + 4, "gate_lost")), stopped)})
    h7 = verdict(hg, held_stop, "H7")
    assert h7["verdict"] == "refuted" and any("held frames" in x for x in h7["clauses"][0]["observed"]["refuting"])
    failed = eleven(hg, build, behave={14944: lambda r: (build.raw_run(loss=None, hold=False), stopped)})
    assert verdict(hg, failed, "H7")["verdict"] == "refuted"
    short_hold = eleven(hg, build, behave={530: lambda r: (build.raw_run(hold_until=K + 5), {})})
    h7 = verdict(hg, short_hold, "H7")
    assert h7["verdict"] == "partly supported" and not h7["clauses"][0]["refutes"]


def test_unscored_runs_never_refute_h2_h3_h5_or_h6(hg, build):
    late = eleven(hg, build, behave={3721: lambda r: _unscored(build, build.raw_run(loss=K + 3))})
    assert verdict(hg, late, "H2")["verdict"] != "refuted"
    stops = eleven(hg, build, behave={3721: lambda r: _unscored(build, build.raw_run(cut_stop=(K + 4, "x")))})
    assert verdict(hg, stops, "H3")["verdict"] != "refuted"
    records = eleven(hg, build, behave={18143: lambda r: _unscored(build, build.raw_run())})
    for run in (r for r in records if r["target"] == 18143):
        next(f for f in run["frames"] if f["index"] == K + 3)["piece_pose"] = [0.01, 0.0, 1.07, 1.0, 0.0, 0.0, 0.0]
        rows = build.p1_rows()
        rows[3] = {**rows[3], "recorded_count": 1, "count_equal": False}
        run["p1"] = hg.jiv.p1_summary(rows, 200)
    assert verdict(hg, records, "H5")["verdict"] != "refuted"
    assert verdict(hg, records, "H6")["verdict"] != "refuted"


def test_h1_and_h4_ignore_an_unscored_run_of_a_target_that_already_holds(hg, build):
    records = eleven(
        hg, build, behave={3721: lambda r: _unscored(build, build.raw_run()) if r == 3 else (build.raw_run(), {})}
    )
    assert verdict(hg, records, "H1")["verdict"] == "supported"
    assert verdict(hg, records, "H4")["verdict"] == "supported"


def test_verify_self_needs_every_scoring_file_tracked_and_unchanged(hg, tmp_path, monkeypatch):
    import subprocess

    repo = tmp_path / "repo"
    for name in hg.SCORING_FILES:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes((ROOT / name).read_bytes())
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.setattr(hg, "IMPORTED_SCORING_MODULES", {})
    with pytest.raises(hg.InputError, match="not tracked"):
        hg.verify_self(repo)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    assert hg.verify_self(repo) == hg.sha256(repo / hg.SELF)
    (repo / "tools/score_jaw_in_view.py").write_text("# changed\n")
    with pytest.raises(hg.InputError, match="differs from HEAD"):
        hg.verify_self(repo)
    subprocess.run([*git, "checkout", "-q", "--", "tools/score_jaw_in_view.py"], check=True)
    monkeypatch.setattr(hg, "IMPORTED_SCORING_MODULES", {"score_jaw_in_view": "tools/score_jaw_in_view.py"})
    with pytest.raises(hg.InputError, match="was not imported from"):
        hg.verify_self(repo)  # the module in this process came from the real checkout, not from repo


def test_frozen_scorer_refuses_an_imported_module_that_changed_since_the_freeze(hg):
    files = {name: hg.sha256(ROOT / name) for name in hg.SCORING_FILES}
    stale = {"source_sha256": {**files, "tools/score_jaw_in_view.py": "0" * 64}}
    with pytest.raises(hg.InputError, match="differ from the copies frozen"):
        hg.frozen_scorer([(Path("x.json"), {}, "artifacts/vision_robustness/b", stale, [])], files[hg.SELF])


def test_h5_the_piece_clause_refutes_on_its_own(hg, build):
    records = eleven(hg, build)
    run = next(r for r in records if r["target"] == 3721)
    frame = next(f for f in run["frames"] if f["index"] == K + 3)
    frame["piece_pose"] = [0.0002, 0.0, 1.07, 1.0, 0.0, 0.0, 0.0]
    frame["tool_pose"] = [0.0002, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # the mouth moves with it: distance unchanged
    h5 = verdict(hg, records, "H5")
    assert h5["verdict"] == "refuted" and h5["clauses"][0]["refutes"] and h5["clauses"][1]["holds"]


def test_a_plan_row_must_be_source_light_with_both_arms_under_its_registered_name(hg):
    def refused(change):
        plan = real_plan(hg, "hold-gen-a-r1-20261004")
        change(plan["runs"][0])
        with pytest.raises(hg.InputError, match="plan row 0"):
            hg.check_plan("hold-gen-a-r1-20261004", plan)

    refused(lambda row: row.update(daylight="evening"))
    refused(lambda row: row["strategy"].update(jaw_self_mask=False))
    refused(lambda row: row["strategy"].update(closure_hold=False))
    refused(lambda row: row["strategy"].update(name="planned_pose_jaw_hold"))


def test_a_grade_file_pointing_at_another_batch_refuses(hg, tmp_path):
    def swap(name, plan, document):
        if name == "hold-gen-a-r1-20261004":  # r2 has the same targets and strategy, so only batch_dir differs
            document["batches"][0]["batch_dir"] = "artifacts/vision_robustness/hold-gen-a-r2-20261004"

    _write_evidence(hg, tmp_path, swap)
    with pytest.raises(hg.InputError, match="not exactly one batch"):
        hg.validated_evidence(tmp_path)


def test_h7_530_must_detach_at_k_plus_6(hg, build):
    records = eleven(hg, build)
    run = next(r for r in records if r["target"] == 530)
    for fact in run["frames"]:
        fact["detach_event"] = fact["index"] == K + 7
    h7 = verdict(hg, records, "H7")
    assert h7["verdict"] == "partly supported" and not h7["clauses"][0]["holds"]
