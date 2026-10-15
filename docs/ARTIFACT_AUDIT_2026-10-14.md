# Recorded-artifact audit — October 14, 2026

This is a documentation audit of existing local evidence. No experiment, controller update, or new simulation was run.

## October 6 closure-hold sweep: aggregate verdict still open

The [registered sweep](EVAL_PROTOCOL_HOLD_SWEEP_2026-10-06.md) has 28 target records and two controls. Every stored sequence grade reports 200 frames and complete/nonblank capture. All 30 experiment records match their registered configuration and identify source commit `1d2676e881e3d31183ba16e3bd57bce4055b8766`.

Four target records pass the stored sequence grader, but eight target records also retain nonzero capture exit codes and `capture_ok=false`. Those flags must be reconciled before treating the sweep as a scored population. The 530 control stops with `gate_lost_during_closure`; the 14944 control passes. Earlier targeted passes therefore do not establish consistent control behavior in this sweep.

The [per-run inventory](evidence/hold_sweep_inventory_2026-10-14.json) retains every flag, failed check, source revision, and hashes of the experiment result, sequence grade, and renderer report. **No 4/28 success-rate claim is made.** The full preregistered S1–S7 verdict was not present in the audited published record. The inventory does not replace it, and no result is pooled with the earlier 0/9, 2/7, or 1/30 studies.

## October 6 ROS 2 DDS graph: a separate source snapshot

A recorded audit of an actual ROS 2 DDS graph received 200 decisions and compared 199 states: **199/199 agreed**, with maximum command difference **1.9954 mm** within a 2 mm tolerance. All eight injected observation faults entered hold on their ten affected frames, with no authorized motion state or nonzero command delta.

[Receipt and frozen source hashes](evidence/ros2_graph_snapshot_2026-10-06.json). The receipt includes per-frame decisions and fault outcomes. Publication only normalizes the capture-directory label and adds the original receipt hash and comparison against public files. **The evaluated local source differs from this public checkout; these are recorded audit results, not a claim that the public quickstart reproduces them.**

Inputs were recorded RGB, optical-Z depth, intrinsics, and tool poses. Target identity, camera world transforms, and contact context remained recorded metadata. ToF was telemetry only in this ROS node; Isaac release and ToF gates were outside this test. The audit establishes neither hardware actuation nor physical safety. The existing [software replay guide](ROS2_SIL.md) remains the public implementation entry point.

**Update, October 15:** the node source evaluated by the original October 6 receipt is now public. [That receipt](evidence/ros2_graph_parity_2026-10-06.json) is published with only its capture-directory label normalized. Its recorded SHA-256 values match the public `replay.py`, `servo_node.py`, `test_stream_graph.py` and `tools/validate_ros_stream.py`. Its measured results are identical to the snapshot receipt above, whose recorded hashes do not match these files. The scope limits above still apply.
