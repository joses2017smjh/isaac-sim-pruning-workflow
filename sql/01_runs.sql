-- One row per attempted target. This is the table every reported number comes
-- from, so anything missing here is missing from the result.
--
-- A run is a PASS only when the independent grader passed every one of its
-- checks. Slurm state is deliberately absent: COMPLETED (0:0) is accounting,
-- never evidence of task success.
CREATE OR REPLACE VIEW runs AS
SELECT
    condition,
    run_directory,
    target_tree_index,
    component_first_vertex,
    daylight,
    photometric_normalization,
    strategy,
    stop_frame,
    final_phase_frame,
    failed_checks,
    status,
    checks_passed,
    checks_total,
    applied_commands,
    stop_reason,
    tracking_frames,
    recorded_frames,
    tracked_feature_count_min,
    tracked_confidence_min,
    centroid_error_m,
    -- A pass must be graded, complete and unanimous. Any other state is not a pass.
    (status = 'graded' AND checks_passed = checks_total AND checks_total > 0) AS passed,
    CASE
        WHEN status = 'graded' AND checks_passed = checks_total AND checks_total > 0 THEN 'pass'
        WHEN status = 'rejected_layout_startup_contact' THEN 'rejected_layout_startup_contact'
        WHEN status = 'rejected_visibility' THEN 'rejected_visibility'
        WHEN status = 'incomplete' THEN 'incomplete'
        WHEN status = 'infrastructure' THEN 'infrastructure'
        WHEN stop_reason IS NOT NULL THEN 'stopped_' || stop_reason
        ELSE 'failed_other'
    END AS outcome
FROM eval_input;
