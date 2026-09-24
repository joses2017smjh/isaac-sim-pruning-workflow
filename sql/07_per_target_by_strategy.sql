-- The same target under each labelled strategy, side by side: the outcome,
-- the frame it stopped at, whether the standoff phase completed, and which
-- grader checks failed. A strategy is judged per target against the
-- baseline row of that target, never by a pooled rate alone.
CREATE OR REPLACE VIEW per_target_by_strategy AS
SELECT
    target_tree_index,
    component_first_vertex,
    strategy,
    condition,
    outcome,
    stop_frame,
    final_phase_frame,
    checks_passed,
    checks_total,
    failed_checks,
    applied_commands
FROM runs
ORDER BY target_tree_index, component_first_vertex,
    CASE strategy WHEN 'baseline' THEN 0 WHEN 'tool_axis_standoff' THEN 1
                  WHEN 'horizontal_standoff' THEN 2 WHEN 'fine_step' THEN 3 ELSE 9 END;
