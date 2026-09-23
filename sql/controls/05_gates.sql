-- The pre-registered September 20 sanity gates, unchanged, per model and
-- condition on the mask-gated target: p95 error <= 20 mm, mean relative
-- error <= 10%, coverage >= 99%, p95 inference <= 100 ms. A gate over zero
-- frames is not a pass.
CREATE OR REPLACE VIEW gates AS
SELECT
    model,
    condition,
    SUM(CASE WHEN target_masked_valid THEN 1 ELSE 0 END) AS frames_with_target,
    SUM(CASE WHEN target_masked_valid AND target_masked_p95_abs_m <= 0.020 THEN 1 ELSE 0 END) AS frames_p95_le_20mm,
    SUM(CASE WHEN target_masked_valid AND target_masked_abs_relative <= 0.10 THEN 1 ELSE 0 END) AS frames_rel_le_10pct,
    SUM(CASE WHEN target_masked_valid AND target_masked_coverage >= 0.99 THEN 1 ELSE 0 END) AS frames_coverage_ok,
    ROUND(QUANTILE_CONT(inference_seconds, 0.95), 4) AS inference_p95_s,
    (SUM(CASE WHEN target_masked_valid THEN 1 ELSE 0 END) > 0
       AND SUM(CASE WHEN target_masked_valid THEN 1 ELSE 0 END)
           = SUM(CASE WHEN target_masked_valid AND target_masked_p95_abs_m <= 0.020
                          AND target_masked_abs_relative <= 0.10 AND target_masked_coverage >= 0.99 THEN 1 ELSE 0 END)
       AND QUANTILE_CONT(inference_seconds, 0.95) <= 0.10) AS all_gates_pass
FROM frames
GROUP BY model, condition
ORDER BY model, condition;
