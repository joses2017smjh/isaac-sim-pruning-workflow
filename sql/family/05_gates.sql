-- The pre-registered sanity gates from September 20, unchanged: target p95
-- error <= 20 mm, mean relative error <= 10%, target coverage >= 99%, p95
-- inference <= 100 ms. Evaluated per model over frames with a valid target,
-- on the mask-gated target (primary for the matrix) and on the unmasked one
-- (kept for continuity with the September 20 runs). A gate over zero frames
-- is not a pass.
CREATE OR REPLACE VIEW gates AS
SELECT
    model,
    'masked' AS target_kind,
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
FROM frames GROUP BY model
UNION ALL
SELECT
    model,
    'unmasked' AS target_kind,
    SUM(CASE WHEN target_valid THEN 1 ELSE 0 END),
    SUM(CASE WHEN target_valid AND target_p95_abs_m <= 0.020 THEN 1 ELSE 0 END),
    SUM(CASE WHEN target_valid AND target_abs_relative <= 0.10 THEN 1 ELSE 0 END),
    SUM(CASE WHEN target_valid AND target_coverage >= 0.99 THEN 1 ELSE 0 END),
    ROUND(QUANTILE_CONT(inference_seconds, 0.95), 4),
    (SUM(CASE WHEN target_valid THEN 1 ELSE 0 END) > 0
       AND SUM(CASE WHEN target_valid THEN 1 ELSE 0 END)
           = SUM(CASE WHEN target_valid AND target_p95_abs_m <= 0.020
                          AND target_abs_relative <= 0.10 AND target_coverage >= 0.99 THEN 1 ELSE 0 END)
       AND QUANTILE_CONT(inference_seconds, 0.95) <= 0.10)
FROM frames GROUP BY model
ORDER BY model, target_kind;
