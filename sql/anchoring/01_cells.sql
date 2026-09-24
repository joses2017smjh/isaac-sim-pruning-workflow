-- Per model x family x condition (the lighting for the matrix and Stage A): error before and after each anchoring
-- variant. Every anchored value uses exactly ONE metric range per frame, the
-- range the rig itself reads at the tracked target. The affine ceiling uses the
-- whole ground-truth frame and is reported as a ceiling, never a result.
CREATE OR REPLACE VIEW anchoring_cells AS
SELECT
    model,
    family,
    condition,
    ANY_VALUE(lighting)                                       AS lighting,
    COUNT(DISTINCT tree_id)                                   AS n_trees,
    COUNT(*)                                                  AS n_frames,
    SUM(CASE WHEN anchored THEN 1 ELSE 0 END)                 AS n_anchored,
    ROUND(AVG(raw_mae_m), 4)                                  AS raw_mae_m,
    ROUND(AVG(CASE WHEN anchored THEN shift_mae_m END), 4)    AS shift_mae_m,
    ROUND(AVG(CASE WHEN anchored THEN scale_mae_m END), 4)    AS scale_mae_m,
    ROUND(AVG(affine_ceiling_mae_m), 4)                       AS affine_ceiling_mae_m,
    ROUND(AVG(n_zones), 1)                                    AS n_zones_mean,
    ROUND(AVG(zone_mae_m), 4)                                 AS zone_mae_m,
    ROUND(AVG(raw_target_abs_m), 4)                           AS raw_target_abs_m,
    ROUND(AVG(zone_target_abs_m), 4)                          AS zone_target_abs_m,
    ROUND(AVG(CASE WHEN anchored THEN shift_mae_m / NULLIF(raw_mae_m, 0) END), 3) AS shift_over_raw,
    ROUND(AVG(CASE WHEN anchored THEN scale_mae_m / NULLIF(raw_mae_m, 0) END), 3) AS scale_over_raw,
    ROUND(AVG(pred_gt_correlation), 3)                        AS pred_gt_correlation_mean
FROM anchor_input
GROUP BY model, family, condition
ORDER BY model, family,
    CASE condition WHEN 'source' THEN 0 WHEN 'morning' THEN 1 WHEN 'noon' THEN 2 WHEN 'evening' THEN 3 ELSE 9 END,
    condition;
