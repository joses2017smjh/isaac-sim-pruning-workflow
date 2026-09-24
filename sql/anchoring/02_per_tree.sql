-- Per tree, so a family-level mean cannot hide one tree that anchoring fails on.
CREATE OR REPLACE VIEW anchoring_per_tree AS
SELECT
    model, family, tree_id, condition,
    ANY_VALUE(lighting)                                       AS lighting,
    COUNT(*)                                                  AS n_frames,
    SUM(CASE WHEN anchored THEN 1 ELSE 0 END)                 AS n_anchored,
    ROUND(AVG(raw_mae_m), 4)                                  AS raw_mae_m,
    ROUND(AVG(CASE WHEN anchored THEN shift_mae_m END), 4)    AS shift_mae_m,
    ROUND(AVG(CASE WHEN anchored THEN scale_mae_m END), 4)    AS scale_mae_m,
    ROUND(AVG(affine_ceiling_mae_m), 4)                       AS affine_ceiling_mae_m,
    ROUND(AVG(n_zones), 1)                                    AS n_zones_mean,
    ROUND(AVG(zone_mae_m), 4)                                 AS zone_mae_m,
    ROUND(AVG(raw_target_abs_m), 4)                           AS raw_target_abs_m,
    ROUND(AVG(zone_target_abs_m), 4)                          AS zone_target_abs_m
FROM anchor_input
GROUP BY model, family, tree_id, condition
ORDER BY model, family, tree_id,
    CASE condition WHEN 'source' THEN 0 WHEN 'morning' THEN 1 WHEN 'noon' THEN 2 WHEN 'evening' THEN 3 ELSE 9 END,
    condition;
