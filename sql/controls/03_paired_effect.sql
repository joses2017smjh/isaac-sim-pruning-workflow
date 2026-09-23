-- Each control against its registered baseline, paired within a tree: the
-- same geometry and seed, one axis changed. The baseline table comes from
-- generalization_controls.BASELINE_OF and is inserted by the aggregator, so
-- the pairing cannot be chosen after seeing the numbers. Per-tree ratios are
-- averaged so one tree with many views cannot dominate.
CREATE OR REPLACE VIEW per_tree_condition AS
SELECT model, family, tree_id, condition,
       AVG(mask_mae_m)                                                        AS mask_mae_m,
       AVG(mask_signed_median_m)                                              AS signed_m,
       AVG(CASE WHEN target_masked_valid THEN target_masked_mae_m END)        AS target_mae_m,
       SUM(CASE WHEN target_masked_valid THEN 1 ELSE 0 END)                   AS target_frames
FROM frames
WHERE mask_mae_m IS NOT NULL
GROUP BY model, family, tree_id, condition;

CREATE OR REPLACE VIEW paired_effect AS
WITH pairs AS (
    SELECT c.model, c.family, c.tree_id, c.condition, b.baseline,
           c.mask_mae_m, base.mask_mae_m AS baseline_mask_mae_m,
           c.signed_m, base.signed_m AS baseline_signed_m,
           c.target_mae_m, base.target_mae_m AS baseline_target_mae_m
    FROM per_tree_condition c
    JOIN baseline_of b ON b.condition = c.condition
    JOIN per_tree_condition base
      ON base.model = c.model AND base.tree_id = c.tree_id AND base.condition = b.baseline
)
SELECT
    model, family, condition, baseline,
    COUNT(DISTINCT tree_id)                                             AS n_trees,
    ROUND(AVG(mask_mae_m), 4)                                           AS mask_mae_mean_m,
    ROUND(AVG(baseline_mask_mae_m), 4)                                  AS baseline_mask_mae_mean_m,
    ROUND(AVG(mask_mae_m / NULLIF(baseline_mask_mae_m, 0)), 3)          AS mean_ratio_to_baseline,
    ROUND(MIN(mask_mae_m / NULLIF(baseline_mask_mae_m, 0)), 3)          AS min_ratio_to_baseline,
    ROUND(MAX(mask_mae_m / NULLIF(baseline_mask_mae_m, 0)), 3)          AS max_ratio_to_baseline,
    SUM(CASE WHEN mask_mae_m < baseline_mask_mae_m THEN 1 ELSE 0 END)   AS trees_improved,
    ROUND(AVG(signed_m), 4)                                             AS signed_median_mean_m,
    ROUND(AVG(baseline_signed_m), 4)                                    AS baseline_signed_median_mean_m,
    ROUND(AVG(target_mae_m), 4)                                         AS target_mae_mean_m,
    ROUND(AVG(baseline_target_mae_m), 4)                                AS baseline_target_mae_mean_m
FROM pairs
GROUP BY model, family, condition, baseline
ORDER BY model, family, condition;
