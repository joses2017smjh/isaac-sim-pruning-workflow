-- The cell: model x family x condition. n_trees supports a population
-- statement; n_frames is what the means average over. Failed and missing
-- frames are not here at all; plan.json lists them and the aggregator reports
-- registered against scored counts beside these numbers.
CREATE OR REPLACE VIEW cells AS
SELECT
    model,
    family,
    condition,
    ANY_VALUE(condition_group)                 AS condition_group,
    COUNT(DISTINCT tree_id)                    AS n_trees,
    COUNT(*)                                   AS n_frames,
    ROUND(AVG(mask_mae_m), 4)                  AS mask_mae_mean_m,
    ROUND(MEDIAN(mask_mae_m), 4)               AS mask_mae_median_m,
    ROUND(AVG(mask_abs_relative), 4)           AS mask_abs_relative_mean,
    ROUND(AVG(mask_signed_median_m), 4)        AS mask_signed_median_mean_m,
    ROUND(AVG(mask_mae_debiased_m), 4)         AS mask_mae_debiased_mean_m,
    ROUND(AVG(full_mae_m), 4)                  AS full_mae_mean_m,
    SUM(CASE WHEN target_masked_valid THEN 1 ELSE 0 END)                          AS target_masked_valid_frames,
    ROUND(AVG(CASE WHEN target_masked_valid THEN target_masked_mae_m END), 4)     AS target_masked_mae_mean_m,
    ROUND(AVG(CASE WHEN target_masked_valid THEN target_masked_signed_m END), 4)  AS target_masked_signed_mean_m,
    ROUND(AVG(CASE WHEN target_masked_valid THEN target_masked_p95_abs_m END), 4) AS target_masked_p95_mean_m,
    ROUND(QUANTILE_CONT(inference_seconds, 0.95), 4)                              AS inference_p95_s
FROM frames
GROUP BY model, family, condition
ORDER BY model, family, condition;
