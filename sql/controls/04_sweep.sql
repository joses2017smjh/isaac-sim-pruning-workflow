-- The distance sweep: one centred pose per nominal distance on the training
-- camera. The predicted and reference target depths are reported side by
-- side so the curve, not a ratio alone, is the evidence. Frames whose target
-- projection is occluded stay in n_frames and drop out of the target columns.
CREATE OR REPLACE VIEW sweep AS
SELECT
    model,
    family,
    nominal_distance_m,
    COUNT(DISTINCT tree_id)                                                        AS n_trees,
    COUNT(*)                                                                       AS n_frames,
    SUM(CASE WHEN target_masked_valid THEN 1 ELSE 0 END)                           AS target_frames,
    ROUND(AVG(CASE WHEN target_masked_valid THEN target_masked_reference_m END), 4) AS target_reference_mean_m,
    ROUND(AVG(CASE WHEN target_masked_valid THEN target_masked_predicted_m END), 4) AS target_predicted_mean_m,
    ROUND(AVG(CASE WHEN target_masked_valid THEN target_masked_signed_m END), 4)    AS target_signed_mean_m,
    ROUND(AVG(CASE WHEN target_masked_valid
                   THEN target_masked_predicted_m / NULLIF(target_masked_reference_m, 0) END), 3)
                                                                                   AS target_ratio_mean,
    ROUND(AVG(mask_mae_m), 4)                                                      AS mask_mae_mean_m,
    ROUND(AVG(mask_signed_median_m), 4)                                            AS mask_signed_median_mean_m
FROM frames
WHERE pose_set = 'sweep'
GROUP BY model, family, nominal_distance_m
ORDER BY model, family, nominal_distance_m;
