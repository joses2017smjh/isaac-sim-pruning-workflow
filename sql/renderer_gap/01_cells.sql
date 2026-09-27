-- One cell per renderer x bark x light x pixel set. Frames are poses of the
-- recorded Stage A approaches; neighbouring poses are strongly correlated, so
-- n_frames overstates the independent sample size.
CREATE OR REPLACE VIEW gap_cells AS
SELECT
    renderer,
    bark,
    lighting,
    pixel_set,
    COUNT(*)                                AS n_frames,
    ROUND(AVG(pixels), 0)                   AS pixels_mean,
    ROUND(AVG(mae_m), 4)                    AS mae_mean_m,
    ROUND(AVG(signed_median_m), 4)          AS signed_median_mean_m,
    ROUND(AVG(median_prediction_m), 4)      AS median_prediction_mean_m,
    ROUND(AVG(median_reference_m), 4)       AS median_reference_mean_m,
    ROUND(AVG(correlation), 3)              AS correlation_mean,
    ROUND(AVG(affine_ceiling_m), 4)         AS affine_ceiling_mean_m,
    ROUND(AVG(constant_floor_m), 4)         AS constant_floor_mean_m
FROM gap_input
GROUP BY renderer, bark, lighting, pixel_set
ORDER BY pixel_set, renderer, bark,
    CASE lighting WHEN 'source' THEN 0 WHEN 'morning' THEN 1 WHEN 'evening' THEN 2 ELSE 9 END;
