-- The renderer gap on identical pixels: Isaac minus Cycles affine ceiling on
-- the shared tree pixels of the same pose, per bark and light.
CREATE OR REPLACE VIEW gap_paired AS
WITH isaac AS (
    SELECT isaac_run, isaac_frame_index, lighting, affine_ceiling_m AS isaac_ceiling_m, correlation AS isaac_corr
    FROM gap_input WHERE renderer = 'isaac' AND pixel_set = 'isaac_on_shared_tree'
),
cycles AS (
    SELECT isaac_run, isaac_frame_index, bark, affine_ceiling_m AS cycles_ceiling_m, constant_floor_m, correlation AS cycles_corr
    FROM gap_input WHERE renderer = 'cycles' AND pixel_set = 'cycles_on_shared_tree'
)
SELECT
    c.bark,
    i.lighting,
    COUNT(*)                                                        AS n_frames,
    ROUND(AVG(i.isaac_ceiling_m), 4)                                AS isaac_ceiling_mean_m,
    ROUND(AVG(c.cycles_ceiling_m), 4)                               AS cycles_ceiling_mean_m,
    ROUND(AVG(c.constant_floor_m), 4)                               AS constant_floor_mean_m,
    ROUND(AVG(i.isaac_ceiling_m - c.cycles_ceiling_m), 4)           AS gap_mean_m,
    ROUND(MIN(i.isaac_ceiling_m - c.cycles_ceiling_m), 4)           AS gap_min_m,
    ROUND(MAX(i.isaac_ceiling_m - c.cycles_ceiling_m), 4)           AS gap_max_m,
    ROUND(STDDEV_SAMP(i.isaac_ceiling_m - c.cycles_ceiling_m), 4)   AS gap_std_m,
    SUM(CASE WHEN i.isaac_ceiling_m > c.cycles_ceiling_m THEN 1 ELSE 0 END) AS frames_isaac_worse,
    ROUND(AVG(i.isaac_corr), 3)                                     AS isaac_correlation_mean,
    ROUND(AVG(c.cycles_corr), 3)                                    AS cycles_correlation_mean
FROM isaac i JOIN cycles c USING (isaac_run, isaac_frame_index)
GROUP BY c.bark, i.lighting
ORDER BY c.bark, CASE i.lighting WHEN 'source' THEN 0 WHEN 'morning' THEN 1 WHEN 'evening' THEN 2 ELSE 9 END;
