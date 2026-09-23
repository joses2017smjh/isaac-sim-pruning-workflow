-- Tracking coverage against measured error, and against the tracker's own
-- reject floors. A run that tracked well and still failed is a different story
-- from one that never tracked at all, and the taxonomy alone does not say which.
CREATE OR REPLACE VIEW tracking_coverage AS
SELECT
    run_directory,
    outcome,
    passed,
    tracking_frames,
    recorded_frames,
    CASE WHEN recorded_frames > 0
         THEN ROUND(tracking_frames::DOUBLE / recorded_frames, 4) END AS tracking_fraction,
    tracked_feature_count_min,
    tracked_confidence_min,
    centroid_error_m,
    -- 4 features and 0.15 confidence are the tracker's configured floors.
    (tracked_feature_count_min IS NOT NULL AND tracked_feature_count_min <= 4)  AS touched_feature_floor,
    (tracked_confidence_min   IS NOT NULL AND tracked_confidence_min   <= 0.20) AS near_confidence_floor
FROM runs
ORDER BY passed DESC, tracking_fraction DESC NULLS LAST;
