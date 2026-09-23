-- The same rate split by the condition it was run under, so a combined number
-- can never hide a condition that did much worse.
CREATE OR REPLACE VIEW per_condition AS
SELECT
    condition,
    daylight,
    photometric_normalization,
    COUNT(*)                                 AS attempts,
    SUM(CASE WHEN passed THEN 1 ELSE 0 END)  AS successes,
    ROUND(SUM(CASE WHEN passed THEN 1 ELSE 0 END)::DOUBLE / COUNT(*), 4) AS rate
FROM runs
GROUP BY condition, daylight, photometric_normalization
ORDER BY condition;
