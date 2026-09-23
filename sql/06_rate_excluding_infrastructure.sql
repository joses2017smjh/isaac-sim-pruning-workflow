-- The registered protocol reports infrastructure failures separately "so a
-- reader can see a rate that excludes them". This is that view. It is
-- secondary: the headline is always the inclusive rate in success_rate.
--
-- Two exclusion levels are shown side by side so neither is mistaken for the
-- other. "presented" drops trials whose target could never be put in front of
-- the robot. "recorded" additionally drops layout rejections, leaving only
-- trials where the controller actually ran.
CREATE OR REPLACE VIEW rate_by_exclusion AS
WITH levels AS (
    SELECT 'inclusive (headline)' AS level, 1 AS ord, * FROM runs
    UNION ALL
    SELECT 'presented (excludes infrastructure)', 2, * FROM runs WHERE outcome <> 'infrastructure'
    UNION ALL
    SELECT 'recorded (excludes infrastructure and layout rejection)', 3, * FROM runs
        WHERE outcome NOT IN ('infrastructure', 'rejected_layout_startup_contact')
),
counted AS (
    SELECT level, ord, COUNT(*)::DOUBLE AS n,
           SUM(CASE WHEN passed THEN 1 ELSE 0 END)::DOUBLE AS x
    FROM levels GROUP BY level, ord
)
SELECT
    level,
    CAST(x AS BIGINT) AS successes,
    CAST(n AS BIGINT) AS attempts,
    x / n AS rate,
    greatest(0.0, ((x / n + 3.841458820694124 / (2 * n)) / (1 + 3.841458820694124 / n))
        - (1.959963984540054 / (1 + 3.841458820694124 / n))
          * sqrt((x / n) * (1 - x / n) / n + 3.841458820694124 / (4 * n * n))) AS wilson_lower,
    least(1.0, ((x / n + 3.841458820694124 / (2 * n)) / (1 + 3.841458820694124 / n))
        + (1.959963984540054 / (1 + 3.841458820694124 / n))
          * sqrt((x / n) * (1 - x / n) / n + 3.841458820694124 / (4 * n * n))) AS wilson_upper
FROM counted
WHERE n > 0
ORDER BY ord;
