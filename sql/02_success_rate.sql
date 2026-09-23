-- X / N with a Wilson 95% score interval.
--
-- Wilson rather than the normal approximation on purpose: at N = 20 with a
-- proportion near 0 or 1, the normal interval runs past 0 or 1 and understates
-- uncertainty. Wilson stays inside [0, 1] and is defined when x = 0.
--
-- The denominator is every attempted target. Rejections and incompletes are in
-- it. Dropping them would report a rate over a population nobody registered.
CREATE OR REPLACE VIEW success_rate AS
WITH counted AS (
    SELECT
        COUNT(*)::DOUBLE                             AS n,
        SUM(CASE WHEN passed THEN 1 ELSE 0 END)::DOUBLE AS x
    FROM runs
),
wilson AS (
    SELECT
        n, x,
        CASE WHEN n > 0 THEN x / n END AS p_hat,
        1.959963984540054              AS z
    FROM counted
)
SELECT
    CAST(x AS BIGINT) AS successes,
    CAST(n AS BIGINT) AS attempts,
    p_hat             AS rate,
    -- Clamped to [0, 1]: at x = 0 or x = n the closed form lands a few parts in
    -- 1e17 outside the interval, and a reported rate of -1e-17 is nonsense.
    CASE WHEN n > 0 THEN greatest(0.0,
        ((p_hat + z * z / (2 * n)) / (1 + z * z / n))
        - (z / (1 + z * z / n)) * sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    ) END AS wilson_lower,
    CASE WHEN n > 0 THEN least(1.0,
        ((p_hat + z * z / (2 * n)) / (1 + z * z / n))
        + (z / (1 + z * z / n)) * sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    ) END AS wilson_upper
FROM wilson;
