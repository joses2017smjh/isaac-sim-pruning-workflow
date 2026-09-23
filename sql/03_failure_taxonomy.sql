-- How the attempts that did not pass actually ended, taken from recorded stop
-- reasons rather than from a hand-written list.
CREATE OR REPLACE VIEW failure_taxonomy AS
SELECT
    outcome,
    COUNT(*)                             AS runs,
    ROUND(AVG(applied_commands), 1)      AS mean_applied_commands,
    MIN(applied_commands)                AS min_applied_commands,
    MAX(applied_commands)                AS max_applied_commands,
    list(run_directory ORDER BY run_directory) AS run_directories
FROM runs
WHERE NOT passed
GROUP BY outcome
ORDER BY runs DESC, outcome;
