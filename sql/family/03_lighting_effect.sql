-- Lighting effect within each tree, so the comparison is paired: the same
-- geometry, views and seed under each preset, only the light differs.
CREATE OR REPLACE VIEW lighting_effect AS
WITH per_tree_light AS (
    SELECT model, family, tree_id, lighting,
           AVG(mask_mae_m) AS mask_mae_m,
           AVG(mask_signed_median_m) AS signed_m
    FROM frames
    WHERE mask_mae_m IS NOT NULL
    GROUP BY model, family, tree_id, lighting
),
source AS (
    SELECT model, tree_id, mask_mae_m AS source_mae_m
    FROM per_tree_light WHERE lighting = 'source'
)
SELECT
    p.model, p.family, p.tree_id, p.lighting,
    ROUND(p.mask_mae_m, 4)            AS mask_mae_m,
    ROUND(s.source_mae_m, 4)          AS source_mae_m,
    ROUND(p.mask_mae_m / NULLIF(s.source_mae_m, 0), 3) AS ratio_to_source,
    ROUND(p.signed_m, 4)              AS signed_median_m
FROM per_tree_light p
JOIN source s USING (model, tree_id)
ORDER BY p.model, p.family, p.tree_id,
    CASE p.lighting WHEN 'source' THEN 0 WHEN 'morning' THEN 1 WHEN 'noon' THEN 2 WHEN 'evening' THEN 3 ELSE 9 END;
