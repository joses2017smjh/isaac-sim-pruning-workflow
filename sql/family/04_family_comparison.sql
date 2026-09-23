-- Envy against UFO under the same light, on per-tree means so one tree with
-- many views cannot dominate. With one tree per family this view still runs,
-- and n_trees says exactly how little it is worth.
CREATE OR REPLACE VIEW family_comparison AS
WITH per_tree AS (
    SELECT model, family, tree_id, lighting,
           AVG(mask_mae_m) AS mask_mae_m,
           AVG(mask_mae_debiased_m) AS mask_mae_debiased_m
    FROM frames
    WHERE mask_mae_m IS NOT NULL AND family IN ('envy', 'ufo')
    GROUP BY model, family, tree_id, lighting
)
SELECT
    model, lighting,
    COUNT(DISTINCT CASE WHEN family = 'envy' THEN tree_id END) AS envy_trees,
    COUNT(DISTINCT CASE WHEN family = 'ufo'  THEN tree_id END) AS ufo_trees,
    ROUND(AVG(CASE WHEN family = 'envy' THEN mask_mae_m END), 4) AS envy_mask_mae_m,
    ROUND(AVG(CASE WHEN family = 'ufo'  THEN mask_mae_m END), 4) AS ufo_mask_mae_m,
    ROUND(AVG(CASE WHEN family = 'ufo'  THEN mask_mae_m END)
        - AVG(CASE WHEN family = 'envy' THEN mask_mae_m END), 4)  AS ufo_minus_envy_m,
    ROUND(AVG(CASE WHEN family = 'envy' THEN mask_mae_debiased_m END), 4) AS envy_debiased_m,
    ROUND(AVG(CASE WHEN family = 'ufo'  THEN mask_mae_debiased_m END), 4) AS ufo_debiased_m
FROM per_tree
GROUP BY model, lighting
ORDER BY model,
    CASE lighting WHEN 'source' THEN 0 WHEN 'morning' THEN 1 WHEN 'noon' THEN 2 WHEN 'evening' THEN 3 ELSE 9 END;
