-- One row per evaluated frame per model. Every reported number comes from
-- here. A frame with no mask (the Isaac Stage A frames) keeps its full-frame
-- and target columns and has NULL mask columns; nothing is imputed.
CREATE OR REPLACE VIEW frames AS
SELECT
    model,
    family,
    tree_id,
    lighting,
    view_id,
    sequence,
    target_visible,
    mask_pixels,
    mask_mae_m,
    mask_rmse_m,
    mask_abs_relative,
    mask_p95_abs_m,
    -- Signed bias over tree pixels: negative means predicted nearer than truth.
    mask_signed_median_m,
    mask_signed_mean_m,
    -- Diagnostic only. Error after removing each frame's own median offset,
    -- which uses ground truth and is therefore never an achievable number.
    mask_mae_debiased_m,
    full_mae_m,
    full_rmse_m,
    full_signed_median_m,
    target_valid,
    target_mae_m,
    target_p95_abs_m,
    target_abs_relative,
    target_coverage,
    target_masked_valid,
    target_masked_mae_m,
    target_masked_p95_abs_m,
    target_masked_abs_relative,
    target_masked_coverage,
    target_masked_tree_pixels,
    inference_seconds
FROM eval_input;
