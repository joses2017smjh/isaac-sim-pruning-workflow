-- One row per evaluated frame per model, with the control axes carried
-- through. condition is the cell label; lighting, camera_model and pose_set
-- say which single axis the condition changed. Nothing is imputed.
CREATE OR REPLACE VIEW frames AS
SELECT
    model,
    family,
    tree_id,
    condition,
    condition_group,
    lighting,
    camera_model,
    pose_set,
    nominal_distance_m,
    pitch_deg,
    view_id,
    target_visible,
    mask_pixels,
    mask_mae_m,
    mask_rmse_m,
    mask_abs_relative,
    mask_p95_abs_m,
    -- Signed bias over tree pixels: negative means predicted nearer than truth.
    mask_signed_median_m,
    mask_signed_mean_m,
    -- Diagnostic only: error after removing each frame's own median offset,
    -- which uses ground truth and is therefore never an achievable number.
    mask_mae_debiased_m,
    full_mae_m,
    full_rmse_m,
    full_signed_median_m,
    target_valid,
    target_mae_m,
    target_masked_valid,
    target_masked_mae_m,
    target_masked_p95_abs_m,
    target_masked_abs_relative,
    target_masked_coverage,
    target_masked_tree_pixels,
    target_masked_predicted_m,
    target_masked_reference_m,
    target_masked_signed_m,
    inference_seconds
FROM eval_input;
