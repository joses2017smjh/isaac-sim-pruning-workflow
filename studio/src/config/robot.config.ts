// Joint order and limits match config/robot/ur5e_pruner.yaml. Fixed base; no slider.
export const JOINT_NAMES = [
  'ur5e__shoulder_pan_joint', 'ur5e__shoulder_lift_joint', 'ur5e__elbow_joint',
  'ur5e__wrist_1_joint', 'ur5e__wrist_2_joint', 'ur5e__wrist_3_joint',
] as const;
export const JOINT_LABELS = ['Shoulder pan', 'Shoulder lift', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3'];
export const JOINT_LIMITS = [2 * Math.PI, 2 * Math.PI, Math.PI, 2 * Math.PI, 2 * Math.PI, 2 * Math.PI];
export const TOOL_FRAME = 'mock_pruner__tool0';
// A modest rigging preview, not an executed Isaac trajectory or safe motion plan.
export const PREVIEW_HOME = [0, -1.3, 1.6, -1.85, -1.57, 0];
export const PREVIEW_DURATION = 12;
export function previewPose(time: number): number[] {
  if (!Number.isFinite(time) || time < 0) throw new Error('Preview time must be finite and non-negative');
  return PREVIEW_HOME.map((value, joint) => value + [0.24, 0.12, 0.16, 0.2, 0.12, 0.2][joint]
    * Math.sin(2 * Math.PI * time / PREVIEW_DURATION + joint * 0.4));
}
export function validateJointPose(values: unknown): number[] {
  if (!Array.isArray(values) || values.length !== JOINT_NAMES.length
    || values.some((value, i) => typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > JOINT_LIMITS[i])) {
    throw new Error('Expected six finite joint angles in UR5e limit ranges');
  }
  return [...values];
}
