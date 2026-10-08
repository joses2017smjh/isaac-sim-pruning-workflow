import { JOINT_NAMES } from './robot.config';
export type Vec3 = [number, number, number];
export type QuatWxyz = [number, number, number, number];
export interface AssetManifest {
  schema_version: 1; robot_id: 'ur5e_pruner'; units: 'm'; world_up: 'Z'; quaternion_order: 'wxyz';
  joint_order: string[];
  robot: { urdf: string; base_position_m: Vec3; base_quaternion_wxyz?: QuatWxyz };
  scene?: { gltf: string; tree_count: number; position_m: Vec3; quaternion_wxyz: QuatWxyz };
}
export function relativeAssetPath(value: unknown): string {
  if (typeof value !== 'string' || !value || !/^[A-Za-z0-9_.\/-]+$/.test(value)
    || value.startsWith('/') || value.split('/').some(part => !part || part === '.' || part === '..')) {
    throw new Error('Asset paths must be relative, without traversal, queries or external URLs');
  }
  return value;
}
function vector(value: unknown, length: number, name: string): number[] {
  if (!Array.isArray(value) || value.length !== length || value.some(v => typeof v !== 'number' || !Number.isFinite(v))) {
    throw new Error(`${name} must contain ${length} finite numbers`);
  }
  return value;
}
function quaternion(value: unknown, name: string): QuatWxyz {
  const q = vector(value, 4, name);
  if (Math.abs(Math.hypot(...q) - 1) > 1e-5) throw new Error(`${name} must be a unit wxyz quaternion`);
  return q as QuatWxyz;
}
export function parseManifest(value: unknown): AssetManifest {
  if (value === null || typeof value !== 'object') throw new Error('Invalid asset manifest');
  const m = value as Record<string, unknown>;
  if (m.schema_version !== 1 || m.robot_id !== 'ur5e_pruner' || m.units !== 'm' || m.world_up !== 'Z'
    || m.quaternion_order !== 'wxyz') throw new Error('Unsupported robot, schema or frame convention');
  if (JSON.stringify(m.joint_order) !== JSON.stringify(JOINT_NAMES)) throw new Error('Joint order differs from the six-joint UR5e config');
  const robot = m.robot as Record<string, unknown> | undefined;
  if (!robot) throw new Error('Missing robot asset');
  const result: AssetManifest = { schema_version: 1, robot_id: 'ur5e_pruner', units: 'm', world_up: 'Z',
    quaternion_order: 'wxyz', joint_order: [...JOINT_NAMES], robot: {
      urdf: relativeAssetPath(robot.urdf), base_position_m: vector(robot.base_position_m, 3, 'Base position') as Vec3,
      base_quaternion_wxyz: quaternion(robot.base_quaternion_wxyz ?? [1, 0, 0, 0], 'Base orientation'),
    } };
  if (m.scene !== undefined) {
    const scene = m.scene as Record<string, unknown>;
    if (!scene || !Number.isInteger(scene.tree_count) || Number(scene.tree_count) < 0) throw new Error('Invalid scene tree count');
    result.scene = { gltf: relativeAssetPath(scene.gltf), tree_count: Number(scene.tree_count),
      position_m: vector(scene.position_m, 3, 'Scene position') as Vec3,
      quaternion_wxyz: quaternion(scene.quaternion_wxyz, 'Scene orientation') };
  }
  return result;
}
