import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import URDFLoader from 'urdf-loader';
import type { URDFRobot } from 'urdf-loader';
import { parseManifest, type AssetManifest, type QuatWxyz } from '../config/assets';
import { JOINT_NAMES, TOOL_FRAME, validateJointPose } from '../config/robot.config';

export type CameraPreset = 'orbit' | 'front' | 'side' | 'top' | 'follow';
export type LightPreset = 'morning' | 'noon' | 'evening';
export interface ViewerStatus { robot: string; scene: string; ready: boolean; error?: string }
export interface ViewerMetrics { fps: number; triangles: number; camera: string; jointCount: number }
export const LIGHT_PRESETS = {
  morning: { azimuth: 65, elevation: 25, color: 0xffd8a6, intensity: 3, ambient: 1.5 },
  noon: { azimuth: 160, elevation: 70, color: 0xfff5e7, intensity: 3.8, ambient: 1.8 },
  evening: { azimuth: 255, elevation: 15, color: 0xffb877, intensity: 2.4, ambient: 1.2 },
} as const;

export class SceneViewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(44, 1, 0.02, 100);
  private wrist = new THREE.PerspectiveCamera(58, 1.5, 0.005, 100);
  private controls: OrbitControls;
  private sun = new THREE.DirectionalLight(0xfff5e7, 3.8);
  private sky = new THREE.HemisphereLight(0xc4daeb, 0x5d4931, 1.8);
  private robot?: URDFRobot;
  private mounted = false;
  private preset: CameraPreset = 'orbit';
  private disposed = false;
  private frame = 0;
  private fpsStart = 0;
  private fpsFrames = 0;
  private resize: ResizeObserver;
  private manifest?: AssetManifest;
  private status: ViewerStatus = { robot: 'Loading local bundle…', scene: 'Waiting for manifest', ready: false };

  constructor(private host: HTMLElement, private onStatus: (value: ViewerStatus) => void,
    private onMetrics: (value: ViewerMetrics) => void, private onTick: (delta: number) => void) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.domElement.setAttribute('aria-label', 'Interactive UR5e scene and tool-mounted preview');
    this.renderer.domElement.setAttribute('role', 'img');
    host.append(this.renderer.domElement);
    this.scene.background = new THREE.Color(0x1a2329);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 0.2;
    this.controls.maxDistance = 18;
    this.controls.addEventListener('start', () => { if (this.preset === 'follow') this.setCamera('orbit'); });
    const grid = new THREE.GridHelper(12, 60, 0x506266, 0x344247);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.009;
    this.scene.add(grid, new THREE.AxesHelper(0.25));
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(12, 12),
      new THREE.MeshStandardMaterial({ color: 0x303e40, roughness: 1 }));
    ground.position.z = -0.02;
    ground.receiveShadow = true;
    this.scene.add(ground, this.sun, this.sun.target, this.sky);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    Object.assign(this.sun.shadow.camera, { left: -4, right: 4, top: 5, bottom: -3, near: 0.1, far: 30 });
    this.sun.shadow.bias = -0.0001;
    this.sun.shadow.normalBias = 0.008;
    this.setLighting('noon');
    this.setCamera('orbit');
    this.resize = new ResizeObserver(() => this.resizeCanvas());
    this.resize.observe(host);
    this.resizeCanvas();
    let previous = performance.now();
    const animate = (now: number) => {
      if (this.disposed) return;
      this.frame = requestAnimationFrame(animate);
      this.onTick(Math.min(Math.max((now - previous) / 1000, 0), 0.1));
      previous = now;
      this.draw(now);
    };
    this.frame = requestAnimationFrame(animate);
    void this.loadAssets();
  }
  private resizeCanvas() {
    const { width, height } = this.host.getBoundingClientRect();
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }
  private updateStatus(patch: Partial<ViewerStatus>) {
    this.status = { ...this.status, ...patch };
    if (!this.disposed) this.onStatus(this.status);
  }
  private manager() {
    const manager = new THREE.LoadingManager();
    manager.setURLModifier(url => {
      // glTF embeds texture data in blob URLs. All fetched files must remain in
      // this local bundle; a URDF must not make arbitrary network requests.
      if (url.startsWith('blob:')) return url;
      const resolved = new URL(url, window.location.href);
      if (resolved.origin !== window.location.origin || !resolved.pathname.startsWith('/local-assets/')) {
        throw new Error('Asset dependency outside the local bundle');
      }
      return resolved.href;
    });
    return manager;
  }
  private async loadAssets() {
    try {
      const response = await fetch('/local-assets/manifest.json');
      if (!response.ok) throw new Error('Local asset bundle not found. Run tools/export_studio_assets.py; see studio/NOTES.md.');
      this.manifest = parseManifest(await response.json());
      this.updateStatus({ robot: 'Loading URDF and CAD meshes…', scene: this.manifest.scene ? 'Loading orchard GLB…' : 'No orchard GLB in bundle' });
      const manager = this.manager();
      const meshErrors: string[] = [];
      manager.onError = url => { meshErrors.push(url); };
      const loader = new URDFLoader(manager);
      loader.parseCollision = false;
      // Manager completion waits for referenced meshes, not just URDF XML.
      const complete = new Promise<void>(resolve => { manager.onLoad = () => resolve(); });
      const robot = await loader.loadAsync(`/local-assets/${this.manifest.robot.urdf}`);
      await complete;
      if (this.disposed) { this.disposeObject(robot); return; }
      if (meshErrors.length) { this.disposeObject(robot); throw new Error(`${meshErrors.length} robot mesh dependencies failed to load`); }
      const movable = Object.values(robot.joints).filter(joint => joint.jointType !== 'fixed');
      if (movable.length !== 6 || JOINT_NAMES.some(name => !robot.joints[name])) {
        this.disposeObject(robot); throw new Error('Loaded URDF does not contain exactly the six expected arm joints');
      }
      if (!robot.frames[TOOL_FRAME]) { this.disposeObject(robot); throw new Error(`URDF is missing ${TOOL_FRAME}`); }
      robot.position.fromArray(this.manifest.robot.base_position_m);
      this.setQuaternion(robot, this.manifest.robot.base_quaternion_wxyz ?? [1, 0, 0, 0]);
      robot.traverse(node => { if (node instanceof THREE.Mesh) { node.castShadow = true; node.receiveShadow = true; } });
      this.robot = robot;
      this.scene.add(robot);
      this.updateStatus({ robot: 'UR5e + mock pruner · 6 joints', ready: true });
      if (this.manifest.scene) {
        try {
          const gltf = await new GLTFLoader(this.manager()).loadAsync(`/local-assets/${this.manifest.scene.gltf}`);
          if (this.disposed) { this.disposeObject(gltf.scene); return; }
          const root = new THREE.Group();
          // glTF has Y up; the source scene manifest uses the Isaac Z-up frame.
          gltf.scene.rotation.x = Math.PI / 2;
          root.add(gltf.scene);
          root.position.fromArray(this.manifest.scene.position_m);
          this.setQuaternion(root, this.manifest.scene.quaternion_wxyz);
          root.traverse(node => { if (node instanceof THREE.Mesh) { node.castShadow = true; node.receiveShadow = true; } });
          this.scene.add(root);
          this.updateStatus({ scene: `${this.manifest.scene.tree_count} original trees · local orchard GLB` });
        } catch (error) { this.updateStatus({ scene: `Orchard unavailable: ${String(error)}` }); }
      }
    } catch (error) {
      this.updateStatus({ robot: 'Assets unavailable', scene: 'No substitute robot or trees generated', error: String(error), ready: false });
    }
  }
  private setQuaternion(object: THREE.Object3D, q: QuatWxyz) { object.quaternion.set(q[1], q[2], q[3], q[0]); }
  setPose(values: number[]) {
    const pose = validateJointPose(values);
    if (this.robot) JOINT_NAMES.forEach((name, i) => this.robot!.setJointValue(name, pose[i]));
  }
  setCamera(preset: CameraPreset) {
    this.preset = preset;
    this.camera.up.set(0, 0, 1);
    this.controls.target.set(0.4, 0.75, 1.35);
    const positions: Record<CameraPreset, [number, number, number]> = {
      orbit: [4, -4.2, 3.15], front: [0.4, -5, 1.7], side: [5, 0.75, 1.7], top: [0.4, 0.75, 7], follow: [1.6, -1.6, 2],
    };
    if (preset === 'top') this.camera.up.set(0, 1, 0);
    this.camera.position.fromArray(positions[preset]);
    this.controls.update();
  }
  setLighting(preset: LightPreset) {
    const light = LIGHT_PRESETS[preset];
    const az = THREE.MathUtils.degToRad(light.azimuth), el = THREE.MathUtils.degToRad(light.elevation);
    this.sun.position.set(10 * Math.cos(el) * Math.cos(az), 10 * Math.cos(el) * Math.sin(az), 10 * Math.sin(el));
    this.sun.target.position.set(0, 0, 0);
    this.sun.color.setHex(light.color); this.sun.intensity = light.intensity; this.sky.intensity = light.ambient;
  }
  private updateMountedCamera() {
    const tool = this.robot?.frames[TOOL_FRAME];
    if (!tool) return;
    tool.updateWorldMatrix(true, false);
    const eye = tool.localToWorld(new THREE.Vector3(-0.07235569, -0.11985263, -0.025));
    const target = tool.localToWorld(new THREE.Vector3(0, 0, 0.18));
    const rotation = tool.getWorldQuaternion(new THREE.Quaternion());
    this.wrist.position.copy(eye);
    this.wrist.up.set(0, 1, 0).applyQuaternion(rotation);
    this.wrist.lookAt(target);
    this.mounted = true;
    if (this.preset === 'follow') {
      const center = tool.getWorldPosition(new THREE.Vector3());
      this.controls.target.copy(center);
      this.camera.position.copy(center).add(new THREE.Vector3(1.4, -1.8, 1.1));
    }
  }
  private draw(now: number) {
    this.scene.updateMatrixWorld(true);
    this.updateMountedCamera();
    this.controls.update();
    const size = this.renderer.getSize(new THREE.Vector2());
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, size.x, size.y);
    this.renderer.render(this.scene, this.camera);
    let triangles = this.renderer.info.render.triangles;
    if (this.mounted) {
      const width = Math.min(288, size.x * 0.34), height = width / 1.5;
      this.renderer.setScissorTest(true);
      this.renderer.setScissor(size.x - width - 16, 16, width, height);
      this.renderer.setViewport(size.x - width - 16, 16, width, height);
      this.renderer.clearDepth();
      this.renderer.render(this.scene, this.wrist);
      triangles += this.renderer.info.render.triangles;
      this.renderer.setScissorTest(false);
    }
    if (!this.fpsStart) this.fpsStart = now;
    this.fpsFrames++;
    const elapsed = now - this.fpsStart;
    if (elapsed >= 1000) {
      this.onMetrics({ fps: this.fpsFrames * 1000 / elapsed, triangles, camera: this.preset, jointCount: this.robot ? 6 : 0 });
      this.fpsStart = now; this.fpsFrames = 0;
    }
  }
  private disposeObject(object: THREE.Object3D) {
    object.traverse(node => {
      if (!(node instanceof THREE.Mesh)) return;
      node.geometry.dispose();
      for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
        for (const value of Object.values(material)) if (value instanceof THREE.Texture) value.dispose();
        material.dispose();
      }
    });
  }
  dispose() {
    this.disposed = true; cancelAnimationFrame(this.frame); this.resize.disconnect(); this.controls.dispose();
    this.disposeObject(this.scene); this.renderer.dispose(); this.renderer.domElement.remove();
  }
}
