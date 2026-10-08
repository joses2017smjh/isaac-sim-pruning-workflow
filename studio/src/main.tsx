import { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { SceneViewer, LIGHT_PRESETS, type CameraPreset, type LightPreset, type ViewerStatus, type ViewerMetrics } from './viewer/SceneViewer';
import { JOINT_LABELS, JOINT_LIMITS, PREVIEW_DURATION, previewPose } from './config/robot.config';
import './style.css';

function App() {
  const host = useRef<HTMLDivElement>(null);
  const viewer = useRef<SceneViewer | null>(null);
  const playingRef = useRef(false), timeRef = useRef(0), poseRef = useRef(previewPose(0));
  const [status, setStatus] = useState<ViewerStatus>({ robot: 'Starting viewer…', scene: 'Waiting for assets', ready: false });
  const [metrics, setMetrics] = useState<ViewerMetrics>({ fps: 0, triangles: 0, camera: 'orbit', jointCount: 0 });
  const [playing, setPlaying] = useState(false), [time, setTime] = useState(0), [pose, setPose] = useState(previewPose(0));
  const [camera, setCamera] = useState<CameraPreset>('orbit'), [light, setLight] = useState<LightPreset>('noon');
  useEffect(() => {
    if (!host.current) return;
    try {
      viewer.current = new SceneViewer(host.current, setStatus, setMetrics, delta => {
        if (playingRef.current) {
          timeRef.current = (timeRef.current + delta) % PREVIEW_DURATION;
          poseRef.current = previewPose(timeRef.current);
          setTime(timeRef.current); setPose([...poseRef.current]);
        }
        viewer.current?.setPose(poseRef.current);
      });
    } catch (error) { setStatus({ robot: 'WebGL unavailable', scene: 'Viewer not started', ready: false, error: String(error) }); }
    return () => viewer.current?.dispose();
  }, []);
  function play(value: boolean) { playingRef.current = value; setPlaying(value); }
  function seek(value: number) { play(false); timeRef.current = value; setTime(value); poseRef.current = previewPose(value); setPose([...poseRef.current]); }
  function setJoint(index: number, value: number) { play(false); poseRef.current = poseRef.current.map((old, i) => i === index ? value : old); setPose([...poseRef.current]); }
  return <main>
    <header><div><p className="eyebrow">PRUNING WORKFLOW / BROWSER LAB</p><h1>Robot Studio<span>M1 · scene viewer</span></h1></div>
      <div className="header-note">LOCAL ASSETS ONLY<br /><strong>No live physics or policy</strong></div></header>
    <section className="workspace">
      <div className="stage">
        <div className="viewport" ref={host} />
        <div className="stage-top"><span className={`status-dot ${status.ready ? 'ready' : ''}`} />{status.robot}</div>
        <div className="metrics" aria-live="off"><strong data-testid="fps">{metrics.fps ? metrics.fps.toFixed(1) : '—'}</strong> FPS
          <span>{metrics.triangles.toLocaleString()} triangles / frame</span><span>Z-up · meters</span></div>
        {status.ready && <div className="wrist-label">TOOL-MOUNTED VIEW · virtual, not sensor data</div>}
        {!status.ready && status.error && <div className="empty-state" role="alert"><h2>Bring your local assets.</h2><p>{status.error}</p>
          <code>python tools/export_studio_assets.py --help</code><p>The viewer never substitutes a different robot or duplicates a tree.</p></div>}
        <div className="scene-caption">{status.scene}</div>
      </div>
      <aside>
        <section><p className="eyebrow">01 / CAMERA</p><div className="button-grid">{(['orbit', 'front', 'side', 'top', 'follow'] as CameraPreset[]).map(value =>
          <button key={value} aria-pressed={camera === value} onClick={() => { setCamera(value); viewer.current?.setCamera(value); }}>{value}</button>)}</div>
          <p className="hint">Drag to orbit · wheel to zoom · right-drag to pan.</p></section>
        <section><p className="eyebrow">02 / VIEWER DAYLIGHT</p><div className="button-grid lights">{(['morning', 'noon', 'evening'] as LightPreset[]).map(value =>
          <button key={value} aria-pressed={light === value} onClick={() => { setLight(value); viewer.current?.setLighting(value); }}>{value}</button>)}</div>
          <p className="hint">Sun {LIGHT_PRESETS[light].elevation}° elevation / {LIGHT_PRESETS[light].azimuth}° azimuth.<br />Browser relighting only. Isaac captures do not change.</p></section>
        <section><p className="eyebrow">03 / SIX-JOINT RIG</p><div className="joints">{JOINT_LABELS.map((label, i) => <label key={label}>
          <span>{label}<output>{pose[i].toFixed(3)} rad</output></span><input aria-label={label} type="range" min={-JOINT_LIMITS[i]} max={JOINT_LIMITS[i]}
            step="0.001" value={pose[i]} disabled={!status.ready} onChange={event => setJoint(i, Number(event.target.value))} /></label>)}</div></section>
        <section className="scope"><h2>What you are seeing</h2><p>URDF forward kinematics. A fixed base, six arm joints, a mock pruner, and local orchard meshes.</p>
          <p>Animation checks rigging only. No collision checking, forces, branch tracking, cutting, or recorded replay runs here yet.</p></section>
      </aside>
    </section>
    <section className="transport"><button className="primary" disabled={!status.ready} onClick={() => play(!playing)}>{playing ? 'Pause preview' : 'Play rigging preview'}</button>
      <button disabled={!status.ready} onClick={() => seek(0)}>Reset</button><label className="timeline">Synthetic joint motion · 12 s loop
        <input aria-label="Preview time" type="range" min="0" max={PREVIEW_DURATION} step="0.01" value={time} disabled={!status.ready} onChange={event => seek(Number(event.target.value))} /></label>
      <output className="time">{time.toFixed(2)} / 12.00 s</output></section>
    <footer><span>Scene viewer ≠ simulator. FPS is measured on this browser, not an Isaac benchmark.</span><span>Next: synchronized success / failure replay.</span></footer>
  </main>;
}
createRoot(document.getElementById('root')!).render(<App />);
