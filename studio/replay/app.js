// Replay studio: draws recorded capture telemetry, and nothing else.
//
// Every value rendered here is read from a telemetry file produced by
// tools/export_studio_replay.py from a recorded capture. There is no physics,
// no interpolation between frames and no policy in this page. When a field was
// not recorded, the page shows a dash rather than a plausible-looking number.

const PHASE_COLORS = {
  vision_approach: '#4f7fa8',
  vision_hold: '#7a6aa0',
  align: '#3f8f77',
  simulated_closure: '#b8863a',
  retreat: '#6b8c4a',
  complete: '#57b87f',
  stopped_failure: '#d9705f',
  observe: '#59666e',
};
const CONFIDENCE_FLOOR = 0.15;   // VisualServoConfig.min_confidence
const FEATURE_FLOOR = 4;         // VisualServoConfig.min_features

const $ = (id) => document.getElementById(id);
const state = { manifest: null, run: null, data: null, frame: 0, playing: false, timer: null };

const fmt = (value, digits = 2, unit = '') =>
  value === null || value === undefined ? '—' : `${Number(value).toFixed(digits)}${unit}`;

async function boot() {
  let manifest;
  try {
    const response = await fetch('data/manifest.json', { cache: 'no-cache' });
    if (!response.ok) throw new Error(`manifest ${response.status}`);
    manifest = await response.json();
  } catch (error) {
    $('provenance').textContent = `Could not load recorded runs: ${error.message}`;
    return;
  }
  state.manifest = manifest;
  $('provenance').textContent =
    `${manifest.runs.length} recorded runs · ${(manifest.payload_bytes / 1024).toFixed(0)} KB total payload`;

  const nav = $('runs');
  manifest.runs.forEach((run, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.setAttribute('aria-pressed', String(index === 0));
    button.innerHTML = `<span class="dot ${run.outcome === 'pass' ? 'pass' : 'fail'}"></span><span>${run.label}</span>`;
    button.addEventListener('click', () => selectRun(run, button));
    nav.appendChild(button);
    if (index === 0) button.dataset.first = '1';
  });
  wireTransport();
  const first = nav.querySelector('button');
  await selectRun(manifest.runs[0], first);
}

async function selectRun(run, button) {
  document.querySelectorAll('#runs button').forEach((element) => element.setAttribute('aria-pressed', 'false'));
  if (button) button.setAttribute('aria-pressed', 'true');
  pause();

  const response = await fetch(`data/${run.telemetry}`, { cache: 'no-cache' });
  state.data = await response.json();
  state.run = run;

  for (const [id, key] of [['wrist', 'wrist'], ['overview', 'overview']]) {
    const element = $(id);
    const media = state.data.media[key];
    element.src = media ? `data/${media.path}` : '';
    element.closest('figure').style.display = media ? '' : 'none';
  }

  $('scrub').max = String(state.data.frame_count - 1);
  drawPhases();
  drawChart();
  renderVerdict();
  seek(0);
}

function renderVerdict() {
  const { grade, target_id, daylight, source_job_id, frame_count, fps, photometric_normalization } = state.data;
  const pass = grade && grade.ok;
  $('verdict').innerHTML =
    `<span class="dot ${pass ? 'pass' : 'fail'}"></span>` +
    (grade ? `${grade.checks_passed} of ${grade.checks_total} independent checks pass` : 'Not graded');

  // A field the capture never recorded is labelled as such. Filling it in from
  // a default would be inventing provenance.
  const absent = '<span class="tag">not recorded</span>';
  const rows = [
    ['Job', source_job_id],
    ['Target', target_id || absent],
    ['Light', daylight || absent],
    ['Tracker', photometric_normalization || absent],
    ['Frames', `${frame_count} at ${fps} Hz`],
  ];
  if (grade && grade.failed_checks.length) {
    rows.push(['Failed checks', grade.failed_checks.join(', ')]);
  }
  $('facts').innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join('');
}

function drawPhases() {
  const frames = state.data.frames;
  const strip = $('phases');
  strip.innerHTML = '';
  const seen = new Map();
  let start = 0;
  for (let index = 1; index <= frames.length; index += 1) {
    const previous = frames[index - 1].phase;
    if (index === frames.length || frames[index].phase !== previous) {
      const span = document.createElement('span');
      const color = PHASE_COLORS[previous] || '#59666e';
      span.style.background = color;
      span.style.width = `${((index - start) / frames.length) * 100}%`;
      span.title = `${previous}: frames ${start}–${index - 1}`;
      strip.appendChild(span);
      if (previous) seen.set(previous, color);
      start = index;
    }
  }
  $('phaseLegend').innerHTML = [...seen.entries()]
    .map(([name, color]) => `<span><i style="background:${color}"></i>${name.replace(/_/g, ' ')}</span>`)
    .join('');
}

function drawChart() {
  const canvas = $('chart');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = 150;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const frames = state.data.frames;
  const style = getComputedStyle(document.body);
  const line = style.getPropertyValue('--line').trim() || '#2b353c';
  const muted = style.getPropertyValue('--muted').trim() || '#94a5ae';
  const maxFeatures = Math.max(FEATURE_FLOOR * 2, ...frames.map((f) => f.features || 0));

  // Leave room under the plot for labels so a series at full scale is never
  // hidden behind them.
  const top = 10;
  const bottom = height - 22;
  const plot = bottom - top;
  const x = (index) => (index / Math.max(1, frames.length - 1)) * (width - 2) + 1;
  const y = (fraction) => bottom - Math.max(0, Math.min(1, fraction)) * plot;

  context.strokeStyle = line;
  context.lineWidth = 1;
  context.setLineDash([3, 4]);
  for (const fraction of [CONFIDENCE_FLOOR, FEATURE_FLOOR / maxFeatures]) {
    context.beginPath();
    context.moveTo(0, y(fraction));
    context.lineTo(width, y(fraction));
    context.stroke();
  }
  context.setLineDash([]);

  const series = (accessor, color, scale) => {
    context.strokeStyle = color;
    context.lineWidth = 1.6;
    context.beginPath();
    let drawing = false;
    frames.forEach((frame, index) => {
      const raw = accessor(frame);
      if (raw === null || raw === undefined) { drawing = false; return; }
      const point = y(scale(raw));
      if (!drawing) { context.moveTo(x(index), point); drawing = true; } else { context.lineTo(x(index), point); }
    });
    context.stroke();
  };
  series((f) => (f.state === 'tracking' ? f.confidence : null), '#57b87f', (v) => v);
  series((f) => (f.state === 'tracking' ? f.features : null), '#e9963c', (v) => v / maxFeatures);

  context.font = '11px ui-sans-serif, system-ui, sans-serif';
  context.fillStyle = '#57b87f';
  context.fillText('confidence', 2, height - 7);
  context.fillStyle = '#e9963c';
  context.fillText(`features (peak ${maxFeatures})`, 76, height - 7);
  context.fillStyle = muted;
  context.textAlign = 'right';
  context.fillText('dashed: reject floors', width - 2, height - 7);
  context.textAlign = 'left';

  state.chartCursor = drawChartCursorOverlay;
  drawChartCursorOverlay(state.frame);
}

let chartOverlay = null;
function drawChartCursorOverlay(index) {
  const canvas = $('chart');
  if (!chartOverlay) {
    chartOverlay = document.createElement('div');
    chartOverlay.style.cssText = 'position:absolute;width:1px;background:#e9963c;opacity:.75;pointer-events:none';
    canvas.parentElement.style.position = 'relative';
    canvas.parentElement.appendChild(chartOverlay);
  }
  const frames = state.data.frames.length;
  chartOverlay.style.height = `${canvas.clientHeight}px`;
  chartOverlay.style.top = `${canvas.offsetTop}px`;
  chartOverlay.style.left = `${canvas.offsetLeft + (index / Math.max(1, frames - 1)) * canvas.clientWidth}px`;
}

function drawToF(canvasId, grid, validId) {
  const canvas = $(canvasId);
  const context = canvas.getContext('2d');
  const cell = canvas.width / 8;
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!grid) {
    context.fillStyle = '#59666e';
    context.fillText('no grid', 8, 20);
    $(validId).textContent = '—';
    return;
  }
  const values = grid.flat().filter((value) => value !== null);
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const value = grid[row][col];
      const x = col * cell;
      const y = row * cell;
      if (value === null) {
        // Hatched, never a colour from the range ramp: this zone has no measurement.
        context.fillStyle = '#1e262b';
        context.fillRect(x, y, cell, cell);
        context.strokeStyle = '#3c4950';
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(x, y + cell); context.lineTo(x + cell, y);
        context.moveTo(x, y + cell / 2); context.lineTo(x + cell / 2, y);
        context.moveTo(x + cell / 2, y + cell); context.lineTo(x + cell, y + cell / 2);
        context.stroke();
      } else {
        const t = max > min ? (value - min) / (max - min) : 0;
        // Near is warm, far is cool.
        const r = Math.round(240 - 85 * t);
        const g = Math.round(214 - 139 * t);
        const b = Math.round(122 + 21 * t);
        context.fillStyle = `rgb(${r},${g},${b})`;
        context.fillRect(x, y, cell, cell);
      }
    }
  }
  $(validId).textContent = `${values.length}/64 valid`;
}

function seek(index) {
  const frames = state.data.frames;
  state.frame = Math.max(0, Math.min(frames.length - 1, index));
  const frame = frames[state.frame];

  $('scrub').value = String(state.frame);
  $('clock').textContent = `${(frame.t ?? 0).toFixed(1)} s`;

  const seconds = (frame.t ?? 0) - (frames[0].t ?? 0);
  for (const id of ['wrist', 'overview']) {
    const element = $(id);
    if (element.src && Number.isFinite(element.duration) && Math.abs(element.currentTime - seconds) > 0.12) {
      try { element.currentTime = seconds; } catch { /* seeking before metadata is loaded */ }
    }
  }

  $('tState').textContent = frame.state || '—';
  $('tConf').textContent = frame.state === 'tracking' ? fmt(frame.confidence, 3) : '—';
  $('tFeat').textContent = frame.state === 'tracking' ? (frame.features ?? '—') : '—';
  $('tDepth').textContent = frame.state === 'tracking' ? fmt(frame.depth_valid_fraction, 2) : '—';

  $('dState').textContent = frame.decision_state || '—';
  $('dReason').textContent = frame.decision_reason || frame.reason || '—';
  $('dPhase').textContent = frame.cut_phase || frame.phase || '—';
  $('dContact').textContent = fmt(frame.contact_n, 2, ' N');

  const proposed = frame.proposed_delta_mm;
  $('cProposed').textContent = proposed
    ? `${proposed.map((v) => (v === null ? '—' : v.toFixed(2))).join(', ')} mm`
    : '—';
  const applied = frame.applied_xyz;
  $('cApplied').textContent = applied ? applied.map((v) => (v === null ? '—' : v.toFixed(3))).join(', ') : '—';
  $('cRemain').textContent = fmt(frame.remaining_m === null ? null : frame.remaining_m * 1000, 1, ' mm');

  const banner = $('stopBanner');
  if (frame.stopped_reason) {
    banner.hidden = false;
    banner.textContent = `Stopped: ${frame.stopped_reason}. The run holds from here; it did not release.`;
  } else if (frame.released) {
    banner.hidden = false;
    banner.textContent = 'Surrogate release requested on this frame.';
  } else {
    banner.hidden = true;
  }

  drawToF('tofL', state.data.tof_mm.left[state.frame], 'tofLv');
  drawToF('tofR', state.data.tof_mm.right[state.frame], 'tofRv');
  if (state.chartCursor) state.chartCursor(state.frame);
}

function wireTransport() {
  $('scrub').addEventListener('input', (event) => { pause(); seek(Number(event.target.value)); });
  $('play').addEventListener('click', () => (state.playing ? pause() : play()));
  window.addEventListener('resize', () => { if (state.data) drawChart(); });
  document.addEventListener('keydown', (event) => {
    if (event.key === ' ') { event.preventDefault(); state.playing ? pause() : play(); }
    if (event.key === 'ArrowRight') seek(state.frame + 1);
    if (event.key === 'ArrowLeft') seek(state.frame - 1);
  });
}

function play() {
  if (!state.data) return;
  state.playing = true;
  $('play').textContent = 'Pause';
  for (const id of ['wrist', 'overview']) { const el = $(id); if (el.src) el.play().catch(() => {}); }
  const period = 1000 / (state.data.fps || 10);
  state.timer = setInterval(() => {
    if (state.frame >= state.data.frames.length - 1) { pause(); return; }
    seek(state.frame + 1);
  }, period);
}

function pause() {
  state.playing = false;
  $('play').textContent = 'Play';
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
  for (const id of ['wrist', 'overview']) { const el = $(id); if (el.src) el.pause(); }
}

boot();
