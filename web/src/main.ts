import './style.css';
import { createChannel, type SimChannel } from './channel';
import type { FrameMsg, ToMain, ToWorker } from './protocol';
import { View } from './render/view';
import { Hud } from './ui/hud';
import { Chart } from './ui/chart';
import { Inspector } from './ui/inspector';
import { setupControls, selectSegment, type UiState } from './ui/controls';
import { attachInput } from './ui/input';
import { DEFAULT_SCALE } from './sim/config';

const params = new URLSearchParams(location.search);
let seed = Number(params.get('seed') ?? 1) || 0;
let scale = Number(params.get('scale') ?? DEFAULT_SCALE) || DEFAULT_SCALE;

const state: UiState = {
  paused: false,
  speed: Number(params.get('speed') ?? 4) || 4,
  colorMode: 'lineage',
  tool: 'inspect',
  layers: { pheromone: true, food: true, patches: true, bonds: true, trails: false, glow: true },
};

let channel: SimChannel | null = null;
const send = (m: ToWorker, transfer: Transferable[] = []) => channel?.send(m, transfer);
let view: View | null = null;
let canvas = document.getElementById('gl') as HTMLCanvasElement;
let latest: FrameMsg | null = null;
let awaiting = false;
let pendingSteps = 0;
let frameNo = 0;
let recycle: ArrayBuffer[] = [];
let lastStats: FrameMsg['stats'] = null;
let stepMs = 0;

const hud = new Hud(document.getElementById('hud')!);
const chart = new Chart(document.getElementById('chart') as HTMLCanvasElement, document.getElementById('chartLegend')!);
const inspector = new Inspector();
inspector.onClose = () => { send({ type: 'deselect' }); inspector.hide(); };

const toastEl = document.getElementById('toast')!;
let toastTimer = 0;
function toast(msg: string): void {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toastEl.classList.remove('show'), 2600);
}

const controls = setupControls(state, {
  onReset: (s, sc) => reset(s, sc),
  onStep: () => { state.paused = true; controls.syncPlay(); pendingSteps += 1; },
  onFit: () => view?.cam.fit(),
  onParam: (key, value) => send({ type: 'param', key, value }),
  onLayers: () => { if (view) { view.layers = state.layers; view.clearTrails(); } },
});
controls.setSeed(seed);
controls.setScale(scale);
controls.syncPlay();

function reset(s: number, sc: number): void {
  seed = s; scale = sc;
  const u = new URL(location.href);
  u.searchParams.set('seed', String(seed));
  u.searchParams.set('scale', String(scale));
  history.replaceState(null, '', u);
  chart.reset();
  inspector.hide();
  send({ type: 'reset', seed, scale });
  controls.resetParams();
  toast(`New world · seed ${seed}`);
}

function onMessage(m: ToMain): void {
  if (m.type === 'bench') return;
  if (m.type === 'ready') {
    seed = m.seed; scale = m.scale;
    if (view) {   // fresh canvas: the old WebGL context is dropped with the old element
      view.renderer.dispose();
      const c = canvas.cloneNode() as HTMLCanvasElement;
      canvas.replaceWith(c);
      canvas = c;
    }
    view = new View(canvas, m.worldSize, m.capacity, m.foodCapacity, m.pheromoneRes, m.patchRadius);
    view.layers = state.layers;
    attachInput(canvas, view, state, {
      click: (wx, wy, uwx, uwy) => clickTool(wx, wy, uwx, uwy),
      toggleHud: () => {},
    });
    view.resize(window.innerWidth, window.innerHeight);
    awaiting = false;
    latest = null;
    return;
  }
  awaiting = false;
  if (latest) recycleFrame(latest);
  latest = m;
  if (m.stats) { lastStats = m.stats; chart.push(m.stats); }
  if (m.stepsDone && m.stepMs > 0) stepMs = stepMs ? stepMs * 0.9 + m.stepMs * 0.1 : m.stepMs;
  if (m.event) { toast(m.event); inspector.hide(); }
}

function recycleFrame(f: FrameMsg): void {
  recycle.push(f.particles.buffer as ArrayBuffer, f.bonds.buffer as ArrayBuffer, f.food.buffer as ArrayBuffer);
  if (f.pheromone) recycle.push(f.pheromone.buffer as ArrayBuffer);
}

function clickTool(wx: number, wy: number, uwx: number, uwy: number): void {
  if (!view) return;
  if (state.tool === 'inspect') {
    send({ type: 'select', x: wx, y: wy, radius: Math.max(0.03, 18 / view.cam.pxPerUnit) });
  } else if (state.tool === 'food') {
    send({ type: 'tool', tool: 'food', x: wx, y: wy });
    view.burst(uwx, uwy, 0x7ee787, 0.09, 700);
  } else {
    send({ type: 'tool', tool: 'meteor', x: wx, y: wy });
    view.burst(uwx, uwy, 0xffa24a, 0.3, 1100);
    view.burst(uwx, uwy, 0xff5e3a, 0.18, 800);
  }
}

// ------------------------------------------------------------------ main loop
let lastT = performance.now();
let fps = 60;
let stepsWindow: { t: number; steps: number }[] = [];
function loop(now: number): void {
  requestAnimationFrame(loop);
  const dt = now - lastT;
  lastT = now;
  if (dt > 0 && dt < 1000) fps = fps * 0.95 + (1000 / dt) * 0.05;
  if (!view) return;

  if (!awaiting) {
    const steps = state.paused ? pendingSteps : state.speed;
    pendingSteps = 0;
    frameNo++;
    const transfer = recycle;
    recycle = [];
    send({ type: 'frame', steps, budgetMs: channel?.mode === 'main' ? 10 : 28, colorMode: state.colorMode, pheromone: state.layers.pheromone && frameNo % 2 === 0,
      stats: frameNo % 4 === 0, recycle: transfer }, transfer);
    awaiting = true;
  }
  if (latest) {
    view.upload(latest);
    if (latest.inspect) inspector.update(latest.inspect);
    stepsWindow.push({ t: now, steps: latest.stepsDone });
    recycleFrame(latest);
    latest = null;
  }
  stepsWindow = stepsWindow.filter((s) => now - s.t < 2000);
  const sps = stepsWindow.length > 1
    ? ((stepsWindow[stepsWindow.length - 1].steps - stepsWindow[0].steps) * 1000) / Math.max(1, stepsWindow[stepsWindow.length - 1].t - stepsWindow[0].t)
    : 0;
  view.render(now);
  if (lastStats && frameNo % 4 === 0) {
    hud.update(lastStats, fps, sps, state.paused);
    chart.draw();
    perf.textContent = `${lastStats.population} particles · ${fps.toFixed(0)} fps · ${sps.toFixed(0)} steps/s · ${stepMs.toFixed(1)} ms/step · sim on ${channel?.mode === 'main' ? 'main thread' : 'worker'}`;
  }
  (window as unknown as { __primordial: object }).__primordial = { fps, sps, stepMs, stats: lastStats, view, mode: channel?.mode };
}
const perf = document.getElementById('perf')!;

window.addEventListener('resize', () => view?.resize(window.innerWidth, window.innerHeight));
window.addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'SELECT') return;
  const speed = document.getElementById('speed') as HTMLInputElement;
  switch (e.key) {
    case ' ': state.paused = !state.paused; controls.syncPlay(); e.preventDefault(); break;
    case 'ArrowRight': case 'ArrowUp': speed.value = String(Math.min(24, state.speed * 2)); speed.dispatchEvent(new Event('input')); break;
    case 'ArrowLeft': case 'ArrowDown': speed.value = String(Math.max(1, Math.floor(state.speed / 2))); speed.dispatchEvent(new Event('input')); break;
    case '1': selectSegment('colorMode', 'lineage'); break;
    case '2': selectSegment('colorMode', 'species'); break;
    case '3': selectSegment('colorMode', 'type'); break;
    case 't': case 'T': (document.querySelector('[data-l="trails"]') as HTMLButtonElement).click(); break;
    case 'p': case 'P': (document.querySelector('[data-l="pheromone"]') as HTMLButtonElement).click(); break;
    case 'f': case 'F': view?.cam.fit(); break;
    case 'Escape': inspector.onClose(); break;
  }
});

createChannel(params.get('thread')).then((ch) => {
  channel = ch;
  ch.onmessage = onMessage;
  if (ch.mode === 'main' && !params.get('thread')) toast('Worker threads are throttled on this device: simulating on the main thread.');
  send({ type: 'init', seed, scale });
  requestAnimationFrame(loop);
});
