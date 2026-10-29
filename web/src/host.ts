/**
 * Simulation host: owns the World, steps it on request and replies with compact render
 * buffers (transferred, and recycled by the main thread to avoid per-frame allocation).
 * Runs inside the Web Worker (worker.ts) or, as a fallback, on the main thread.
 */
import { scaledConfig, sizeOf, speedOf, senseOf, type LiveParam } from './sim/config';
import { World } from './sim/world';
import { computeStats } from './sim/stats';
import { N_INPUTS, N_OUTPUTS, N_GROUPS } from './sim/genome';
import { wrapDelta } from './sim/grid';
import { P_STRIDE, type ColorMode, type FrameMsg, type Inspect, type ToMain, type ToWorker } from './protocol';
import { bench } from './bench';
import { lineageColor, SPECIES_PALETTE, TYPE_PALETTE } from './palette';

let world: World | null = null;
let scale = 12;
let seed = 1;
let selected = -1;
let selectedUid = -1;
let pendingEvent: string | null = null;
const pool: ArrayBuffer[] = [];

let postFn: (msg: ToMain, transfer: Transferable[]) => void = () => {};

/** Where replies go (the worker scope, or a main-thread callback). */
export function setPost(fn: (msg: ToMain, transfer: Transferable[]) => void): void {
  postFn = fn;
}

function post(msg: ToMain, transfer: Transferable[] = []): void {
  postFn(msg, transfer);
}

function make(s: number, sc: number): void {
  seed = s; scale = sc;
  world = new World(scaledConfig(sc, { seed: s }));
  selected = -1; selectedUid = -1;
  const c = world.cfg;
  post({ type: 'ready', worldSize: c.worldSize, capacity: c.capacity, pheromoneRes: c.pheromoneRes,
    foodCapacity: c.foodCapacity, patchRadius: c.patchRadius, seed, scale });
}

/** A Float32Array of at least `n` floats, reusing a recycled buffer when one is big enough. */
function f32(n: number): Float32Array {
  const need = n * 4;
  for (let i = 0; i < pool.length; i++) {
    if (pool[i].byteLength >= need) {
      const b = pool.splice(i, 1)[0];
      return new Float32Array(b, 0, n);
    }
  }
  return new Float32Array(Math.max(n, 64));
}

function u8(n: number): Uint8Array {
  for (let i = 0; i < pool.length; i++) {
    if (pool[i].byteLength === n) return new Uint8Array(pool.splice(i, 1)[0]);
  }
  return new Uint8Array(n);
}

function frame(steps: number, budgetMs: number, mode: ColorMode, wantPh: boolean, wantStats: boolean): void {
  const w = world!;
  const t0 = performance.now();
  let done = 0;
  // Run up to `steps` steps, but stop early once the frame budget is spent (always at least one).
  while (done < steps) {
    w.step();
    done++;
    if (performance.now() - t0 > budgetMs) break;
  }
  const stepMs = done > 0 ? (performance.now() - t0) / done : 0;
  const cfg = w.cfg, g = w.genome, W = cfg.worldSize, mb = cfg.maxBonds;
  const stats = wantStats ? computeStats(w) : null;
  const n = w.collectAlive(), list = w.aliveList;

  const particles = f32(n * P_STRIDE);
  let nBonds = 0;
  for (let l = 0; l < n; l++) {
    const s = list[l];
    for (let b = 0; b < mb; b++) if (w.bonds[s * mb + b] > s) nBonds++;
  }
  const bonds = f32(Math.max(1, nBonds) * 4);
  let bi = 0;
  const rgb: [number, number, number] = [0, 0, 0];
  for (let l = 0; l < n; l++) {
    const s = list[l], o = l * P_STRIDE;
    particles[o] = w.posX[s]; particles[o + 1] = w.posY[s];
    particles[o + 2] = sizeOf(cfg, g.body[s * 3]);
    if (mode === 'lineage') lineageColor(g.color[s * 3], g.color[s * 3 + 1], g.color[s * 3 + 2], rgb);
    else {
      const pal = mode === 'type' ? TYPE_PALETTE : SPECIES_PALETTE;
      const id = mode === 'type' ? g.ptype[s] : w.speciesId[s];
      const c = id >= 0 ? pal[id % pal.length] : [0.62, 0.62, 0.6];
      rgb[0] = c[0]; rgb[1] = c[1]; rgb[2] = c[2];
    }
    particles[o + 3] = rgb[0]; particles[o + 4] = rgb[1]; particles[o + 5] = rgb[2];
    particles[o + 6] = w.signal[s];
    particles[o + 7] = w.aggression[s];
    for (let b = 0; b < mb; b++) {
      const p = w.bonds[s * mb + b];
      if (p <= s) continue;
      bonds[bi++] = w.posX[s]; bonds[bi++] = w.posY[s];
      bonds[bi++] = w.posX[s] + wrapDelta(w.posX[p] - w.posX[s], W);
      bonds[bi++] = w.posY[s] + wrapDelta(w.posY[p] - w.posY[s], W);
    }
  }

  let nFood = 0;
  for (let i = 0; i < cfg.foodCapacity; i++) nFood += w.foodAlive[i];
  const food = f32(Math.max(1, nFood) * 2);
  let fi = 0;
  for (let i = 0; i < cfg.foodCapacity; i++) {
    if (!w.foodAlive[i]) continue;
    food[fi++] = w.foodX[i]; food[fi++] = w.foodY[i];
  }
  const patches = new Float32Array(cfg.foodPatches * 2);
  for (let i = 0; i < cfg.foodPatches; i++) { patches[i * 2] = w.patchX[i]; patches[i * 2 + 1] = w.patchY[i]; }

  let pheromone: Uint8Array | null = null;
  if (wantPh) {
    const r = cfg.pheromoneRes, rr = r * r, ph = w.pheromone;
    pheromone = u8(rr * 4);
    for (let cx = 0; cx < r; cx++) {
      for (let cy = 0; cy < r; cy++) {
        const t = (cy * r + cx) * 4;
        pheromone[t] = Math.min(255, Math.tanh(Math.sqrt(ph[cx * r + cy]) * 1.8) * 255);
        pheromone[t + 1] = Math.min(255, Math.tanh(Math.sqrt(ph[rr + cx * r + cy]) * 1.8) * 255);
        pheromone[t + 2] = 0;
        pheromone[t + 3] = 255;
      }
    }
  }

  const inspect = inspectSelected(w);
  const msg: FrameMsg = {
    type: 'frame', n, particles, nBonds, bonds, nFood, food, patches, pheromone, stats, stepMs,
    stepsDone: w.stepCount, inspect, event: pendingEvent,
  };
  pendingEvent = null;
  const transfer: Transferable[] = [particles.buffer, bonds.buffer, food.buffer, patches.buffer];
  if (pheromone) transfer.push(pheromone.buffer);
  post(msg, transfer);
}

function inspectSelected(w: World): Inspect | null {
  if (selected < 0) return null;
  const s = selected;
  if (!w.alive[s] || w.uid[s] !== selectedUid) { selected = -1; pendingEvent = 'The inspected particle has died.'; return null; }
  const cfg = w.cfg, g = w.genome, H = cfg.hidden, mb = cfg.maxBonds;
  const size = sizeOf(cfg, g.body[s * 3]);
  const w2: number[] = [];
  for (let k = 0; k < H * N_OUTPUTS; k++) w2.push(g.w2[s * g.sw2 + k] + w.plastic[s * g.sw2 + k]);
  let bonds = 0;
  for (let b = 0; b < mb; b++) if (w.bonds[s * mb + b] >= 0) bonds++;
  const sid = w.speciesId[s];
  return {
    slot: s, uid: w.uid[s], x: w.posX[s], y: w.posY[s],
    energy: w.energy[s], threshold: cfg.reproduceThreshold * size,
    age: w.age[s], maxAge: cfg.maxAge, generation: w.generation[s],
    size, speed: speedOf(cfg, g.body[s * 3 + 1]), sense: senseOf(cfg, g.body[s * 3 + 2]),
    ptype: g.ptype[s],
    color: [g.color[s * 3], g.color[s * 3 + 1], g.color[s * 3 + 2]],
    species: sid >= 0 ? (w.species.names[sid] ?? '?') : 'not yet classified',
    bonds, aggression: w.aggression[s], signal: w.signal[s], logEta: g.logEta[s],
    sigma: Array.from(g.sigma.subarray(s * N_GROUPS, (s + 1) * N_GROUPS)),
    foodGained: w.foodGained[s], preyGained: w.preyGained[s],
    inputs: Array.from(w.lastInputs.subarray(s * N_INPUTS, (s + 1) * N_INPUTS)),
    hidden: Array.from(w.lastHidden.subarray(s * H, (s + 1) * H)),
    outputs: Array.from(w.lastOutputs.subarray(s * N_OUTPUTS, (s + 1) * N_OUTPUTS)),
    w1: Array.from(g.w1.subarray(s * g.sw1, (s + 1) * g.sw1)),
    w2,
    attraction: Array.from(g.attraction.subarray(s * cfg.nTypes, (s + 1) * cfg.nTypes)),
  };
}

function select(x: number, y: number, radius: number): void {
  const w = world!;
  const W = w.cfg.worldSize;
  let best = radius * radius, bs = -1;
  for (let s = 0; s < w.C; s++) {
    if (!w.alive[s]) continue;
    const dx = wrapDelta(w.posX[s] - x, W), dy = wrapDelta(w.posY[s] - y, W);
    const d2 = dx * dx + dy * dy;
    if (d2 < best) { best = d2; bs = s; }
  }
  selected = bs;
  selectedUid = bs >= 0 ? w.uid[bs] : -1;
}

export function handle(m: ToWorker): void {
  switch (m.type) {
    case 'init':
    case 'reset':
      make(m.seed, m.scale);
      break;
    case 'frame':
      for (const b of m.recycle) if (pool.length < 16) pool.push(b);
      if (world) frame(m.steps, m.budgetMs, m.colorMode, m.pheromone, m.stats);
      break;
    case 'select':
      if (world) select(m.x, m.y, m.radius);
      break;
    case 'bench':
      post({ type: 'bench', mmacs: bench() });
      break;
    case 'deselect':
      selected = -1;
      break;
    case 'tool':
      if (!world) break;
      if (m.tool === 'food') world.dropFood(m.x, m.y, 40, 0.04);
      else pendingEvent = `Removed ${world.meteor(m.x, m.y, 0.15)} particles; the area was refilled with food.`;
      break;
    case 'param':
      if (world) (world.cfg as unknown as Record<LiveParam, number>)[m.key] = m.value;
      break;
  }
}
