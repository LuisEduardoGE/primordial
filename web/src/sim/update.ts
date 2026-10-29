/**
 * World._update_living from primordial/world.py, as plain loops over local indices.
 *
 * Semantics match the Python: every sense, force, predation and bond decision is computed
 * from the state at the start of the step (pre-move positions, last step's signals);
 * eating and scent deposits use the post-move positions; results are written back at the end.
 */
import type { Config } from './config';
import { sizeOf, speedOf, senseOf } from './config';
import { NeighbourList, selfNeighbours, nearest, nearestHit, wrapDelta } from './grid';
import { N_INPUTS, N_MEMORY, N_OUTPUTS, N_SENSES, think } from './genome';
import type { World } from './world';
import { act } from './act';

export class Scratch {
  local: Int32Array;
  dead: Int32Array;
  px: Float32Array; py: Float32Array; nx: Float32Array; ny: Float32Array; vx: Float32Array; vy: Float32Array;
  size: Float32Array; speed: Float32Array; sense: Float32Array;
  e: Float32Array; e0: Float32Array; eTmp: Float32Array;
  demanded: Float32Array; lost: Float32Array; gained: Float32Array;
  inputs: Float32Array; hidden: Float32Array; out: Float32Array;
  pl: Int32Array; bdx: Float32Array; bdy: Float32Array; bdist: Float32Array; bondCount: Int32Array;
  freePre: Uint8Array;
  sim = new Float32Array(1024);
  // compact food view
  fIdx: Int32Array; fx: Float32Array; fy: Float32Array;
  fBest: Float32Array; fWin: Int32Array; fTouched: Int32Array;
  // predation pairs
  pa = new Int32Array(256); pb = new Int32Array(256); pd = new Float32Array(256);

  constructor(cfg: Config) {
    const C = cfg.capacity, H = cfg.hidden, mb = cfg.maxBonds, F = cfg.foodCapacity;
    const f = (n: number) => new Float32Array(n);
    this.local = new Int32Array(C).fill(-1);
    this.dead = new Int32Array(C);
    this.px = f(C); this.py = f(C); this.nx = f(C); this.ny = f(C); this.vx = f(C); this.vy = f(C);
    this.size = f(C); this.speed = f(C); this.sense = f(C);
    this.e = f(C); this.e0 = f(C); this.eTmp = f(C);
    this.demanded = f(C); this.lost = f(C); this.gained = f(C);
    this.inputs = f(C * N_INPUTS); this.hidden = f(C * H); this.out = f(C * N_OUTPUTS);
    this.pl = new Int32Array(C * mb); this.bdx = f(C * mb); this.bdy = f(C * mb); this.bdist = f(C * mb);
    this.bondCount = new Int32Array(C);
    this.freePre = new Uint8Array(C);
    this.fIdx = new Int32Array(F); this.fx = f(F); this.fy = f(F);
    this.fBest = f(F).fill(Infinity); this.fWin = new Int32Array(F).fill(-1); this.fTouched = new Int32Array(F);
  }

  ensureSim(n: number): void {
    if (this.sim.length < n) this.sim = new Float32Array(n * 2);
  }

  ensurePairs(n: number): void {
    if (this.pa.length >= n) return;
    const m = n * 2;
    const a = new Int32Array(m); a.set(this.pa); this.pa = a;
    const b = new Int32Array(m); b.set(this.pb); this.pb = b;
    const d = new Float32Array(m); d.set(this.pd); this.pd = d;
  }
}

/** Build the compact list of living food and its grid. Returns the count. */
function prepareFood(w: World): number {
  const sc = w.scratch, F = w.cfg.foodCapacity;
  let nf = 0;
  for (let i = 0; i < F; i++) {
    if (!w.foodAlive[i]) continue;
    sc.fIdx[nf] = i; sc.fx[nf] = w.foodX[i]; sc.fy[nf] = w.foodY[i]; nf++;
  }
  w.foodGrid.build(sc.fx, sc.fy, nf, w.cfg.worldSize, 0.05);
  return nf;
}

/** Steps 1-8 for all `n` living particles (w.aliveList[0..n)). */
export function updateLiving(w: World, n: number): void {
  const cfg = w.cfg, sc = w.scratch, g = w.genome, list = w.aliveList;
  const W = cfg.worldSize, mb = cfg.maxBonds, H = cfg.hidden;
  const { px, py, size, speed, sense, local } = sc;

  local.fill(-1);
  for (let l = 0; l < n; l++) {
    const s = list[l];
    local[s] = l;
    px[l] = w.posX[s]; py[l] = w.posY[s];
    size[l] = sizeOf(cfg, g.body[s * 3]);
    speed[l] = speedOf(cfg, g.body[s * 3 + 1]);
    sense[l] = senseOf(cfg, g.body[s * 3 + 2]);
  }

  // Neighbourhood (pre-move positions).
  const radius = Math.max(cfg.interactionRadius, cfg.senseRadius * cfg.senseMax);
  w.grid.build(px, py, n, W, radius);
  const nb: NeighbourList = w.nb;
  selfNeighbours(w.grid, radius, nb);
  sc.ensureSim(nb.len);
  const sim = sc.sim, inv2 = 1 / (cfg.kinSigma * cfg.kinSigma), col = g.color;
  for (let l = 0; l < n; l++) {
    const s = list[l];
    const r = col[s * 3], gg = col[s * 3 + 1], b = col[s * 3 + 2];
    for (let k = nb.start[l]; k < nb.start[l + 1]; k++) {
      const t = list[nb.j[k]];
      const dr = r - col[t * 3], dg = gg - col[t * 3 + 1], db = b - col[t * 3 + 2];
      sim[k] = Math.exp(-(dr * dr + dg * dg + db * db) * inv2);
    }
  }

  // Bonded partners.
  for (let l = 0; l < n; l++) {
    const s = list[l];
    let cnt = 0, free = 0;
    for (let b = 0; b < mb; b++) {
      const p = w.bonds[s * mb + b];
      const pl = p >= 0 ? local[p] : -1;
      if (p < 0) free = 1;
      sc.pl[l * mb + b] = pl;
      if (pl >= 0) {
        const dx = wrapDelta(px[pl] - px[l], W), dy = wrapDelta(py[pl] - py[l], W);
        sc.bdx[l * mb + b] = dx; sc.bdy[l * mb + b] = dy; sc.bdist[l * mb + b] = Math.sqrt(dx * dx + dy * dy);
        cnt++;
      }
    }
    sc.bondCount[l] = cnt;
    sc.freePre[l] = free;
  }

  const nf = prepareFood(w);
  senseAll(w, n, nf);
  for (let l = 0; l < n; l++) {
    think(g, list[l], w.plastic, sc.inputs, l * N_INPUTS, sc.hidden, l * H, sc.out, l * N_OUTPUTS);
  }
  act(w, n, nf);
}

/** Step 1: the (n, N_SENSES) sensory vectors plus memory, into sc.inputs. */
function senseAll(w: World, n: number, nf: number): void {
  const cfg = w.cfg, sc = w.scratch, list = w.aliveList, nb = w.nb, sim = sc.sim;
  const W = cfg.worldSize, r = cfg.pheromoneRes, ph = w.pheromone, rr = r * r;
  const season = w.season();
  for (let l = 0; l < n; l++) {
    const s = list[l];
    const o = l * N_INPUTS, inp = sc.inputs;
    // [0:3] nearest food
    let fdx = 0, fdy = 0, close = 0;
    if (nf > 0) {
      const fr = cfg.foodSenseRadius * sc.sense[l];
      if (nearest(w.foodGrid, sc.px[l], sc.py[l], fr)) {
        const d = nearestHit.d;
        close = Math.max(0, 1 - d / fr);
        const id = 1 / Math.max(d, 1e-9);
        fdx = nearestHit.dx * id * close; fdy = nearestHit.dy * id * close;
      }
    }
    // [3:7] kin / strangers, [7] crowding, [11:14] signals, [21] threat
    const sr = cfg.senseRadius * sc.sense[l], sz = sc.size[l];
    let cnt = 0, kx = 0, ky = 0, sx = 0, sy = 0, sigSum = 0, proxSum = 0, sgx = 0, sgy = 0, strW = 0, threat = 0;
    for (let k = nb.start[l]; k < nb.start[l + 1]; k++) {
      const d = nb.dist[k];
      if (!(d < sr)) continue;
      const j = nb.j[k];
      const id = 1 / Math.max(d, 1e-9);
      const ux = nb.dx[k] * id, uy = nb.dy[k] * id;
      const si = sim[k], st = 1 - si;
      cnt++;
      kx += si * ux; ky += si * uy;
      sx += st * ux; sy += st * uy;
      const prox = 1 - d / sr;
      const sg = w.signal[list[j]] * prox;
      sigSum += sg; proxSum += prox; sgx += sg * ux; sgy += sg * uy;
      strW += st;
      threat += st * Math.log(sc.size[j] / sz);
    }
    const c1 = Math.max(cnt, 1);
    inp[o] = fdx; inp[o + 1] = fdy; inp[o + 2] = close;
    inp[o + 3] = kx / c1; inp[o + 4] = ky / c1; inp[o + 5] = sx / c1; inp[o + 6] = sy / c1;
    inp[o + 7] = cnt / 10;
    inp[o + 8] = w.energy[s] / (cfg.reproduceThreshold * sz);
    inp[o + 9] = w.velX[s] * 5; inp[o + 10] = w.velY[s] * 5;
    inp[o + 11] = sigSum / Math.max(proxSum, 1e-6);
    inp[o + 12] = sgx / c1; inp[o + 13] = sgy / c1;
    // [14:20] scent channels: gradient x, gradient y, level
    let cx = Math.floor((sc.px[l] / W) * r), cy = Math.floor((sc.py[l] / W) * r);
    cx = cx < 0 ? 0 : cx >= r ? r - 1 : cx;
    cy = cy < 0 ? 0 : cy >= r ? r - 1 : cy;
    const xp = (cx + 1) % r, xm = (cx - 1 + r) % r, yp = (cy + 1) % r, ym = (cy - 1 + r) % r;
    for (let c = 0; c < 2; c++) {
      const b = c * rr;
      const gx = ph[b + xp * r + cy] - ph[b + xm * r + cy];
      const gy = ph[b + cx * r + yp] - ph[b + cx * r + ym];
      inp[o + 14 + c * 3] = Math.tanh(gx * 5);
      inp[o + 15 + c * 3] = Math.tanh(gy * 5);
      inp[o + 16 + c * 3] = Math.tanh(ph[b + cx * r + cy]);
    }
    inp[o + 20] = sc.bondCount[l] / cfg.maxBonds;
    inp[o + 21] = threat / Math.max(strW, 1e-6);
    inp[o + 22] = season;
    for (let m = 0; m < N_MEMORY; m++) inp[o + N_SENSES + m] = w.memory[s * N_MEMORY + m];
  }
}
