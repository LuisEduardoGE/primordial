/**
 * Behavioural species (primordial/species.py): cluster particles on body genes + slow
 * behaviour averages + diet with warm-started k-means, then name each cluster.
 */
import { Rng } from './rng';
import type { World } from './world';
import { N_EMA } from './world';

export const FEATURES = ['size', 'speed', 'sense', 'moving', 'attacking', 'signalling', 'scenting', 'bonds', 'predation'];
const F = FEATURES.length;

export function nameCluster(c: Record<string, number>): string {
  if (c.predation > 0.35) return 'hunter';
  if (c.bonds > 1.0) return 'colonial';
  if (c.moving < 0.15) return 'drifter';
  if (c.speed > 0.6 && c.sense > 0.6) return 'scout';
  if (c.signalling > 0.35) return 'signaller';
  if (c.scenting > 0.35) return 'trail-layer';
  return 'grazer';
}

export function uniqueNames(names: string[]): string[] {
  const seen: Record<string, number> = {};
  return names.map((n) => {
    seen[n] = (seen[n] ?? 0) + 1;
    return seen[n] === 1 ? n : `${n} ${seen[n]}`;
  });
}

export class SpeciesTracker {
  k: number;
  iters: number;
  rng: Rng;
  centroids: Float64Array | null = null;   // raw feature space, k * F
  names: string[] = [];
  private x = new Float64Array(0);
  private labels = new Int32Array(0);

  constructor(k = 5, seed = 0, iters = 15) {
    this.k = k;
    this.iters = iters;
    this.rng = new Rng(seed + 12345);
  }

  classify(w: World): void {
    const n = w.collectAlive();
    w.speciesId.fill(-1);
    if (n < this.k) return;
    if (this.x.length < n * F) { this.x = new Float64Array(n * F * 2); this.labels = new Int32Array(n * 2); }
    const x = this.x, list = w.aliveList, g = w.genome;
    for (let l = 0; l < n; l++) {
      const s = list[l], o = l * F;
      x[o] = g.body[s * 3]; x[o + 1] = g.body[s * 3 + 1]; x[o + 2] = g.body[s * 3 + 2];
      for (let e = 0; e < N_EMA; e++) x[o + 3 + e] = w.ema[s * N_EMA + e];
      x[o + 8] = w.preyGained[s] / (w.preyGained[s] + w.foodGained[s] + 1e-6);
    }
    this.update(x, n, this.labels);
    for (let l = 0; l < n; l++) w.speciesId[list[l]] = this.labels[l];
  }

  /** x (n*F raw features, overwritten with z-scores) -> labels. Updates centroids + names. */
  update(x: Float64Array, n: number, labels: Int32Array): void {
    const k = Math.min(this.k, n);
    const mu = new Float64Array(F), sd = new Float64Array(F).fill(1);
    for (let i = 0; i < n; i++) for (let f = 0; f < F; f++) mu[f] += x[i * F + f];
    for (let f = 0; f < F; f++) mu[f] /= n;
    if (n > 1) {
      sd.fill(0);
      for (let i = 0; i < n; i++) for (let f = 0; f < F; f++) { const d = x[i * F + f] - mu[f]; sd[f] += d * d; }
      for (let f = 0; f < F; f++) sd[f] = Math.max(Math.sqrt(sd[f] / (n - 1)), 1e-3);
    }
    for (let i = 0; i < n; i++) for (let f = 0; f < F; f++) x[i * F + f] = (x[i * F + f] - mu[f]) / sd[f];

    let c: Float64Array;
    if (!this.centroids || this.centroids.length !== k * F) c = this.kmeansPP(x, n, k);
    else {
      c = new Float64Array(k * F);
      for (let j = 0; j < k; j++) for (let f = 0; f < F; f++) c[j * F + f] = (this.centroids[j * F + f] - mu[f]) / sd[f];
    }
    const sums = new Float64Array(k * F), counts = new Int32Array(k);
    for (let it = 0; it < this.iters; it++) {
      assign(x, n, c, k, labels);
      sums.fill(0); counts.fill(0);
      for (let i = 0; i < n; i++) {
        const j = labels[i];
        counts[j]++;
        for (let f = 0; f < F; f++) sums[j * F + f] += x[i * F + f];
      }
      for (let j = 0; j < k; j++) {
        if (counts[j] > 0) for (let f = 0; f < F; f++) c[j * F + f] = sums[j * F + f] / counts[j];
        else {   // empty cluster: re-seed at the worst-served point
          let worst = -1, wi = 0;
          for (let i = 0; i < n; i++) { const d = minDist(x, i, c, k); if (d > worst) { worst = d; wi = i; } }
          for (let f = 0; f < F; f++) c[j * F + f] = x[wi * F + f];
        }
      }
    }
    assign(x, n, c, k, labels);
    const raw = new Float64Array(k * F);
    for (let j = 0; j < k; j++) for (let f = 0; f < F; f++) raw[j * F + f] = c[j * F + f] * sd[f] + mu[f];
    this.centroids = raw;
    const names: string[] = [];
    for (let j = 0; j < k; j++) {
      const row: Record<string, number> = {};
      FEATURES.forEach((name, f) => (row[name] = raw[j * F + f]));
      names.push(nameCluster(row));
    }
    this.names = uniqueNames(names);
  }

  private kmeansPP(z: Float64Array, n: number, k: number): Float64Array {
    const c = new Float64Array(k * F);
    const first = this.rng.int(n);
    for (let f = 0; f < F; f++) c[f] = z[first * F + f];
    const d2 = new Float64Array(n);
    for (let m = 1; m < k; m++) {
      let total = 0;
      for (let i = 0; i < n; i++) { const d = minDist(z, i, c, m); d2[i] = d * d; total += d2[i]; }
      let pick = n - 1;
      if (total > 0) {
        let r = this.rng.rand() * total;
        for (let i = 0; i < n; i++) { r -= d2[i]; if (r <= 0) { pick = i; break; } }
      } else pick = this.rng.int(n);
      for (let f = 0; f < F; f++) c[m * F + f] = z[pick * F + f];
    }
    return c;
  }
}

function dist(x: Float64Array, i: number, c: Float64Array, j: number): number {
  let s = 0;
  for (let f = 0; f < F; f++) { const d = x[i * F + f] - c[j * F + f]; s += d * d; }
  return Math.sqrt(s);
}

function minDist(x: Float64Array, i: number, c: Float64Array, k: number): number {
  let best = Infinity;
  for (let j = 0; j < k; j++) best = Math.min(best, dist(x, i, c, j));
  return best;
}

function assign(x: Float64Array, n: number, c: Float64Array, k: number, labels: Int32Array): void {
  for (let i = 0; i < n; i++) {
    let best = Infinity, bj = 0;
    for (let j = 0; j < k; j++) { const d = dist(x, i, c, j); if (d < best) { best = d; bj = j; } }
    labels[i] = bj;
  }
}
