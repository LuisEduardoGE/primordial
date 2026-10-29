/** Step 10 (World._births): splitting, mate search, crossover, mutation and budding. */
import { sizeOf } from './config';
import type { World } from './world';
import { mod } from './world';
import { freeSlot } from './bonds';

let parents = new Int32Array(0);
let free = new Int32Array(0);
let mates = new Int32Array(0);
let gift = new Float32Array(0);
let share = new Float32Array(0);

export function births(w: World): void {
  const cfg = w.cfg, C = w.C, rng = w.rng, g = w.genome;
  if (parents.length < C) {
    parents = new Int32Array(C); free = new Int32Array(C); mates = new Int32Array(C);
    gift = new Float32Array(C); share = new Float32Array(C);
  }
  let np = 0, nfree = 0;
  for (let s = 0; s < C; s++) {
    if (!w.alive[s]) { free[nfree++] = s; continue; }
    if (w.energy[s] >= cfg.reproduceThreshold * sizeOf(cfg, g.body[s * 3])) parents[np++] = s;
  }
  const k = Math.min(np, nfree);
  if (k === 0) return;
  rng.shuffle(parents, np);

  // Compatible mate near each parent: nearest living kin-enough particle within mate_radius.
  for (let i = 0; i < k; i++) mates[i] = -1;
  if (cfg.mateProb > 0) {
    const n = w.collectAlive();
    const sc = w.scratch, list = w.aliveList;
    for (let l = 0; l < n; l++) { sc.px[l] = w.posX[list[l]]; sc.py[l] = w.posY[list[l]]; }
    w.grid.build(sc.px, sc.py, n, cfg.worldSize, cfg.mateRadius);
    const inv2 = 1 / (cfg.kinSigma * cfg.kinSigma), col = g.color;
    const gr = w.grid;
    for (let i = 0; i < k; i++) {
      const p = parents[i];
      const pr = col[p * 3], pg = col[p * 3 + 1], pb = col[p * 3 + 2];
      let best = Infinity, bm = -1;
      gr.query(w.posX[p], w.posY[p], cfg.mateRadius, (j, _dx, _dy, d2) => {
        const m = list[j];
        if (m === p || !(d2 < best) || d2 >= cfg.mateRadius * cfg.mateRadius) return;
        const dr = pr - col[m * 3], dg = pg - col[m * 3 + 1], db = pb - col[m * 3 + 2];
        if (Math.exp(-(dr * dr + dg * dg + db * db) * inv2) > cfg.mateSimilarity) { best = d2; bm = m; }
      });
      if (rng.rand() < cfg.mateProb && bm >= 0) mates[i] = bm;
    }
  }

  for (let i = 0; i < k; i++) share[i] = (w.energy[parents[i]] * (1 - cfg.birthCost)) / 2;
  for (let i = 0; i < k; i++) gift[i] = mates[i] >= 0 ? w.energy[mates[i]] * cfg.mateContribution : 0;
  for (let i = 0; i < k; i++) if (mates[i] >= 0) w.energy[mates[i]] -= gift[i];

  for (let i = 0; i < k; i++) {
    const p = parents[i], c = free[i], m = mates[i];
    w.resetSlot(c);
    w.energy[p] = share[i];
    w.energy[c] = share[i] + gift[i];
    w.posX[c] = mod(w.posX[p] + rng.randn() * 0.01, cfg.worldSize);
    w.posY[c] = mod(w.posY[p] + rng.randn() * 0.01, cfg.worldSize);
    w.generation[c] = Math.max(w.generation[p], m >= 0 ? w.generation[m] : 0) + 1;
    w.alive[c] = 1;
    const bud = rng.rand() < g.bud[p];
    g.copy(c, p);
    if (m >= 0) g.crossover(c, m, rng);
    g.mutate(c, cfg, rng);
    w.uid[c] = w.nextUid++;
    // Budding: the child stays attached to its parent.
    if (bud) {
      const fp = freeSlot(w, p);
      if (fp >= 0) {
        w.bonds[fp] = c;
        w.bonds[c * cfg.maxBonds] = p;
        w.bondsFormed++;
      }
    }
  }
  w.births += k;
  for (let i = 0; i < k; i++) if (mates[i] >= 0) w.matedBirths++;
  w.population += k;
}
