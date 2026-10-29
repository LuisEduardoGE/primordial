/** Step 8: bond breaking and formation (World._update_bonds). Uses pre-move distances. */
import { N_OUTPUTS, BOND } from './genome';
import type { World } from './world';

export function hasBond(w: World, a: number, b: number): boolean {
  const mb = w.cfg.maxBonds;
  for (let q = 0; q < mb; q++) if (w.bonds[a * mb + q] === b) return true;
  return false;
}

/** Index into w.bonds of a free slot of particle `a`, or -1. */
export function freeSlot(w: World, a: number): number {
  const mb = w.cfg.maxBonds;
  for (let q = 0; q < mb; q++) if (w.bonds[a * mb + q] < 0) return a * mb + q;
  return -1;
}

export function unbond(w: World, a: number, b: number): void {
  const mb = w.cfg.maxBonds;
  for (let q = 0; q < mb; q++) if (w.bonds[a * mb + q] === b) { w.bonds[a * mb + q] = -1; break; }
  for (let q = 0; q < mb; q++) if (w.bonds[b * mb + q] === a) { w.bonds[b * mb + q] = -1; break; }
}

export function updateBonds(w: World, n: number): void {
  const cfg = w.cfg, sc = w.scratch, list = w.aliveList, nb = w.nb, mb = cfg.maxBonds;
  const out = sc.out;
  // Break: overstretched, or either side no longer wants it.
  for (let l = 0; l < n; l++) {
    const il = out[l * N_OUTPUTS + BOND];
    for (let b = 0; b < mb; b++) {
      const p = sc.pl[l * mb + b];
      if (p < 0) continue;
      if (sc.bdist[l * mb + b] > cfg.bondBreak || il < -0.5 || out[p * N_OUTPUTS + BOND] < -0.5) {
        unbond(w, list[l], list[p]);
      }
    }
  }
  // Form: both want it, touching, kin, both had a free slot this step, not already bonded.
  for (let l = 0; l < n; l++) {
    if (!(out[l * N_OUTPUTS + BOND] > 0.5) || !sc.freePre[l]) continue;
    let best = Infinity, bj = -1;
    for (let k = nb.start[l]; k < nb.start[l + 1]; k++) {
      const j = nb.j[k], d = nb.dist[k];
      if (d < best && d < cfg.contactRadius * (sc.size[l] + sc.size[j]) && sc.sim[k] > cfg.kinThreshold
        && out[j * N_OUTPUTS + BOND] > 0.5 && sc.freePre[j]) { best = d; bj = j; }
    }
    if (bj < 0) continue;
    const a = list[l], b = list[bj];
    if (hasBond(w, a, b)) continue;
    const fa = freeSlot(w, a), fb = freeSlot(w, b);
    if (fa < 0 || fb < 0) continue;
    w.bonds[fa] = b; w.bonds[fb] = a;
    w.bondsFormed++;
  }
}
