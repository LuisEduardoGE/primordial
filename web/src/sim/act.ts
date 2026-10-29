/**
 * Steps 3-8 of World._update_living: physics, metabolism, energy sharing, predation,
 * eating, lifetime learning, scent deposit, write-back and bond updates.
 */
import { ftanh, N_INPUTS, N_MEMORY, N_OUTPUTS, THRUST_X, THRUST_Y, AGGRESSION, SIGNAL, DEPOSIT_A, DEPOSIT_B, MEMORY0 } from './genome';
import type { World } from './world';
import { N_EMA, mod } from './world';
import { updateBonds } from './bonds';

export function particleLifeForce(r: number, a: number, beta: number): number {
  if (r < beta) return r / beta - 1;
  if (r < 1) return a * (1 - Math.abs(2 * r - 1 - beta) / (1 - beta));
  return 0;
}

export function act(w: World, n: number, nf: number): void {
  const cfg = w.cfg, sc = w.scratch, g = w.genome, list = w.aliveList, nb = w.nb;
  const W = cfg.worldSize, mb = cfg.maxBonds, H = cfg.hidden, T = cfg.nTypes;
  const { px, py, nx, ny, vx, vy, size, speed, out, e, e0 } = sc;
  const ir = cfg.interactionRadius, beta = cfg.repulsionBeta, kf = ir * cfg.forceFactor;
  const friction = Math.pow(0.5, cfg.dt / cfg.frictionHalfLife);
  let anyBond = false;

  // 3. physics + 4. metabolism
  for (let l = 0; l < n; l++) {
    const s = list[l], o = l * N_OUTPUTS;
    let fx = 0, fy = 0;
    const arow = s * T;
    for (let k = nb.start[l]; k < nb.start[l + 1]; k++) {
      const d = nb.dist[k];
      if (!(d < ir)) continue;
      const a = g.attraction[arow + g.ptype[list[nb.j[k]]]];
      const f = particleLifeForce(d / ir, a, beta);
      const id = 1 / Math.max(d, 1e-9);
      fx += f * nb.dx[k] * id; fy += f * nb.dy[k] * id;
    }
    let ax = fx * kf, ay = fy * kf;
    for (let b = 0; b < mb; b++) {
      const p = sc.pl[l * mb + b];
      if (p < 0) continue;
      anyBond = true;
      const bd = sc.bdist[l * mb + b];
      const rest = (cfg.bondRest * (size[l] + size[p])) / 2;
      const spring = (cfg.bondStrength * (bd - rest)) / Math.max(bd, 1e-9);
      ax += spring * sc.bdx[l * mb + b]; ay += spring * sc.bdy[l * mb + b];
    }
    const tx = out[o + THRUST_X], ty = out[o + THRUST_Y];
    const th = cfg.maxThrust * speed[l];
    ax += (tx * th) / size[l]; ay += (ty * th) / size[l];
    vx[l] = w.velX[s] * friction + ax * cfg.dt;
    vy[l] = w.velY[s] * friction + ay * cfg.dt;
    nx[l] = mod(px[l] + vx[l] * cfg.dt, W);
    ny[l] = mod(py[l] + vy[l] * cfg.dt, W);

    const aggr = Math.max(out[o + AGGRESSION], 0);
    const dep = Math.max(out[o + DEPOSIT_A], 0) + Math.max(out[o + DEPOSIT_B], 0);
    e0[l] = w.energy[s];
    e[l] = e0[l] - metabolicCost(w, l, tx, ty, aggr, out[o + SIGNAL], dep, sc.bondCount[l]);
  }

  // Bonded organisms share energy toward the poorer side.
  if (anyBond) {
    sc.eTmp.set(e.subarray(0, n));
    for (let l = 0; l < n; l++) {
      let flow = 0;
      for (let b = 0; b < mb; b++) {
        const p = sc.pl[l * mb + b];
        if (p >= 0) flow += (cfg.bondShare * (sc.eTmp[p] - sc.eTmp[l])) / 2;
      }
      e[l] += flow;
    }
  }

  predation(w, n);
  if (nf > 0) eat(w, n);

  // 7. learning, scent deposit, write-back
  const r = cfg.pheromoneRes, rr = r * r;
  const ph = w.pheromone;
  for (let l = 0; l < n; l++) {
    const s = list[l], o = l * N_OUTPUTS, ho = l * H;
    const dE = e[l] - e0[l];
    if (cfg.plasticity > 0) {
      const m = ftanh(cfg.rewardScale * (dE - w.rewardAvg[s]));
      const eta = Math.pow(10, g.logEta[s]) * cfg.plasticity * m;
      const keep = 1 - cfg.plasticDecay, clip = cfg.plasticClip;
      const po = s * H * N_OUTPUTS;
      for (let h = 0; h < H; h++) {
        const hv = eta * sc.hidden[ho + h];
        const row = po + h * N_OUTPUTS;
        for (let q = 0; q < N_OUTPUTS; q++) {
          let v = w.plastic[row + q] * keep + hv * out[o + q];
          if (v > clip) v = clip; else if (v < -clip) v = -clip;
          w.plastic[row + q] = v;
        }
      }
    }
    w.rewardAvg[s] = 0.95 * w.rewardAvg[s] + 0.05 * dE;

    const da = Math.max(out[o + DEPOSIT_A], 0), db = Math.max(out[o + DEPOSIT_B], 0);
    if (da > 0 || db > 0) {
      let cx = Math.floor((nx[l] / W) * r), cy = Math.floor((ny[l] / W) * r);
      cx = cx < 0 ? 0 : cx >= r ? r - 1 : cx;
      cy = cy < 0 ? 0 : cy >= r ? r - 1 : cy;
      ph[cx * r + cy] += da * cfg.pheromoneDeposit;
      ph[rr + cx * r + cy] += db * cfg.pheromoneDeposit;
      w.pheromoneActive = true;
    }

    w.posX[s] = nx[l]; w.posY[s] = ny[l]; w.velX[s] = vx[l]; w.velY[s] = vy[l]; w.energy[s] = e[l];
    const aggr = Math.max(out[o + AGGRESSION], 0), sig = out[o + SIGNAL];
    w.aggression[s] = aggr; w.signal[s] = sig;
    w.memory[s * N_MEMORY] = out[o + MEMORY0]; w.memory[s * N_MEMORY + 1] = out[o + MEMORY0 + 1];
    for (let q = 0; q < N_INPUTS; q++) w.lastInputs[s * N_INPUTS + q] = sc.inputs[l * N_INPUTS + q];
    for (let q = 0; q < H; q++) w.lastHidden[s * H + q] = sc.hidden[ho + q];
    for (let q = 0; q < N_OUTPUTS; q++) w.lastOutputs[s * N_OUTPUTS + q] = out[o + q];
    const eo = s * N_EMA;
    const beh0 = Math.hypot(out[o + THRUST_X], out[o + THRUST_Y]);
    w.ema[eo] = 0.98 * w.ema[eo] + 0.02 * beh0;
    w.ema[eo + 1] = 0.98 * w.ema[eo + 1] + 0.02 * aggr;
    w.ema[eo + 2] = 0.98 * w.ema[eo + 2] + 0.02 * Math.abs(sig);
    w.ema[eo + 3] = 0.98 * w.ema[eo + 3] + 0.02 * (da + db);
    w.ema[eo + 4] = 0.98 * w.ema[eo + 4] + 0.02 * sc.bondCount[l];
    w.age[s] += 1;
  }

  // 8. bonds
  updateBonds(w, n);
}

/** Per-step energy cost (step 4) for local particle `l` given its actions. */
export function metabolicCost(w: World, l: number, tx: number, ty: number, aggr: number,
  signal: number, deposit: number, bonds: number): number {
  const cfg = w.cfg, sc = w.scratch;
  const size = sc.size[l], speed = sc.speed[l];
  return cfg.baseMetabolism * Math.pow(size, 0.75)
    + cfg.speedUpkeep * speed
    + cfg.senseUpkeep * sc.sense[l]
    + cfg.moveCost * (tx * tx + ty * ty) * speed * speed * size
    + cfg.aggressionCost * aggr
    + cfg.signalCost * Math.abs(signal)
    + cfg.depositCost * deposit
    + cfg.bondCost * bonds;
}

/** Step 5: big-enough aggressive particles drain touching non-kin (pre-move contacts). */
function predation(w: World, n: number): void {
  const cfg = w.cfg, sc = w.scratch, nb = w.nb, list = w.aliveList;
  const { demanded, lost, gained, size, e, out, sim } = sc;
  demanded.fill(0, 0, n); lost.fill(0, 0, n); gained.fill(0, 0, n);
  let np = 0;
  for (let l = 0; l < n; l++) {
    const aggr = Math.max(out[l * N_OUTPUTS + AGGRESSION], 0);
    if (aggr <= 0) continue;
    const drain = aggr * size[l] * cfg.attackRate;
    for (let k = nb.start[l]; k < nb.start[l + 1]; k++) {
      const j = nb.j[k];
      if (nb.dist[k] < (cfg.contactRadius * (size[l] + size[j])) / 2 && sim[k] < cfg.kinThreshold
        && size[l] >= cfg.preySizeRatio * size[j]) {
        sc.ensurePairs(np + 1);
        sc.pa[np] = l; sc.pb[np] = j; sc.pd[np] = drain; np++;
        demanded[j] += drain;
      }
    }
  }
  w.lastPredationPairs = np;
  if (np === 0) return;
  for (let q = 0; q < np; q++) {
    const j = sc.pb[q];
    const scale = Math.min(1, Math.max(e[j], 0) / Math.max(demanded[j], 1e-9));
    const taken = sc.pd[q] * scale;
    lost[j] += taken;
    gained[sc.pa[q]] += cfg.attackEfficiency * taken;
  }
  for (let l = 0; l < n; l++) {
    e[l] += gained[l] - lost[l];
    w.preyGained[list[l]] += gained[l];
  }
}

/** Step 6: each food item within reach goes to its single closest particle (post-move). */
function eat(w: World, n: number): void {
  const cfg = w.cfg, sc = w.scratch, list = w.aliveList;
  const fg = w.foodGrid, W = cfg.worldSize, half = W / 2;
  const { fBest, fWin, fTouched, nx, ny, size } = sc;
  const { g, xs, ys, items, starts } = fg;
  const inv = g / W;
  let nt = 0;
  for (let l = 0; l < n; l++) {
    const reach = cfg.eatRadius * size[l], r2 = reach * reach;
    const qx = nx[l], qy = ny[l];
    const R = Math.ceil(reach * inv), full = 2 * R + 1 >= g, span = full ? g : 2 * R + 1;
    const cx0 = full ? 0 : Math.min(g - 1, Math.floor(qx * inv)) - R;
    const cy0 = full ? 0 : Math.min(g - 1, Math.floor(qy * inv)) - R;
    for (let ix = 0; ix < span; ix++) {
      let cx = cx0 + ix;
      if (cx < 0) cx += g; else if (cx >= g) cx -= g;
      for (let iy = 0; iy < span; iy++) {
        let cy = cy0 + iy;
        if (cy < 0) cy += g; else if (cy >= g) cy -= g;
        const c = cx * g + cy;
        for (let k = starts[c]; k < starts[c + 1]; k++) {
          const f = items[k];
          let dx = xs[f] - qx, dy = ys[f] - qy;
          if (dx > half) dx -= W; else if (dx < -half) dx += W;
          if (dy > half) dy -= W; else if (dy < -half) dy += W;
          const d2 = dx * dx + dy * dy;
          if (d2 >= r2) continue;
          if (fWin[f] < 0) fTouched[nt++] = f;
          if (d2 < fBest[f]) { fBest[f] = d2; fWin[f] = l; }
        }
      }
    }
  }
  for (let q = 0; q < nt; q++) {
    const f = fTouched[q], l = fWin[f];
    sc.e[l] += cfg.foodEnergy;
    w.foodGained[list[l]] += cfg.foodEnergy;
    w.foodAlive[sc.fIdx[f]] = 0;
    fBest[f] = Infinity; fWin[f] = -1;
  }
}
