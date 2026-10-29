import { describe, expect, it } from 'vitest';
import { Grid, NeighbourList, selfNeighbours, nearest, nearestHit, wrapDelta } from '../src/sim/grid';
import { Rng } from '../src/sim/rng';
import { Genome, N_INPUTS, N_OUTPUTS, think, ftanh } from '../src/sim/genome';
import { PY_DEFAULTS, scaledConfig, sizeOf, type Config } from '../src/sim/config';
import { World } from '../src/sim/world';
import { particleLifeForce } from '../src/sim/act';

const W = 1.4;

function randomPoints(n: number, seed: number) {
  const rng = new Rng(seed);
  const xs = new Float32Array(n), ys = new Float32Array(n);
  for (let i = 0; i < n; i++) { xs[i] = rng.rand() * W; ys[i] = rng.rand() * W; }
  return { xs, ys };
}

describe('uniform grid on a torus', () => {
  it('selfNeighbours matches brute force (incl. wrap-around)', () => {
    const n = 700, radius = 0.128;
    const { xs, ys } = randomPoints(n, 3);
    const grid = new Grid();
    grid.build(xs, ys, n, W, radius);
    const nb = new NeighbourList();
    selfNeighbours(grid, radius, nb);
    for (let i = 0; i < n; i++) {
      const got = new Map<number, number>();
      for (let k = nb.start[i]; k < nb.start[i + 1]; k++) got.set(nb.j[k], nb.dist[k]);
      const want = new Map<number, number>();
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = wrapDelta(xs[j] - xs[i], W), dy = wrapDelta(ys[j] - ys[i], W);
        const d = Math.hypot(dx, dy);
        if (d <= radius) want.set(j, d);
      }
      expect([...got.keys()].sort((a, b) => a - b)).toEqual([...want.keys()].sort((a, b) => a - b));
      for (const [j, d] of want) expect(got.get(j)!).toBeCloseTo(d, 5);
    }
  });

  it('falls back to a full scan when the radius covers the whole grid', () => {
    const n = 50, radius = 0.6;
    const { xs, ys } = randomPoints(n, 9);
    const grid = new Grid();
    grid.build(xs, ys, n, W, radius);
    const nb = new NeighbourList();
    selfNeighbours(grid, radius, nb);
    let brute = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (i !== j && Math.hypot(wrapDelta(xs[j] - xs[i], W), wrapDelta(ys[j] - ys[i], W)) <= radius) brute++;
    }
    expect(nb.len).toBe(brute);
  });

  it('nearest (ring search) matches brute force', () => {
    const { xs, ys } = randomPoints(400, 5);
    const grid = new Grid();
    grid.build(xs, ys, 400, W, 0.05);
    const rng = new Rng(11);
    for (let q = 0; q < 300; q++) {
      const qx = rng.rand() * W, qy = rng.rand() * W, r = 0.02 + rng.rand() * 0.25;
      let best = Infinity, bj = -1;
      for (let j = 0; j < 400; j++) {
        const d = Math.hypot(wrapDelta(xs[j] - qx, W), wrapDelta(ys[j] - qy, W));
        if (d <= r && d < best) { best = d; bj = j; }
      }
      const hit = nearest(grid, qx, qy, r);
      expect(hit).toBe(bj >= 0);
      if (hit) {
        expect(nearestHit.d).toBeCloseTo(best, 5);
        expect(Math.hypot(nearestHit.dx, nearestHit.dy)).toBeCloseTo(best, 5);
      }
    }
  });
});

describe('brain', () => {
  const cfg: Config = { ...PY_DEFAULTS, capacity: 4 };

  it('forward pass equals tanh(tanh(x.W1 + b1).(W2 + P) + b2)', () => {
    const g = new Genome(cfg);
    const rng = new Rng(1);
    g.randomize(2, cfg, rng);
    const H = cfg.hidden;
    const plastic = new Float32Array(cfg.capacity * H * N_OUTPUTS);
    for (let k = 0; k < H * N_OUTPUTS; k++) plastic[2 * H * N_OUTPUTS + k] = rng.randn() * 0.1;
    const inp = new Float32Array(N_INPUTS);
    for (let i = 0; i < N_INPUTS; i++) inp[i] = rng.randn();
    const hid = new Float32Array(H), out = new Float32Array(N_OUTPUTS);
    think(g, 2, plastic, inp, 0, hid, 0, out, 0);

    const refH: number[] = [];
    for (let h = 0; h < H; h++) {
      let a = g.b1[2 * H + h];
      for (let i = 0; i < N_INPUTS; i++) a += inp[i] * g.w1[2 * g.sw1 + i * H + h];
      refH.push(Math.tanh(a));
    }
    for (let o = 0; o < N_OUTPUTS; o++) {
      let a = g.b2[2 * N_OUTPUTS + o];
      for (let h = 0; h < H; h++) a += refH[h] * (g.w2[2 * g.sw2 + h * N_OUTPUTS + o] + plastic[2 * g.sw2 + h * N_OUTPUTS + o]);
      expect(out[o]).toBeCloseTo(Math.tanh(a), 3);
    }
    for (let h = 0; h < H; h++) expect(hid[h]).toBeCloseTo(refH[h], 3);
  });

  it('fast tanh stays within 1e-4 of Math.tanh', () => {
    for (let x = -8; x <= 8; x += 0.001) expect(Math.abs(ftanh(x) - Math.tanh(x))).toBeLessThan(1e-4);
  });

  it('crossover inherits whole hidden neurons from one parent', () => {
    const g = new Genome(cfg);
    const rng = new Rng(4);
    g.randomize(0, cfg, rng);
    g.randomize(1, cfg, rng);
    g.copy(2, 0);
    g.crossover(2, 1, rng);
    const H = cfg.hidden;
    for (let h = 0; h < H; h++) {
      const fromMate = g.b1[2 * H + h] === g.b1[1 * H + h];
      for (let i = 0; i < N_INPUTS; i++) {
        const src = fromMate ? 1 : 0;
        expect(g.w1[2 * g.sw1 + i * H + h]).toBe(g.w1[src * g.sw1 + i * H + h]);
      }
    }
  });

  it('particle-life kernel: repels inside beta, triangular bump outside, zero past 1', () => {
    expect(particleLifeForce(0, 0.7, 0.3)).toBeCloseTo(-1);
    expect(particleLifeForce(0.65, 0.7, 0.3)).toBeCloseTo(0.7);
    expect(particleLifeForce(1.2, 0.7, 0.3)).toBe(0);
  });
});

describe('energy bookkeeping', () => {
  it('an isolated particle pays exactly the metabolic cost of its actions', () => {
    const cfg = scaledConfig(1, { seed: 2, initialParticles: 1, minPopulation: 0, foodCapacity: 1, foodPatches: 0, speciesInterval: 0 });
    const w = new World(cfg);
    w.foodAlive.fill(0);                  // no food anywhere
    w.collectAlive();
    const s = w.aliveList[0];
    const e0 = w.energy[s];
    w.step();
    const out = w.lastOutputs.subarray(s * N_OUTPUTS, (s + 1) * N_OUTPUTS);
    const g = w.genome;
    const size = sizeOf(cfg, g.body[s * 3]);
    const speed = cfg.speedMin + g.body[s * 3 + 1] * (cfg.speedMax - cfg.speedMin);
    const sense = cfg.senseMin + g.body[s * 3 + 2] * (cfg.senseMax - cfg.senseMin);
    const cost = cfg.baseMetabolism * size ** 0.75 + cfg.speedUpkeep * speed + cfg.senseUpkeep * sense
      + cfg.moveCost * (out[0] ** 2 + out[1] ** 2) * speed ** 2 * size
      + cfg.aggressionCost * Math.max(out[2], 0) + cfg.signalCost * Math.abs(out[3])
      + cfg.depositCost * (Math.max(out[4], 0) + Math.max(out[5], 0));
    expect(w.energy[s]).toBeCloseTo(e0 - cost, 5);
  });

  it('splitting conserves energy minus the birth cost (asexual)', () => {
    const cfg = scaledConfig(1, { seed: 5, initialParticles: 1, minPopulation: 0, foodCapacity: 1, foodPatches: 0, mateProb: 0, speciesInterval: 0 });
    const w = new World(cfg);
    w.foodAlive.fill(0);
    w.collectAlive();
    const s = w.aliveList[0];
    w.energy[s] = 10;   // force a split this step
    w.step();
    w.collectAlive();
    expect(w.population).toBe(2);
    const total = w.energy[w.aliveList[0]] + w.energy[w.aliveList[1]];
    // parent and child each get half of (energy after this step's upkeep) * (1 - birth_cost)
    expect(w.energy[w.aliveList[0]]).toBeCloseTo(w.energy[w.aliveList[1]], 6);
    expect(total).toBeLessThan(10 * (1 - cfg.birthCost));
    expect(total).toBeGreaterThan(10 * (1 - cfg.birthCost) - 0.05);
  });

  it('predation moves energy from prey to predator at attack_efficiency', () => {
    const cfg = scaledConfig(1, { seed: 1, initialParticles: 2, minPopulation: 0, foodCapacity: 1, foodPatches: 0, speciesInterval: 0 });
    const w = new World(cfg);
    w.foodAlive.fill(0);
    w.collectAlive();
    const [a, b] = [w.aliveList[0], w.aliveList[1]];
    const g = w.genome;
    // a: big, always attacking; b: small, a stranger, sitting on top of a
    g.body[a * 3] = 1; g.body[b * 3] = 0;
    g.color.set([0, 0, 0], a * 3); g.color.set([1, 1, 1], b * 3);
    g.w2.fill(0, a * g.sw2, (a + 1) * g.sw2); g.b2.fill(0, a * N_OUTPUTS, (a + 1) * N_OUTPUTS);
    g.w2.fill(0, b * g.sw2, (b + 1) * g.sw2); g.b2.fill(0, b * N_OUTPUTS, (b + 1) * N_OUTPUTS);
    g.b2[a * N_OUTPUTS + 2] = 10;   // aggression ~ 1
    w.posX[a] = 0.5; w.posY[a] = 0.5; w.posX[b] = 0.501; w.posY[b] = 0.5;
    w.energy[a] = 1; w.energy[b] = 1;
    w.step();
    const drain = Math.tanh(10) * sizeOf(cfg, 1) * cfg.attackRate;
    expect(w.preyGained[a]).toBeCloseTo(cfg.attackEfficiency * drain, 4);
    expect(w.preyGained[b]).toBe(0);
    expect(w.energy[b]).toBeLessThan(1 - drain + 1e-6);
  });
});
