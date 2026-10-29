/**
 * Heritable traits stored as structure-of-arrays, plus the batched brain.
 * Mirrors primordial/genome.py: same input/output layout, same initialisation,
 * crossover (neuron-wise) and mutation (self-adaptive sigma, per-layer rates, neuron resets).
 */
import type { Config } from './config';
import type { Rng } from './rng';

export const N_SENSES = 23;
export const N_MEMORY = 2;
export const N_INPUTS = N_SENSES + N_MEMORY;
export const THRUST_X = 0;
export const THRUST_Y = 1;
export const AGGRESSION = 2;
export const SIGNAL = 3;
export const DEPOSIT_A = 4;
export const DEPOSIT_B = 5;
export const BOND = 6;
export const N_ACTIONS = 7;
export const MEMORY0 = N_ACTIONS;
export const N_OUTPUTS = N_ACTIONS + N_MEMORY;

export const SENSE_NAMES = [
  'food dx', 'food dy', 'food near', 'kin dx', 'kin dy', 'stranger dx', 'stranger dy',
  'crowding', 'energy', 'vel x', 'vel y', 'signal', 'signal dx', 'signal dy',
  'scent A dx', 'scent A dy', 'scent A', 'scent B dx', 'scent B dy', 'scent B',
  'bonds', 'threat', 'season', 'memory 1', 'memory 2',
];
export const OUTPUT_NAMES = ['thrust x', 'thrust y', 'attack', 'signal', 'drop A', 'drop B', 'bond', 'memory 1', 'memory 2'];

export const PERCEPTION = 0, BEHAVIOUR = 1, PHYSICS = 2, TRAITS = 3;
export const N_GROUPS = 4;

export class Genome {
  readonly H: number;
  readonly T: number;
  readonly sw1: number;   // stride of w1 per particle (N_INPUTS * H), layout [i][h]
  readonly sw2: number;   // stride of w2 per particle (H * N_OUTPUTS), layout [h][o]
  w1: Float32Array; b1: Float32Array; w2: Float32Array; b2: Float32Array;
  attraction: Float32Array; color: Float32Array; ptype: Uint8Array;
  sigma: Float32Array; body: Float32Array; bud: Float32Array; logEta: Float32Array;

  constructor(cfg: Config) {
    const C = cfg.capacity;
    this.H = cfg.hidden;
    this.T = cfg.nTypes;
    this.sw1 = N_INPUTS * this.H;
    this.sw2 = this.H * N_OUTPUTS;
    this.w1 = new Float32Array(C * this.sw1);
    this.b1 = new Float32Array(C * this.H);
    this.w2 = new Float32Array(C * this.sw2);
    this.b2 = new Float32Array(C * N_OUTPUTS);
    this.attraction = new Float32Array(C * this.T);
    this.color = new Float32Array(C * 3);
    this.ptype = new Uint8Array(C);
    this.sigma = new Float32Array(C * N_GROUPS);
    this.body = new Float32Array(C * 3);
    this.bud = new Float32Array(C);
    this.logEta = new Float32Array(C);
  }

  randomize(s: number, cfg: Config, rng: Rng): void {
    const H = this.H;
    const a1 = 1 / Math.sqrt(N_INPUTS), a2 = 1 / Math.sqrt(H);
    for (let k = 0; k < this.sw1; k++) this.w1[s * this.sw1 + k] = rng.randn() * a1;
    for (let k = 0; k < H; k++) this.b1[s * H + k] = rng.randn() * 0.1;
    for (let k = 0; k < this.sw2; k++) this.w2[s * this.sw2 + k] = rng.randn() * a2;
    for (let k = 0; k < N_OUTPUTS; k++) this.b2[s * N_OUTPUTS + k] = rng.randn() * 0.1;
    for (let k = 0; k < this.T; k++) this.attraction[s * this.T + k] = rng.rand() * 2 - 1;
    for (let k = 0; k < 3; k++) this.color[s * 3 + k] = rng.rand();
    this.ptype[s] = rng.int(cfg.nTypes);
    for (let k = 0; k < N_GROUPS; k++) this.sigma[s * N_GROUPS + k] = cfg.mutationScale;
    for (let k = 0; k < 3; k++) this.body[s * 3 + k] = 0.3 + rng.rand() * 0.4;
    this.bud[s] = rng.rand() * 0.2;
    this.logEta[s] = -3 + rng.rand() * 1.5;
  }

  copy(dst: number, src: number): void {
    const cp = (a: Float32Array | Uint8Array, stride: number) =>
      a.copyWithin(dst * stride, src * stride, src * stride + stride);
    cp(this.w1, this.sw1); cp(this.b1, this.H); cp(this.w2, this.sw2); cp(this.b2, N_OUTPUTS);
    cp(this.attraction, this.T); cp(this.color, 3); cp(this.ptype, 1);
    cp(this.sigma, N_GROUPS); cp(this.body, 3); cp(this.bud, 1); cp(this.logEta, 1);
  }

  /** Mix mate `m` into child `c` (which already holds a copy of its first parent). */
  crossover(c: number, m: number, rng: Rng): void {
    const H = this.H;
    for (let h = 0; h < H; h++) {
      if (rng.rand() < 0.5) {   // whole neuron: incoming weights, bias, outgoing weights
        for (let i = 0; i < N_INPUTS; i++) this.w1[c * this.sw1 + i * H + h] = this.w1[m * this.sw1 + i * H + h];
        this.b1[c * H + h] = this.b1[m * H + h];
        for (let o = 0; o < N_OUTPUTS; o++) this.w2[c * this.sw2 + h * N_OUTPUTS + o] = this.w2[m * this.sw2 + h * N_OUTPUTS + o];
      }
    }
    const per = (a: Float32Array | Uint8Array, stride: number) => {
      for (let k = 0; k < stride; k++) if (rng.rand() < 0.5) a[c * stride + k] = a[m * stride + k];
    };
    per(this.b2, N_OUTPUTS); per(this.attraction, this.T); per(this.sigma, N_GROUPS);
    per(this.body, 3); per(this.bud, 1); per(this.logEta, 1); per(this.ptype, 1);
    for (let k = 0; k < 3; k++) this.color[c * 3 + k] = (this.color[c * 3 + k] + this.color[m * 3 + k]) / 2;
  }

  /** Mutate the freshly copied genome in slot `s`. */
  mutate(s: number, cfg: Config, rng: Rng): void {
    const H = this.H;
    for (let k = 0; k < 3; k++) {
      const v = this.color[s * 3 + k] + rng.randn() * cfg.colorDrift;
      this.color[s * 3 + k] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    const flip = rng.rand() < cfg.typeFlipProb;
    const nt = rng.int(cfg.nTypes);
    if (flip) this.ptype[s] = nt;

    // 1. self-adaptation
    if (cfg.sigmaTau > 0) {
      for (let k = 0; k < N_GROUPS; k++) {
        const v = this.sigma[s * N_GROUPS + k] * Math.exp(cfg.sigmaTau * rng.randn());
        this.sigma[s * N_GROUPS + k] = Math.min(cfg.sigmaMax, Math.max(cfg.sigmaMin, v));
      }
    }
    const sg = (g: number) => this.sigma[s * N_GROUPS + g];
    const r = cfg.mutationRate;
    // 2. per-layer rates
    const jitter = (a: Float32Array, off: number, len: number, rate: number, scale: number, lo = -Infinity, hi = Infinity) => {
      for (let k = off; k < off + len; k++) {
        if (rng.rand() < rate) {
          let v = a[k] + rng.randn() * scale;
          if (v < lo) v = lo; else if (v > hi) v = hi;
          a[k] = v;
        }
      }
    };
    jitter(this.w1, s * this.sw1, this.sw1, r * cfg.perceptionRateMult, sg(PERCEPTION));
    jitter(this.b1, s * H, H, r * cfg.perceptionRateMult, sg(PERCEPTION));
    jitter(this.w2, s * this.sw2, this.sw2, r * cfg.behaviourRateMult, sg(BEHAVIOUR));
    jitter(this.b2, s * N_OUTPUTS, N_OUTPUTS, r * cfg.behaviourRateMult, sg(BEHAVIOUR));
    jitter(this.attraction, s * this.T, this.T, r * cfg.attractionRateMult, sg(PHYSICS), -1, 1);
    const tr = r * cfg.traitsRateMult;
    jitter(this.body, s * 3, 3, tr, sg(TRAITS) * 0.25, 0, 1);
    jitter(this.bud, s, 1, tr, sg(TRAITS) * 0.25, 0, 1);
    jitter(this.logEta, s, 1, tr, sg(TRAITS), -4, -0.5);

    // 3. neuron reset
    if (rng.rand() < cfg.neuronResetProb) {
      const h = rng.int(H);
      const a1 = 1 / Math.sqrt(N_INPUTS), a2 = 1 / Math.sqrt(H);
      for (let i = 0; i < N_INPUTS; i++) this.w1[s * this.sw1 + i * H + h] = rng.randn() * a1;
      this.b1[s * H + h] = rng.randn() * 0.1;
      for (let o = 0; o < N_OUTPUTS; o++) this.w2[s * this.sw2 + h * N_OUTPUTS + o] = rng.randn() * a2;
    }
  }
}

/**
 * tanh via a 7/6 Pade approximant, clamped at |x| > 4.97 (max abs error 1e-4, ~2.5x faster
 * than Math.tanh in V8). Used for every neuron activation.
 */
export function ftanh(x: number): number {
  if (x > 4.97) return 1;
  if (x < -4.97) return -1;
  const x2 = x * x;
  return (x * (135135 + x2 * (17325 + x2 * (378 + x2)))) / (135135 + x2 * (62370 + x2 * (3150 + 28 * x2)));
}

/**
 * One brain forward pass for slot `s`:
 *   hidden = tanh(inputs . w1 + b1);  out = tanh(hidden . (w2 + plastic) + b2)
 * `inp` (N_INPUTS) is read at inOff; hidden/out are written at their offsets.
 */
export function think(
  g: Genome, s: number, plastic: Float32Array,
  inp: Float32Array, inOff: number, hid: Float32Array, hOff: number, out: Float32Array, oOff: number,
): void {
  const H = g.H;
  const w1 = g.w1, w2 = g.w2;
  const w1o = s * g.sw1, w2o = s * g.sw2;
  for (let h = 0; h < H; h++) hid[hOff + h] = g.b1[s * H + h];
  for (let i = 0; i < N_INPUTS; i++) {
    const x = inp[inOff + i];
    if (x === 0) continue;
    const row = w1o + i * H;
    for (let h = 0; h < H; h++) hid[hOff + h] += x * w1[row + h];
  }
  for (let h = 0; h < H; h++) hid[hOff + h] = ftanh(hid[hOff + h]);
  for (let o = 0; o < N_OUTPUTS; o++) out[oOff + o] = g.b2[s * N_OUTPUTS + o];
  for (let h = 0; h < H; h++) {
    const x = hid[hOff + h];
    const row = w2o + h * N_OUTPUTS;
    for (let o = 0; o < N_OUTPUTS; o++) out[oOff + o] += x * (w2[row + o] + plastic[row + o]);
  }
  for (let o = 0; o < N_OUTPUTS; o++) out[oOff + o] = ftanh(out[oOff + o]);
}
