/**
 * The simulation state and the per-step orchestration (primordial/world.py).
 * Fixed-capacity structure-of-arrays with an alive mask: births and deaths never allocate.
 * The heavy per-particle update lives in update.ts; births in births.ts.
 */
import { type Config, sizeOf, speedOf, senseOf } from './config';
import { Rng } from './rng';
import { Grid, NeighbourList, wrapDelta } from './grid';
import { Genome, N_INPUTS, N_MEMORY, N_OUTPUTS } from './genome';
import { updateLiving, Scratch } from './update';
import { births } from './births';
import { SpeciesTracker } from './species';

export const N_EMA = 5;

export class World {
  cfg: Config;
  rng: Rng;
  C: number;
  posX: Float32Array; posY: Float32Array; velX: Float32Array; velY: Float32Array;
  energy: Float32Array;
  age: Int32Array; generation: Int32Array; alive: Uint8Array; uid: Int32Array;
  aggression: Float32Array; signal: Float32Array;
  memory: Float32Array; plastic: Float32Array; rewardAvg: Float32Array;
  bonds: Int32Array; ema: Float32Array;
  foodGained: Float32Array; preyGained: Float32Array; speciesId: Int16Array;
  lastInputs: Float32Array; lastHidden: Float32Array; lastOutputs: Float32Array;
  genome: Genome;
  foodX: Float32Array; foodY: Float32Array; foodAlive: Uint8Array;
  patchX: Float32Array; patchY: Float32Array; patchHeading: Float32Array;
  pheromone: Float32Array; pheromoneTmp: Float32Array; pheromoneActive = false;
  foodMultiplier = 1;
  stepCount = 0;
  births = 0; deaths = 0; matedBirths = 0; bondsFormed = 0; nextUid = 0;
  lastPredationPairs = 0;
  species: SpeciesTracker;
  // reusable scratch
  scratch: Scratch;
  grid = new Grid();
  foodGrid = new Grid();
  nb = new NeighbourList();
  aliveList: Int32Array;
  population = 0;

  constructor(cfg: Config) {
    this.cfg = cfg;
    this.rng = new Rng(cfg.seed);
    const C = (this.C = cfg.capacity);
    const H = cfg.hidden;
    const f = (n: number) => new Float32Array(n);
    this.posX = f(C); this.posY = f(C); this.velX = f(C); this.velY = f(C);
    this.energy = f(C);
    this.age = new Int32Array(C); this.generation = new Int32Array(C);
    this.alive = new Uint8Array(C); this.uid = new Int32Array(C).fill(-1);
    this.aggression = f(C); this.signal = f(C);
    this.memory = f(C * N_MEMORY); this.plastic = f(C * H * N_OUTPUTS); this.rewardAvg = f(C);
    this.bonds = new Int32Array(C * cfg.maxBonds).fill(-1);
    this.ema = f(C * N_EMA);
    this.foodGained = f(C); this.preyGained = f(C);
    this.speciesId = new Int16Array(C).fill(-1);
    this.lastInputs = f(C * N_INPUTS); this.lastHidden = f(C * H); this.lastOutputs = f(C * N_OUTPUTS);
    this.genome = new Genome(cfg);
    this.aliveList = new Int32Array(C);

    const W = cfg.worldSize;
    this.foodX = f(cfg.foodCapacity); this.foodY = f(cfg.foodCapacity);
    this.foodAlive = new Uint8Array(cfg.foodCapacity).fill(1);
    for (let i = 0; i < cfg.foodCapacity; i++) {
      this.foodX[i] = this.rng.rand() * W;
      this.foodY[i] = this.rng.rand() * W;
    }
    this.patchX = f(cfg.foodPatches); this.patchY = f(cfg.foodPatches); this.patchHeading = f(cfg.foodPatches);
    for (let i = 0; i < cfg.foodPatches; i++) {
      this.patchX[i] = this.rng.rand() * W;
      this.patchY[i] = this.rng.rand() * W;
    }
    for (let i = 0; i < cfg.foodPatches; i++) this.patchHeading[i] = this.rng.rand() * 2 * Math.PI;
    const r = cfg.pheromoneRes;
    this.pheromone = f(2 * r * r);
    this.pheromoneTmp = f(2 * r * r);
    this.scratch = new Scratch(cfg);
    this.species = new SpeciesTracker(cfg.speciesK, cfg.seed);
    this.spawn(cfg.initialParticles);
  }

  sizeAt(s: number): number { return sizeOf(this.cfg, this.genome.body[s * 3]); }
  speedAt(s: number): number { return speedOf(this.cfg, this.genome.body[s * 3 + 1]); }
  senseAt(s: number): number { return senseOf(this.cfg, this.genome.body[s * 3 + 2]); }

  season(): number {
    const p = this.cfg.climatePeriod;
    return p > 0 ? Math.sin((2 * Math.PI * this.stepCount) / p) : 0;
  }

  /** Refresh the compact list of living slots (ascending slot order, like torch.nonzero). */
  collectAlive(): number {
    let n = 0;
    const a = this.alive, list = this.aliveList;
    for (let s = 0; s < this.C; s++) if (a[s]) list[n++] = s;
    this.population = n;
    return n;
  }

  resetSlot(s: number): void {
    const H = this.cfg.hidden, mb = this.cfg.maxBonds;
    this.velX[s] = 0; this.velY[s] = 0; this.age[s] = 0;
    this.aggression[s] = 0; this.signal[s] = 0; this.rewardAvg[s] = 0;
    this.foodGained[s] = 0; this.preyGained[s] = 0;
    this.memory.fill(0, s * N_MEMORY, (s + 1) * N_MEMORY);
    this.plastic.fill(0, s * H * N_OUTPUTS, (s + 1) * H * N_OUTPUTS);
    this.ema.fill(0, s * N_EMA, (s + 1) * N_EMA);
    this.lastInputs.fill(0, s * N_INPUTS, (s + 1) * N_INPUTS);
    this.lastHidden.fill(0, s * H, (s + 1) * H);
    this.lastOutputs.fill(0, s * N_OUTPUTS, (s + 1) * N_OUTPUTS);
    this.bonds.fill(-1, s * mb, (s + 1) * mb);
    this.speciesId[s] = -1;
  }

  /** Abiogenesis: brand-new random particles in the lowest free slots. */
  spawn(n: number): number {
    const W = this.cfg.worldSize;
    let k = 0;
    for (let s = 0; s < this.C && k < n; s++) {
      if (this.alive[s]) continue;
      this.resetSlot(s);
      this.posX[s] = this.rng.rand() * W;
      this.posY[s] = this.rng.rand() * W;
      this.energy[s] = this.cfg.initialEnergy;
      this.generation[s] = 0;
      this.alive[s] = 1;
      this.genome.randomize(s, this.cfg, this.rng);
      this.uid[s] = this.nextUid++;
      k++;
    }
    this.population += k;
    return k;
  }

  /** Remove particles in `dead` (slot list); corpses become food where there's room. */
  kill(dead: ArrayLike<number>, count: number): void {
    if (count === 0) return;
    const mb = this.cfg.maxBonds;
    let fi = 0;
    const F = this.cfg.foodCapacity;
    for (let k = 0; k < count; k++) {
      const s = dead[k];
      this.alive[s] = 0;
      for (let b = 0; b < mb; b++) {
        const p = this.bonds[s * mb + b];
        if (p >= 0) for (let q = 0; q < mb; q++) if (this.bonds[p * mb + q] === s) this.bonds[p * mb + q] = -1;
        this.bonds[s * mb + b] = -1;
      }
      while (fi < F && this.foodAlive[fi]) fi++;
      if (fi < F) {
        this.foodX[fi] = this.posX[s];
        this.foodY[fi] = this.posY[s];
        this.foodAlive[fi] = 1;
      }
    }
    this.deaths += count;
    this.population -= count;
  }

  step(): void {
    const cfg = this.cfg;
    const n = this.collectAlive();
    if (n > 0) updateLiving(this, n);
    // deaths
    const dead = this.scratch.dead;
    let nd = 0;
    for (let k = 0; k < n; k++) {
      const s = this.aliveList[k];
      if (this.energy[s] <= 0 || this.age[s] > cfg.maxAge) dead[nd++] = s;
    }
    this.kill(dead, nd);
    births(this);
    this.movePatches();
    this.regrowFood();
    this.diffusePheromone();
    if (this.population < cfg.minPopulation) this.spawn(cfg.minPopulation - this.population);
    this.stepCount++;
    if (cfg.speciesInterval && this.stepCount % cfg.speciesInterval === 0) this.species.classify(this);
  }

  movePatches(): void {
    const cfg = this.cfg, W = cfg.worldSize;
    for (let i = 0; i < cfg.foodPatches; i++) this.patchHeading[i] += this.rng.randn() * 0.05;
    for (let i = 0; i < cfg.foodPatches; i++) {
      const h = this.patchHeading[i];
      this.patchX[i] = mod(this.patchX[i] + Math.cos(h) * cfg.patchSpeed, W);
      this.patchY[i] = mod(this.patchY[i] + Math.sin(h) * cfg.patchSpeed, W);
    }
  }

  regrowFood(): void {
    const cfg = this.cfg, W = cfg.worldSize, rng = this.rng;
    const rate = cfg.foodRegrowProb * (1 + cfg.climateAmplitude * this.season()) * this.foodMultiplier;
    for (let i = 0; i < cfg.foodCapacity; i++) {
      if (this.foodAlive[i] || rng.rand() >= rate) continue;
      if (cfg.foodPatches > 0 && rng.rand() < cfg.foodClumping) {
        const p = rng.int(cfg.foodPatches);
        this.foodX[i] = mod(this.patchX[p] + rng.randn() * cfg.patchRadius, W);
        this.foodY[i] = mod(this.patchY[p] + rng.randn() * cfg.patchRadius, W);
      } else {
        this.foodX[i] = rng.rand() * W;
        this.foodY[i] = rng.rand() * W;
      }
      this.foodAlive[i] = 1;
    }
  }

  diffusePheromone(): void {
    if (!this.pheromoneActive) return;
    const cfg = this.cfg, r = cfg.pheromoneRes, p = this.pheromone, out = this.pheromoneTmp;
    const a = cfg.pheromoneDiffusion, keep = (1 - a) * (1 - cfg.pheromoneDecay), mix = (a / 4) * (1 - cfg.pheromoneDecay);
    let peak = 0;
    for (let c = 0; c < 2; c++) {
      const base = c * r * r;
      for (let x = 0; x < r; x++) {
        const xm = base + (x === 0 ? r - 1 : x - 1) * r, xp = base + (x === r - 1 ? 0 : x + 1) * r, x0 = base + x * r;
        // wrapped edges of the row
        let v = keep * p[x0] + mix * (p[xm] + p[xp] + p[x0 + r - 1] + p[x0 + 1]);
        out[x0] = v > 1e-7 ? v : 0;
        v = keep * p[x0 + r - 1] + mix * (p[xm + r - 1] + p[xp + r - 1] + p[x0 + r - 2] + p[x0]);
        out[x0 + r - 1] = v > 1e-7 ? v : 0;
        for (let y = 1; y < r - 1; y++) {
          v = keep * p[x0 + y] + mix * (p[xm + y] + p[xp + y] + p[x0 + y - 1] + p[x0 + y + 1]);
          out[x0 + y] = v > 1e-7 ? v : 0;   // flush tiny values (avoids float32 denormals)
          if (v > peak) peak = v;
        }
      }
    }
    const any = peak > 1e-6;
    this.pheromoneTmp = p;
    this.pheromone = out;
    this.pheromoneActive = any;
  }

  // ---------------------------------------------------------------- god tools
  dropFood(x: number, y: number, n = 12, spread = 0.02): number {
    const W = this.cfg.worldSize;
    let k = 0;
    for (let i = 0; i < this.cfg.foodCapacity && k < n; i++) {
      if (this.foodAlive[i]) continue;
      this.foodX[i] = mod(x + this.rng.randn() * spread, W);
      this.foodY[i] = mod(y + this.rng.randn() * spread, W);
      this.foodAlive[i] = 1;
      k++;
    }
    return k;
  }

  meteor(x: number, y: number, radius = 0.12): number {
    const W = this.cfg.worldSize, dead = this.scratch.dead;
    let n = 0;
    for (let s = 0; s < this.C; s++) {
      if (!this.alive[s]) continue;
      const dx = wrapDelta(this.posX[s] - x, W), dy = wrapDelta(this.posY[s] - y, W);
      if (dx * dx + dy * dy < radius * radius) dead[n++] = s;
    }
    this.kill(dead, n);
    this.dropFood(x, y, 60, radius / 2);
    return n;
  }
}

export function mod(v: number, W: number): number {
  v %= W;
  return v < 0 ? v + W : v >= W ? 0 : v;
}
