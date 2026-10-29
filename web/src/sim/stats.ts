/** World.stats() equivalent: cheap population summaries for the HUD and chart. */
import { sizeOf, speedOf, senseOf } from './config';
import { N_GROUPS } from './genome';
import type { World } from './world';

export interface Stats {
  step: number;
  population: number;
  food: number;
  births: number;
  deaths: number;
  matedBirths: number;
  bondsFormed: number;
  meanEnergy: number;
  meanGeneration: number;
  maxGeneration: number;
  meanAggression: number;
  meanSignal: number;
  predationShare: number;
  bondedFraction: number;
  meanSize: number;
  meanSpeed: number;
  meanSense: number;
  meanLogEta: number;
  colorDiversity: number;
  lineages: number;
  season: number;
  sigma: number[];
  species: Record<string, number>;
  speciesNames: string[];
  attackers: number;
  minPopulation: number;
}

export function computeStats(w: World): Stats {
  const cfg = w.cfg, g = w.genome, n = w.collectAlive(), list = w.aliveList, mb = cfg.maxBonds;
  let food = 0;
  for (let i = 0; i < cfg.foodCapacity; i++) food += w.foodAlive[i];
  let e = 0, gen = 0, maxGen = 0, ag = 0, sig = 0, prey = 0, fed = 0, bonded = 0, sz = 0, sp = 0, se = 0, eta = 0, attackers = 0;
  const sigma = new Array(N_GROUPS).fill(0);
  const cm = [0, 0, 0], c2 = [0, 0, 0];
  const bins = 4;
  const binCounts = new Int32Array(bins * bins * bins);
  const speciesCounts = new Int32Array(Math.max(1, w.species.names.length));
  for (let l = 0; l < n; l++) {
    const s = list[l];
    e += w.energy[s];
    gen += w.generation[s];
    if (w.generation[s] > maxGen) maxGen = w.generation[s];
    ag += w.aggression[s];
    if (w.aggression[s] > 0.3) attackers++;
    sig += Math.abs(w.signal[s]);
    prey += w.preyGained[s];
    fed += w.foodGained[s];
    for (let b = 0; b < mb; b++) if (w.bonds[s * mb + b] >= 0) { bonded++; break; }
    sz += sizeOf(cfg, g.body[s * 3]);
    sp += speedOf(cfg, g.body[s * 3 + 1]);
    se += senseOf(cfg, g.body[s * 3 + 2]);
    eta += g.logEta[s];
    for (let q = 0; q < N_GROUPS; q++) sigma[q] += g.sigma[s * N_GROUPS + q];
    let key = 0;
    for (let c = 0; c < 3; c++) {
      const v = g.color[s * 3 + c];
      cm[c] += v; c2[c] += v * v;
      key = key * bins + Math.min(bins - 1, Math.floor(v * bins));
    }
    binCounts[key]++;
    const sid = w.speciesId[s];
    if (sid >= 0 && sid < speciesCounts.length) speciesCounts[sid]++;
  }
  const d = Math.max(n, 1);
  let div = 0;
  for (let c = 0; c < 3; c++) {
    const m = cm[c] / d;
    div += n > 1 ? Math.sqrt(Math.max(0, (c2[c] - n * m * m) / (n - 1))) : 0;
  }
  // Effective number of lineages: exp(Shannon entropy) over colour cells.
  let entropy = 0;
  for (let i = 0; i < binCounts.length; i++) {
    if (binCounts[i] > 0) { const p = binCounts[i] / d; entropy -= p * Math.log(p); }
  }
  const lineages = n > 0 ? Math.round(Math.exp(entropy) * 10) / 10 : 0;
  const species: Record<string, number> = {};
  w.species.names.forEach((name, i) => { if (speciesCounts[i] > 0) species[name] = speciesCounts[i]; });
  return {
    step: w.stepCount,
    population: n,
    food,
    births: w.births,
    deaths: w.deaths,
    matedBirths: w.matedBirths,
    bondsFormed: w.bondsFormed,
    meanEnergy: e / d,
    meanGeneration: gen / d,
    maxGeneration: maxGen,
    meanAggression: ag / d,
    meanSignal: sig / d,
    predationShare: prey / (prey + fed + 1e-6),
    bondedFraction: bonded / d,
    meanSize: sz / d,
    meanSpeed: sp / d,
    meanSense: se / d,
    meanLogEta: eta / d,
    colorDiversity: div / 3,
    lineages,
    season: w.season(),
    sigma: sigma.map((v) => v / d),
    species,
    speciesNames: [...w.species.names],
    attackers,
    minPopulation: cfg.minPopulation,
  };
}
