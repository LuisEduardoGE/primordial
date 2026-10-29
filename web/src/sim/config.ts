/**
 * All tunable knobs, mirroring primordial/config.py.
 *
 * Defaults are the Python defaults. `scaledConfig(scale)` multiplies the world *area*
 * by `scale` and scales every extensive quantity (capacity, food, patches, population
 * floor, pheromone resolution) with it, so densities stay the same as the reference run.
 */
export interface Config {
  seed: number;
  // world
  worldSize: number;
  capacity: number;
  initialParticles: number;
  minPopulation: number;
  // particle-life physics
  nTypes: number;
  interactionRadius: number;
  repulsionBeta: number;
  forceFactor: number;
  frictionHalfLife: number;
  dt: number;
  // bodies
  sizeMin: number; sizeMax: number;
  speedMin: number; speedMax: number;
  senseMin: number; senseMax: number;
  senseRadius: number;
  // brains
  hidden: number;
  maxThrust: number;
  kinSigma: number;
  // lifetime learning
  plasticity: number;
  plasticDecay: number;
  plasticClip: number;
  rewardScale: number;
  // energy
  initialEnergy: number;
  reproduceThreshold: number;
  birthCost: number;
  baseMetabolism: number;
  speedUpkeep: number;
  senseUpkeep: number;
  moveCost: number;
  aggressionCost: number;
  signalCost: number;
  depositCost: number;
  maxAge: number;
  // predation
  contactRadius: number;
  attackRate: number;
  attackEfficiency: number;
  kinThreshold: number;
  preySizeRatio: number;
  // food
  foodCapacity: number;
  foodEnergy: number;
  foodRegrowProb: number;
  foodClumping: number;
  foodPatches: number;
  patchRadius: number;
  patchSpeed: number;
  foodSenseRadius: number;
  eatRadius: number;
  // climate
  climatePeriod: number;
  climateAmplitude: number;
  // pheromones
  pheromoneRes: number;
  pheromoneDeposit: number;
  pheromoneDiffusion: number;
  pheromoneDecay: number;
  // bonds
  maxBonds: number;
  bondRest: number;
  bondStrength: number;
  bondBreak: number;
  bondShare: number;
  bondCost: number;
  // sex
  mateProb: number;
  mateRadius: number;
  mateSimilarity: number;
  mateContribution: number;
  // mutation
  mutationRate: number;
  mutationScale: number;
  perceptionRateMult: number;
  behaviourRateMult: number;
  attractionRateMult: number;
  traitsRateMult: number;
  sigmaTau: number;
  sigmaMin: number;
  sigmaMax: number;
  neuronResetProb: number;
  colorDrift: number;
  typeFlipProb: number;
  // analysis
  speciesInterval: number;
  speciesK: number;
}

export const PY_DEFAULTS: Config = {
  seed: 0,
  worldSize: 1.4,
  capacity: 2000,
  initialParticles: 500,
  minPopulation: 80,
  nTypes: 4,
  interactionRadius: 0.08,
  repulsionBeta: 0.3,
  forceFactor: 10.0,
  frictionHalfLife: 0.04,
  dt: 0.02,
  sizeMin: 0.6, sizeMax: 1.8,
  speedMin: 0.5, speedMax: 2.0,
  senseMin: 0.5, senseMax: 1.6,
  senseRadius: 0.08,
  hidden: 16,
  maxThrust: 1.0,
  kinSigma: 0.15,
  plasticity: 1.0,
  plasticDecay: 0.01,
  plasticClip: 2.0,
  rewardScale: 20.0,
  initialEnergy: 1.5,
  reproduceThreshold: 2.0,
  birthCost: 0.1,
  baseMetabolism: 0.0012,
  speedUpkeep: 0.0002,
  senseUpkeep: 0.0002,
  moveCost: 0.002,
  aggressionCost: 0.002,
  signalCost: 0.0003,
  depositCost: 0.0005,
  maxAge: 4000,
  contactRadius: 0.015,
  attackRate: 0.03,
  attackEfficiency: 0.6,
  kinThreshold: 0.5,
  preySizeRatio: 0.8,
  foodCapacity: 700,
  foodEnergy: 0.6,
  foodRegrowProb: 0.004,
  foodClumping: 0.7,
  foodPatches: 5,
  patchRadius: 0.05,
  patchSpeed: 0.0004,
  foodSenseRadius: 0.15,
  eatRadius: 0.012,
  climatePeriod: 8000,
  climateAmplitude: 0.5,
  pheromoneRes: 96,
  pheromoneDeposit: 0.05,
  pheromoneDiffusion: 0.2,
  pheromoneDecay: 0.01,
  maxBonds: 4,
  bondRest: 0.02,
  bondStrength: 40.0,
  bondBreak: 0.07,
  bondShare: 0.05,
  bondCost: 0.0003,
  mateProb: 1.0,
  mateRadius: 0.06,
  mateSimilarity: 0.6,
  mateContribution: 0.15,
  mutationRate: 0.1,
  mutationScale: 0.2,
  perceptionRateMult: 1.0,
  behaviourRateMult: 1.0,
  attractionRateMult: 1.0,
  traitsRateMult: 1.0,
  sigmaTau: 0.0,
  sigmaMin: 0.01,
  sigmaMax: 1.0,
  neuronResetProb: 0.0,
  colorDrift: 0.05,
  typeFlipProb: 0.01,
  speciesInterval: 250,
  speciesK: 5,
};

/** Default area multiplier for the web build: 12x the reference world -> ~3-5k particles. */
export const DEFAULT_SCALE = 12;

/**
 * Scale the reference world up by `scale` in area, keeping all densities equal.
 * Capacity is not scaled linearly (the Python capacity of 2000 is ~6x the equilibrium
 * population and never binds); we keep a generous headroom instead to bound memory.
 */
export function scaledConfig(scale = DEFAULT_SCALE, overrides: Partial<Config> = {}): Config {
  const c: Config = { ...PY_DEFAULTS };
  const lin = Math.sqrt(scale);
  c.worldSize = PY_DEFAULTS.worldSize * lin;
  c.foodCapacity = Math.round(PY_DEFAULTS.foodCapacity * scale);
  c.foodPatches = Math.max(1, Math.round(PY_DEFAULTS.foodPatches * scale));
  c.initialParticles = Math.round(PY_DEFAULTS.initialParticles * scale * 0.5);
  c.minPopulation = Math.round(PY_DEFAULTS.minPopulation * scale);
  c.capacity = Math.max(2000, Math.round(700 * scale));
  c.pheromoneRes = Math.round((PY_DEFAULTS.pheromoneRes * lin) / 4) * 4;
  return { ...c, ...overrides };
}

export const sizeOf = (c: Config, g: number) => c.sizeMin + g * (c.sizeMax - c.sizeMin);
export const speedOf = (c: Config, g: number) => c.speedMin + g * (c.speedMax - c.speedMin);
export const senseOf = (c: Config, g: number) => c.senseMin + g * (c.senseMax - c.senseMin);

/** Live-tweakable parameters exposed as sliders. */
export type LiveParam = 'foodRegrowProb' | 'attackRate' | 'mutationRate' | 'climateAmplitude';
