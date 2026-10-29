/** Messages between the main thread and the simulation worker. */
import type { Stats } from './sim/stats';
import type { LiveParam } from './sim/config';

export type ColorMode = 'lineage' | 'species' | 'type';

/** Floats per particle in FrameMsg.particles: x, y, size, r, g, b, signal, aggression. */
export const P_STRIDE = 8;

export type ToMain =
  | { type: 'bench'; mmacs: number }
  | { type: 'ready'; worldSize: number; capacity: number; pheromoneRes: number; foodCapacity: number; patchRadius: number; seed: number; scale: number }
  | FrameMsg;

export interface Inspect {
  slot: number;
  uid: number;
  x: number; y: number;
  energy: number; threshold: number;
  age: number; maxAge: number;
  generation: number;
  size: number; speed: number; sense: number;
  ptype: number;
  color: [number, number, number];
  species: string;
  bonds: number;
  aggression: number; signal: number;
  logEta: number;
  sigma: number[];
  foodGained: number; preyGained: number;
  inputs: number[]; hidden: number[]; outputs: number[];
  w1: number[];   // N_INPUTS x hidden, row-major [i][h]
  w2: number[];   // hidden x N_OUTPUTS (genetic + plastic), row-major [h][o]
  attraction: number[];
}

export interface FrameMsg {
  type: 'frame';
  n: number;
  particles: Float32Array;
  nBonds: number;
  bonds: Float32Array;      // x1, y1, x2, y2 (second end unwrapped relative to the first)
  nFood: number;
  food: Float32Array;       // x, y
  patches: Float32Array;    // x, y
  pheromone: Uint8Array | null;  // RGBA, res x res, texel (x, y) at y * res + x
  stats: Stats | null;
  stepMs: number;           // mean wall time per sim step over the last frame
  stepsDone: number;
  inspect: Inspect | null;
  event: string | null;
}

export type ToWorker =
  | { type: 'init'; seed: number; scale: number }
  | { type: 'reset'; seed: number; scale: number }
  | { type: 'frame'; steps: number; budgetMs: number; colorMode: ColorMode; pheromone: boolean; stats: boolean; recycle: ArrayBuffer[] }
  | { type: 'select'; x: number; y: number; radius: number }
  | { type: 'deselect' }
  | { type: 'bench' }
  | { type: 'tool'; tool: 'food' | 'meteor'; x: number; y: number }
  | { type: 'param'; key: LiveParam; value: number };
