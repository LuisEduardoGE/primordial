// Headless run for tuning: npx vite-node scripts/headless.ts -- [steps] [scale] [seed]
import { scaledConfig } from '../src/sim/config';
import { World } from '../src/sim/world';
import { computeStats } from '../src/sim/stats';

const args = process.argv.slice(2).filter((a) => a !== '--');
const steps = Number(args[0] ?? 3000), scale = Number(args[1] ?? 12), seed = Number(args[2] ?? 1), every = Number(args[3] ?? 250);
const w = new World(scaledConfig(scale, { seed }));
let t0 = performance.now();
for (let i = 1; i <= steps; i++) {
  w.step();
  if (i % every === 0) {
    const s = computeStats(w);
    const dt = performance.now() - t0;
    console.log(`step ${i} pop ${s.population} food ${s.food} gen ${s.meanGeneration.toFixed(1)}/${s.maxGeneration} E ${s.meanEnergy.toFixed(2)} pred ${s.predationShare.toFixed(3)} bonded ${s.bondedFraction.toFixed(3)} births ${s.births} deaths ${s.deaths} mated ${s.matedBirths} species ${JSON.stringify(s.species)} | ${(every / (dt / 1000)).toFixed(0)} steps/s`);
    t0 = performance.now();
  }
}
