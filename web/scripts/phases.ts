// Per-phase timing: npx esbuild ... then node out.mjs [steps] [scale]
import { scaledConfig } from '../src/sim/config';
import { World } from '../src/sim/world';
import { updateLiving } from '../src/sim/update';
import { births } from '../src/sim/births';
const args = process.argv.slice(2);
const steps = Number(args[0] ?? 120), scale = Number(args[1] ?? 16);
const w = new World(scaledConfig(scale, { seed: 1 }));
const acc: Record<string, number> = {};
const t = (k: string, f: () => void) => { const t0 = performance.now(); f(); acc[k] = (acc[k] ?? 0) + performance.now() - t0; };
for (let i = 1; i <= steps; i++) {
  const n = w.collectAlive();
  t('living', () => updateLiving(w, n));
  t('deaths', () => { let nd = 0; for (let k = 0; k < n; k++) { const s = w.aliveList[k]; if (w.energy[s] <= 0 || w.age[s] > w.cfg.maxAge) w.scratch.dead[nd++] = s; } w.kill(w.scratch.dead, nd); });
  t('births', () => births(w));
  t('patches+food', () => { w.movePatches(); w.regrowFood(); });
  t('diffuse', () => w.diffusePheromone());
  w.stepCount++;
  if (i % 20 === 0) {
    console.log(i, 'pop', w.population, Object.entries(acc).map(([k, v]) => `${k} ${(v / 20).toFixed(1)}ms`).join('  '));
    for (const k in acc) acc[k] = 0;
  }
}
