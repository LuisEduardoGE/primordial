# primordial · web

A browser port of the `primordial` artificial-life simulation. The same ecosystem runs live at
~12x the reference world area (about 3–5k particles), rendered with three.js. `npm run build`
produces **one self-contained file**, `dist/index.html`, with all JS inlined, including three.js
and the simulation worker. It makes no network requests and needs no external fonts.

## Run / build

```bash
cd web
npm install
npm run dev        # Vite dev server with hot reload
npm run build      # typecheck + single-file build -> dist/index.html
npm test           # vitest: grid vs brute force, brain forward pass, energy bookkeeping
npm run preview    # serve dist/ locally (the worker is inlined as a blob URL, so any static host works)
```

URL parameters: `?seed=1&scale=12&speed=4&thread=worker|main`.

- `scale` is the world-area multiplier relative to the Python defaults (4, 8, 12, 20 are in the UI).
- `thread` forces where the simulation runs (see Performance).

Headless tuning helpers (Node, bundled with esbuild to keep imports cheap):

```bash
npx esbuild scripts/headless.ts --bundle --platform=node --format=esm --outfile=/tmp/h.mjs
node /tmp/h.mjs 8000 12 1 250    # steps, scale, seed, print-every
```

`scripts/phases.ts` prints per-phase step timings in the same way.

## Architecture

```
src/
  sim/                       pure TypeScript simulation, no DOM (runs in a worker or in Node)
    config.ts                Python defaults (PY_DEFAULTS) + scaledConfig(scale)
    rng.ts                   seeded sfc32 PRNG: rand / randn / int / shuffle
    grid.ts                  torus spatial hash: counting-sort build, CSR neighbour lists, ring-search nearest
    genome.ts                SoA genome, brain forward pass, crossover, mutation
    world.ts                 state arrays (fixed capacity + alive mask), step(), spawn/kill, food, patches, pheromone, god tools
    update.ts                step 1-2: neighbourhood, kinship, bonds, senses, think
    act.ts                   step 3-7: physics, metabolism, sharing, predation, eating, plasticity, scent, write-back
    bonds.ts                 step 8: bond break/form
    births.ts                step 10: mate search, crossover, mutation, budding
    species.ts               warm-started k-means behavioural species + naming (species.py)
    stats.ts                 HUD/chart statistics (World.stats)
  host.ts                    owns the World; answers frame requests with render buffers (transferred + recycled)
  worker.ts                  Web Worker entry -> host
  channel.ts                 worker transport, with automatic main-thread fallback
  protocol.ts                message types; particle buffer layout (x, y, size, r, g, b, signal, aggression)
  render/                    three.js: view.ts (layers, 3x3 torus tiling), camera.ts (pan/zoom), trails.ts (ping-pong RT), shaders.ts
  ui/                        HUD, controls, inspector + brain diagram, chart, pointer input
```

Each animation frame, the main thread asks the host for a frame, with at most one request in
flight. The host runs up to *speed* steps within a 28 ms budget and then posts:

- particles as one interleaved `Float32Array`, uploaded straight into a three.js `InterleavedBuffer`
- bond segments and food positions
- the pheromone field as RGBA bytes, every other frame
- stats, every fourth frame
- the inspected particle's brain

Typed-array buffers are transferred and then sent back for reuse, so steady state does not
allocate per frame. Rendering runs at display rate whatever the simulation speed is.

The torus is drawn as a 3x3 tiling around the camera centre. The camera centre is kept in
`[0, W)` and zoom-out is capped at a 2W-wide view, so nine fixed tiles always cover the screen.
Geometry and materials are shared between tiles, so each extra tile costs only draw calls.

## Fidelity to the Python engine

These match `primordial/world.py` and `genome.py`:

- **Brain and senses:** the 25-input layout (23 senses + 2 memory), the 9 outputs and every
  sense formula.
- **Body and physics:** the body gene → size/speed/sense mappings, the particle-life kernel,
  thrust/size, friction and bond springs.
- **Energy:** every metabolic term, bond energy sharing, size-gated non-kin predation with
  proportional draining, and eating (each pellet goes to the closest particle in reach).
- **Learning:** the reward-modulated Hebbian update (per-particle eta gene, baseline, decay, clip).
- **Bonds:** form and break rules, using the pre-move distances and the pre-break free slots,
  as in Python.
- **Reproduction:** births (random parent order, nearest compatible mate, mate energy
  contribution, generation = max + 1), neuron-wise crossover, self-adaptive mutation, budding,
  abiogenesis, and corpses becoming food.
- **Environment:** wandering patches, the climate cycle, and pheromone deposit, diffusion and
  decay.
- **Order of operations:** senses, forces, predation and bond decisions all use the state at
  the start of the step. Eating and scent deposits use the post-move positions.

Deliberate deviations:

| What | Python | Web | Why |
|---|---|---|---|
| World size, food, patches, population floor, pheromone res | 1.4, 700, 5, 80, 96 | ×√12 / ×12 (4.85, 8400, 60, 960, 332) | same densities, ~12× the population |
| capacity / initial particles | 2000 / 500 | 8400 / 3000 | headroom for the larger world; capacity never binds in either |
| neighbour search | all-pairs or dense grid tensors | CSR lists from a counting-sort grid; ring search for nearest food | O(n) per step instead of O(n·K) padding |
| `tanh` activation | exact | Padé 7/6 approximant, \|error\| < 1e-4 | ~2.5× faster than `Math.tanh` in V8 |
| pheromone | exact float | values < 1e-7 flushed to 0 | avoids float32 denormal slowdowns |
| RNG | torch Philox/MT streams | sfc32 + Box–Muller | same distributions, so runs are reproducible per seed but not bit-identical to Python |
| state | float32 tensors | Float32Array SoA | same precision |
| lineage tree (lineage.py) | full parent/mate record | only uid/generation | not needed for the live view |
| meteor | radius 0.08, 40 food | radius 0.15, 60 food | reads better at the larger scale |

The energy economy did not need retuning. At 12x the web world settles at densities
comparable to the reference run. Python at step 1000 had pop 261 with 76% of food slots full;
the web build at 12x had pop 2616 with 77% full. Over a few thousand steps the population climbs
to about 4–4.6k as foraging evolves, well above the 960 abiogenesis floor.

## Performance notes

- One simulation step at ~4k particles takes about 20–30 ms of single-threaded JS on a laptop
  CPU (i7-12650H, measured while the machine was heavily loaded by other jobs). The rough
  breakdown:

  | Part of the step | Share |
  |---|---|
  | Brain forward pass | ~30% |
  | Plasticity update | ~20% |
  | Sensing | ~15% |
  | Neighbour build | ~10% |
  | Nearest-food search | ~8% |
  | Pheromone diffusion | ~5% |

  Everything is plain typed-array loops with no per-step allocation.
- Measured in Chromium 149 on that machine with ~3.7k particles. Worker mode gave 60 fps
  render and 17–35 sim steps/s (24–45 ms/step), dropping to 4–7 steps/s whenever the OS
  starved the worker thread. Main-thread fallback gave 29 fps and 29 steps/s (31 ms/step).
- The step budget (28 ms per frame) keeps frames flowing at any speed setting. Rendering stays
  at display rate (60 fps measured) because it runs on the main thread, independent of the sim.
- Main-thread fallback: some OS/browser combinations deprioritise worker threads heavily.
  We saw 5x slower worker threads under load on Windows. At startup the channel runs a 100 ms
  multiply-add benchmark in both the worker and the main thread. If the worker is below 45% of
  main-thread speed, the host runs on the main thread with a 10 ms budget per frame. Override
  with `?thread=worker` or `?thread=main`.
- Ideas for more speed: a WASM/SIMD brain kernel, and skipping plasticity decay for idle
  synapses. Splitting particles across several workers would need SharedArrayBuffer, which
  needs cross-origin isolation headers that a single hosted file can't rely on.

## Display design

The page is styled as an analysis tool, not a game:

- **Themes:** light by default, with a dark theme under Display → Theme. You can also force it
  with `?theme=dark`; the choice is remembered in localStorage when available.
- **Particles:** flat filled circles with a thin darker outline; the outline turns red while
  the particle is attacking.
- **Colour:** muted palettes. Lineage mode maps the hue of the genome colour to mid
  saturation and lightness. Species mode uses Tableau 10 and physical type uses ColorBrewer
  Dark2.
- **Layers:** all are alpha-blended, with no additive glow. Bonds are thin grey lines, food is
  small grey-green dots and fertile patches are faint discs.
- **Off by default:** the pheromone field (a single-hue heatmap of both channels), the signal
  display (a ring whose opacity follows |signal|) and trails.
- **Explanations:** every statistic and control has a plain-language tooltip. A persistent
  legend explains the marks, and the header links to `./explainer.html`, which is a separate
  page not included in this build.
