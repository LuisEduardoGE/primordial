# primordial

An artificial-life simulation in which particles with small neural-network brains and evolvable
bodies forage, hunt, signal, bond into multi-particle organisms, mate and die. There is no fitness
function and no training data. Behaviour comes from selection alone: particles that gather energy
reproduce, and their mutated offspring inherit what worked.

New to the idea? Read the plain-language guide in [`docs/explainer.html`](docs/explainer.html).

There are two implementations of the same ecosystem:

| | Python / PyTorch (`primordial/`) | Browser / TypeScript + three.js (`web/`) |
|---|---|---|
| Purpose | research: experiments, discovery, recording | sharing and large-scale viewing |
| Scale | ~300 particles at ~35 steps/s, ~1,600 at ~20 steps/s | ~2,500–4,500 particles |
| Extras | family tree, brain visualiser, timelapse, experiment reports | runs anywhere, single HTML file |

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows   (source .venv/bin/activate on macOS/Linux)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]" matplotlib imageio imageio-ffmpeg

python -m primordial view                                  # interactive viewer
python -m primordial run --steps 20000 --log stats.csv --save run.pt
python -m primordial timelapse --load run.pt --steps 5000 --out evolution.mp4
python -m primordial experiment experiments/mutation.json --workers 8
python -m primordial discover --evals 24 --workers 6
python scripts/compete.py run.pt [other.pt]                # head-to-head contest
pytest
```

Any value in `primordial/config.py` can be overridden: `--set attack_rate=0.06 food_patches=8`.

Browser version: `cd web && npm install && npm run dev` (or `npm run build` for a single
self-contained `web/dist/index.html`). See [`web/README.md`](web/README.md).

## The model

Each simulation step (`World.step` in `primordial/world.py`):

1. **Sense**: 23 readings plus 2 memory values. They cover the nearest food, the direction of kin and of strangers, crowding, own energy and velocity, neighbours' signals, two pheromone gradients, bonds in use, the relative size of nearby strangers, and the season.
2. **Think**: a recurrent MLP (25 → 16 → 9), whose output layer carries reward-modulated Hebbian weights learned within each lifetime. All brains run as one batched `einsum`.
3. **Act**: thrust, aggression, signal, two pheromone deposits, bond intent, and 2 memory outputs.
4. **Physics**: particle-life attraction by physical type, plus bond springs, plus thrust divided by body size, with friction, on a torus.
5. **Metabolism**: base cost proportional to size^0.75, upkeep for speed and senses, and costs for movement, attack, signalling, scent and each bond.
6. **Predation**: aggressive particles drain non-kin they touch, but only victims no bigger than 1/0.8 of their own size.
7. **Eating**: food goes to the closest particle in reach. Food regrows mostly in wandering fertile patches, at a rate that follows a seasonal cycle.
8. **Learning**: plastic weights update from reward, which is the change in energy relative to the particle's own running baseline.
9. **Bonds**: kin form a bond when both want to and they are touching. Bonds break when strained or no longer wanted. Bonded particles share energy.
10. **Death and birth**: corpses become food. A particle above its energy threshold splits. If a compatible mate is near, the child gets neuron-wise crossover of both parents' genomes, and some children stay bonded to the parent (budding).

The genome holds the brain weights, particle-life attractions, body genes (size, speed, sense), a budding probability, a learning rate, a kinship colour, and four self-adaptive mutation step sizes.

Mutation is plain fixed-step Gaussian noise by default. Three optional mechanisms can be switched on: self-adaptive step sizes (`sigma_tau=0.2`), per-layer rates (`perception_rate_mult=0.5 behaviour_rate_mult=1.5`), and whole-neuron resets (`neuron_reset_prob=0.05`). They were the default until the experiment below found no benefit.

Analysis runs alongside the simulation:
- Every particle has a lineage ID, so the family tree of the living population can be extracted (`lineage.py`).
- Particles are clustered by body and behaviour with k-means and named automatically (`species.py`: grazer, hunter, colonial, signaller, trail-layer, drifter, scout).

## Viewer (`python -m primordial view`)

Click a particle to inspect its body, diet, evolved mutation sizes and a live diagram of its neural
network. Key bindings: SPACE pause · UP/DOWN speed · 1/2/3 inspect/add food/meteor tool ·
`[` `]` climate · C colour mode (lineage/type/species) · T trails · O pheromones · B bonds ·
P patches · F family tree · G parameter sliders · R record video · S save · H hide HUD · Q quit.

## Results so far

- **Evolved vs random brains:** evolved populations drove random-brained ones extinct in 5/5 contests. This was with the earlier engine.
- **Self-adaptive mutation vs fixed mutation** (`experiments/mutation.json`): 8 seeds per variant, 6,000 steps, then head-to-head contests. The new scheme is not shown to be better. No end-of-run metric differs after Holm correction. In contests, fixed mutation won the majority in 6 of 8 seeds (per-seed sign test p = 0.29). The physics step size is the one sigma that consistently evolves upward (0.20 → 0.29).
- **World discovery** (MAP-Elites, 24 evaluations): found 9 elite worlds. The most interesting combine abundant food, strong short-period seasons, a high kin threshold and strong energy sharing along bonds. The archive (`runs/discovery/archive.json`) holds a `--set` command line to reproduce each one.

## Performance notes

- Neighbour search uses a uniform grid on the torus, built with tensor operations only. The candidate lists are then compacted to the largest number of in-range neighbours, which matters because populations clump.
- x and y coordinates are kept as separate arrays in the hot path. Broadcasting over a trailing dimension of size 2 was about 15× slower on CPU.
- In multi-process runs, use one torch thread per worker.

## Layout

```
primordial/   config, genome, world, spatial, lineage, species       engine
              render, viewer, timelapse                               visuals
              experiments, discovery, tournament                      research tools
web/          TypeScript + three.js port (sim in a worker, single-file build)
docs/         plain-language explainer
experiments/  experiment specs
scripts/      compete, bench
tests/        pytest suite
```
