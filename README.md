# primordial

Particles with tiny neural-network brains that evolve through survival and reproduction.
Nobody tells them what to do: there's no fitness function and no training loop. Particles that
find food, avoid being eaten, and stay alive long enough to split pass their (slightly mutated)
brains on. The rest don't.

It's a mix of two ideas:

- **Neuroevolution / artificial life**: every particle carries its own MLP brain as its genome.
- **Particle Life physics**: every particle is also pushed around by attraction/repulsion forces
  toward other particle *types*. Those attraction values are part of the genome too, so evolution
  can tune how lineages clump, chase, or avoid each other.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows   (source .venv/bin/activate on macOS/Linux)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"

python -m primordial view                  # watch it live
python -m primordial run --steps 20000 --log stats.csv --save run.pt   # headless
python -m primordial view --load run.pt    # watch an evolved world
pytest                                     # tests
```

Override any value in `primordial/config.py` from the command line:

```bash
python -m primordial view --set attack_rate=0.08 food_regrow_prob=0.002 capacity=2000
```

### Viewer keys

| Key | Action |
|-----|--------|
| `SPACE` | pause |
| `UP` / `DOWN` | double / halve simulation steps per frame |
| `C` | toggle colouring: **lineage** (inherited colour) vs. physical **type** |
| `T` | toggle motion trails |
| `P` | toggle food-patch outlines |
| `H` | hide / show the HUD |
| left-click | inspect a particle (generation, energy, evolved mutation sizes, memory, attraction); it's followed until it dies |
| right-click | drop a burst of food |
| `S` | save a checkpoint |
| `Q` / `Esc` | quit |

Circle size shows energy. A red ring marks a particle that is currently attacking. The graph in the
corner plots population and the number of distinct lineages.

## How it works

```
             ┌────────── sense ──────────┐
 world ───►  │ food direction + closeness│
             │ kin direction             │──► MLP brain ──► thrust (x, y)
             │ stranger direction        │    (13→12→5)     aggression
             │ crowding, energy, velocity│        ▲         memory ──┐
             └───────────────────────────┘        └──────────────────┘ (fed back next step)
                                                   │
 particle-life forces (genome.attraction) ──► + ───┴──► move on a torus
```

Each step (`World.step` in `primordial/world.py`):

1. **Sense**: 11 numbers from the surroundings, plus 2 memory values the brain wrote last step.
2. **Think**: all brains run at once as one batched `einsum`.
3. **Physics**: particle-life forces plus thrust, with friction, on a wrap-around world.
4. **Metabolism**: living, moving, and being aggressive all cost energy.
5. **Predation**: aggressive particles drain energy from touching *non-kin*. Kinship is how close
   the two particles' inherited colours are, so families evolve to cooperate and strangers to fight.
6. **Eating**: food goes to the nearest particle within reach. Most food regrows inside a few
   fertile patches that slowly wander around the world ("seasons"), so foragers have to keep tracking them.
7. **Death**: when energy or lifespan runs out. The corpse becomes food.
8. **Birth**: above `reproduce_threshold`, a particle splits. The child gets a mutated copy of the genome.

If the population ever drops below `min_population`, fresh random particles are spawned, so a
run never ends in permanent extinction.

### Genome

| Field | Shape | Meaning |
|-------|-------|---------|
| `w1, b1, w2, b2` | MLP weights | the brain |
| `attraction` | `n_types` | how much this particle is pulled toward each physical type |
| `ptype` | int | its own physical type (rarely mutates) |
| `color` | RGB | kinship marker; drifts slowly so lineages stay visible |
| `sigma` | 3 | mutation step size for perception / behaviour / physics. These evolve too |

### Performance notes

All particles live in fixed-capacity tensors with an `alive` mask, so births and deaths never
reallocate memory. Interactions are computed all-pairs (O(N²)), which is fine up to a few
thousand particles. Past that, a spatial hash or grid is the next step. Pass
`--set device=cuda` if you have a GPU build of PyTorch.

## What to look for

- **Early (first few thousand steps):** random jitter. Most particles starve, and the
  random-spawn fallback keeps the world populated.
- **Then:** lineages that steer toward food take over, and the max generation number starts climbing.
- **Later:** watch `mean_aggression` and the red rings. Depending on settings you get predator
  waves, colour-uniform swarms (kin cooperation), or clumps shaped by particle-life forces.

`stats.csv` from a headless run has population, generations, aggression, and colour diversity over time.

## Mutation strategy

`mutate()` in `primordial/genome.py` stacks three mechanisms on every birth:

1. **Self-adaptive step sizes.** Each particle carries `sigma` (one value per group: perception,
   behaviour, physics). A child first perturbs its sigmas log-normally (`sigma_tau`), then uses
   them to mutate its weights. Good step sizes "hitch-hike" with the good weights they produce,
   so evolution tunes its own learning rate.
2. **Per-layer rates.** Perception (input layer) mutates at `0.5×` the base rate, behaviour
   (output layer) at `1.5×`, physics at `1×`.
3. **Neuron resets.** With probability `neuron_reset_prob`, one whole hidden neuron is
   re-randomised. Most of these jumps are harmful, but they can escape local optima that
   small nudges never leave.

Watch `sigma_*` in the HUD or the CSV. Under steady selection the sigmas usually shrink
(careful offspring survive), and they rise again when the environment shifts.

To get the original fixed-step behaviour for comparison:

```bash
--set sigma_tau=0 neuron_reset_prob=0 perception_rate_mult=1 behaviour_rate_mult=1 attraction_rate_mult=1
```

### Measuring it

```bash
python -m primordial run --steps 15000 --save run.pt
python scripts/compete.py run.pt              # evolved vs random brains
python scripts/compete.py new.pt old.pt       # two evolved populations head-to-head
bash scripts/experiment.sh                    # full old-vs-new mutation experiment, 2 seeds
```

### Results so far (2 seeds, honest version)

| | seed 1 | seed 2 |
|---|---|---|
| new vs old mutation, after 5k steps | new wins **5/5** | new wins **0/5** |
| new vs old mutation, after 15k steps | new wins **5/5** | new wins **2/5** |
| either one vs random brains, after 5k | wins 3/3 | wins 3/3 |

**The new mutation scheme is not yet proven better.** Results depend heavily on the seed, and
two seeds are too few to decide. The more interesting finding is what the sigmas evolved to on
their own. In both seeds `sigma_physics` *rose* from 0.2 to about 0.5, so evolution wants to
re-tune particle-life attraction aggressively. Behaviour and perception sigmas stayed at or
below 0.2, and in one seed behaviour dropped to 0.08 (it settled on careful fine-tuning).

The world changes, wandering food patches and stronger colour drift, raised end-of-run colour
diversity from 0.09 (one lineage takes over) to 0.15–0.17 (5–8 coexisting lineages), whichever
mutation scheme is used.

## Ideas for extending

- Let particles emit and sense a signal channel, so communication can evolve.
- Add a spatial grid to scale to 10k+ particles.
- Track lineages (parent IDs) and render a phylogenetic tree.
- Evolve the brain's *size* (NEAT-style topology growth).
