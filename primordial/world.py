"""The simulation: a torus full of particles with evolving brains.

Each step:
  1. sense      every particle builds an input vector from its surroundings
  2. think      its neural network turns that into thrust + aggression
  3. physics    particle-life forces + thrust move everything
  4. metabolism energy is spent on living, moving and attacking
  5. predation  aggressive particles drain non-kin they touch
  6. eating     food under a particle becomes its energy
  7. death      out of energy or too old -> slot freed, corpse becomes food
  8. birth      energy above threshold -> split into a mutated child
"""

import math

import torch

from .config import Config
from .genome import N_ACTIONS, N_MEMORY, Genome, mutate, think


def particle_life_force(r: torch.Tensor, a: torch.Tensor, beta: float) -> torch.Tensor:
    """Classic particle-life kernel on normalised distance r in [0, 1).

    Below `beta` everything repels (keeps particles from overlapping);
    between `beta` and 1 a triangular bump scaled by attraction `a`.
    """
    repulse = r / beta - 1
    attract = a * (1 - (2 * r - 1 - beta).abs() / (1 - beta))
    return torch.where(r < beta, repulse, torch.where(r < 1, attract, torch.zeros_like(r)))


class World:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        c, d = cfg.capacity, self.device

        self.pos = torch.zeros(c, 2, device=d)
        self.vel = torch.zeros(c, 2, device=d)
        self.energy = torch.zeros(c, device=d)
        self.age = torch.zeros(c, dtype=torch.long, device=d)
        self.generation = torch.zeros(c, dtype=torch.long, device=d)
        self.alive = torch.zeros(c, dtype=torch.bool, device=d)
        self.aggression = torch.zeros(c, device=d)  # last output, kept for rendering
        self.memory = torch.zeros(c, N_MEMORY, device=d)  # recurrent brain state
        self.genome = Genome.empty(cfg)

        self.food = self._rand(cfg.food_capacity, 2) * cfg.world_size
        self.food_alive = torch.ones(cfg.food_capacity, dtype=torch.bool, device=d)
        self.patch_pos = self._rand(cfg.food_patches, 2) * cfg.world_size
        self.patch_heading = self._rand(cfg.food_patches) * 2 * math.pi

        self.step_count = 0
        self.births = 0
        self.deaths = 0
        self.spawn(cfg.initial_particles)

    # ------------------------------------------------------------------ helpers

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def _wrap_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Shortest displacement on the torus."""
        w = self.cfg.world_size
        return delta - torch.round(delta / w) * w

    def spawn(self, n: int) -> None:
        """Abiogenesis: drop `n` brand-new random particles into free slots."""
        idx = (~self.alive).nonzero().squeeze(1)[:n]
        if len(idx) == 0:
            return
        k = len(idx)
        self.pos[idx] = self._rand(k, 2) * self.cfg.world_size
        self.vel[idx] = 0
        self.energy[idx] = self.cfg.initial_energy
        self.age[idx] = 0
        self.generation[idx] = 0
        self.aggression[idx] = 0
        self.memory[idx] = 0
        self.alive[idx] = True
        self.genome.randomize(idx, self.cfg, self.gen)

    @property
    def population(self) -> int:
        return int(self.alive.sum())

    # --------------------------------------------------------------------- step

    @torch.no_grad()
    def step(self) -> None:
        cfg = self.cfg
        idx = self.alive.nonzero().squeeze(1)
        if len(idx) > 0:
            self._update_living(idx)
        self._deaths()
        self._births()
        self._move_patches()
        self._regrow_food()
        if self.population < cfg.min_population:
            self.spawn(cfg.min_population - self.population)
        self.step_count += 1

    def _update_living(self, idx: torch.Tensor) -> None:
        cfg = self.cfg
        pos, vel, energy = self.pos[idx], self.vel[idx], self.energy[idx]
        g = self.genome.take(idx)

        # Pairwise geometry. delta[i, j] points from i to j.
        delta = self._wrap_delta(pos[None, :, :] - pos[:, None, :])
        dist = delta.norm(dim=-1)
        dist.fill_diagonal_(math.inf)
        unit = delta / dist.clamp_min(1e-9)[..., None]
        near = dist < cfg.interaction_radius

        # Kinship: 1 for identical colour, -> 0 as lineages diverge.
        sim = torch.exp(-((g.color[:, None] - g.color[None, :]) ** 2).sum(-1) / cfg.kin_sigma**2)

        # 1. sense + 2. think
        inputs = torch.cat([self._sense(pos, vel, energy, unit, near, sim), self.memory[idx]], dim=1)
        out = think(g, inputs)
        thrust = out[:, :2] * cfg.max_thrust
        aggression = out[:, 2].clamp_min(0)
        self.memory[idx] = out[:, N_ACTIONS:]

        # 3. physics: particle i feels attraction[i, type[j]] toward each neighbour j
        a = g.attraction[:, g.ptype]
        f = particle_life_force(dist / cfg.interaction_radius, a, cfg.repulsion_beta) * near
        acc = (f[..., None] * unit).sum(1) * cfg.interaction_radius * cfg.force_factor
        friction = 0.5 ** (cfg.dt / cfg.friction_half_life)
        vel = vel * friction + (acc + thrust) * cfg.dt
        pos = (pos + vel * cfg.dt) % cfg.world_size

        # 4. metabolism
        energy = energy - (
            cfg.base_metabolism
            + cfg.move_cost * (thrust**2).sum(1)
            + cfg.aggression_cost * aggression
        )

        # 5. predation: attackers drain touching non-kin, capped by the victim's reserves
        contact = (dist < cfg.contact_radius) & (sim < cfg.kin_threshold)
        drain = contact * aggression[:, None] * cfg.attack_rate
        demanded = drain.sum(0)
        available = energy.clamp_min(0)
        drain = drain * (available / demanded.clamp_min(1e-9)).clamp(max=1)[None, :]
        energy = energy - drain.sum(0) + cfg.attack_efficiency * drain.sum(1)

        # 6. eating: each food item goes to the nearest particle within reach
        if self.food_alive.any():
            fd = self._wrap_delta(self.food[None, :, :] - pos[:, None, :]).norm(dim=-1)
            fd[:, ~self.food_alive] = math.inf
            nearest_dist, eater = fd.min(0)
            eaten = nearest_dist < cfg.eat_radius
            energy = energy.index_add(0, eater[eaten], torch.full_like(nearest_dist[eaten], cfg.food_energy))
            self.food_alive &= ~eaten

        self.pos[idx], self.vel[idx], self.energy[idx] = pos, vel, energy
        self.aggression[idx] = aggression
        self.age[idx] += 1

    def _sense(self, pos, vel, energy, unit, near, sim) -> torch.Tensor:
        """Build the (n, N_SENSES) sensory vector; memory is appended by the caller.

        [0:2] direction to nearest food, scaled by closeness
        [2]   closeness of nearest food (0 = out of sense range, 1 = touching)
        [3:5] average direction toward nearby kin
        [5:7] average direction toward nearby strangers
        [7]   crowding (neighbour count / 10)
        [8]   own energy relative to reproduction threshold
        [9:11] own velocity
        """
        cfg = self.cfg
        n = len(pos)
        food_vec = torch.zeros(n, 2, device=self.device)
        closeness = torch.zeros(n, device=self.device)
        if self.food_alive.any():
            food = self.food[self.food_alive]
            fdelta = self._wrap_delta(food[None, :, :] - pos[:, None, :])
            fdist = fdelta.norm(dim=-1)
            d_min, j = fdist.min(1)
            closeness = (1 - d_min / cfg.food_sense_radius).clamp_min(0)
            food_vec = fdelta[torch.arange(n, device=self.device), j] / d_min.clamp_min(1e-9)[:, None]
            food_vec = food_vec * closeness[:, None]

        w = near.float()
        count = w.sum(1, keepdim=True).clamp_min(1)
        kin_dir = ((w * sim)[..., None] * unit).sum(1) / count
        stranger_dir = ((w * (1 - sim))[..., None] * unit).sum(1) / count

        return torch.cat(
            [
                food_vec,
                closeness[:, None],
                kin_dir,
                stranger_dir,
                w.sum(1, keepdim=True) / 10,
                (energy / cfg.reproduce_threshold)[:, None],
                vel * 5,
            ],
            dim=1,
        )

    # ------------------------------------------------------- population changes

    def _deaths(self) -> None:
        dead = self.alive & ((self.energy <= 0) | (self.age > self.cfg.max_age))
        n_dead = int(dead.sum())
        if n_dead == 0:
            return
        self.alive &= ~dead
        self.deaths += n_dead
        # Corpses fertilise the ground: they become food where empty slots exist.
        slots = (~self.food_alive).nonzero().squeeze(1)
        corpses = dead.nonzero().squeeze(1)[: len(slots)]
        slots = slots[: len(corpses)]
        self.food[slots] = self.pos[corpses]
        self.food_alive[slots] = True

    def _births(self) -> None:
        cfg = self.cfg
        parents = (self.alive & (self.energy >= cfg.reproduce_threshold)).nonzero().squeeze(1)
        free = (~self.alive).nonzero().squeeze(1)
        k = min(len(parents), len(free))
        if k == 0:
            return
        parents = parents[torch.randperm(len(parents), generator=self.gen, device=self.device)[:k]]
        children = free[:k]

        share = self.energy[parents] * (1 - cfg.birth_cost) / 2
        self.energy[parents] = share
        self.energy[children] = share
        self.pos[children] = (self.pos[parents] + self._randn(k, 2) * 0.01) % cfg.world_size
        self.vel[children] = 0
        self.age[children] = 0
        self.generation[children] = self.generation[parents] + 1
        self.aggression[children] = 0
        self.memory[children] = 0
        self.alive[children] = True
        self.genome.copy(children, parents)
        mutate(self.genome, children, cfg, self.gen)
        self.births += k

    def _regrow_food(self) -> None:
        cfg = self.cfg
        empty = ~self.food_alive
        sprout = empty & (self._rand(cfg.food_capacity) < cfg.food_regrow_prob)
        n = int(sprout.sum())
        if n == 0:
            return
        new = self._rand(n, 2) * cfg.world_size
        # Most food sprouts inside the wandering fertile patches, so foragers must track them.
        if cfg.food_patches > 0:
            clump = self._rand(n) < cfg.food_clumping
            anchors = self.patch_pos[torch.randint(0, cfg.food_patches, (n,), generator=self.gen, device=self.device)]
            in_patch = (anchors + self._randn(n, 2) * cfg.patch_radius) % cfg.world_size
            new = torch.where(clump[:, None], in_patch, new)
        self.food[sprout] = new
        self.food_alive |= sprout

    def drop_food(self, x: float, y: float, n: int = 12, spread: float = 0.02) -> int:
        """Scatter up to `n` food items around (x, y). Returns how many were placed."""
        slots = (~self.food_alive).nonzero().squeeze(1)[:n]
        k = len(slots)
        if k:
            centre = torch.tensor([x, y], device=self.device)
            self.food[slots] = (centre + self._randn(k, 2) * spread) % self.cfg.world_size
            self.food_alive[slots] = True
        return k

    def _move_patches(self) -> None:
        """Fertile zones wander: a slowly turning heading, constant speed."""
        cfg = self.cfg
        if cfg.food_patches == 0:
            return
        self.patch_heading += self._randn(cfg.food_patches) * 0.05
        step = torch.stack([self.patch_heading.cos(), self.patch_heading.sin()], dim=1) * cfg.patch_speed
        self.patch_pos = (self.patch_pos + step) % cfg.world_size

    # -------------------------------------------------------------- inspection

    def stats(self) -> dict:
        a = self.alive
        pop = int(a.sum())
        if pop == 0:
            return {"step": self.step_count, "population": 0}
        return {
            "step": self.step_count,
            "population": pop,
            "food": int(self.food_alive.sum()),
            "births": self.births,
            "deaths": self.deaths,
            "mean_energy": round(float(self.energy[a].mean()), 4),
            "mean_generation": round(float(self.generation[a].float().mean()), 2),
            "max_generation": int(self.generation[a].max()),
            "mean_aggression": round(float(self.aggression[a].mean()), 4),
            "color_diversity": round(float(self.genome.color[a].std(0).mean()), 4),
            "lineages": self.lineage_count(),
            "sigma_perception": round(float(self.genome.sigma[a, 0].mean()), 4),
            "sigma_behaviour": round(float(self.genome.sigma[a, 1].mean()), 4),
            "sigma_physics": round(float(self.genome.sigma[a, 2].mean()), 4),
        }

    def lineage_count(self, bins: int = 4) -> int:
        """Rough species count: occupied cells when colour space is cut into bins^3 cubes,
        counting only cells holding at least 5% of the population."""
        c = self.genome.color[self.alive]
        if len(c) == 0:
            return 0
        cell = (c * bins).long().clamp(max=bins - 1)
        keys = cell[:, 0] * bins * bins + cell[:, 1] * bins + cell[:, 2]
        counts = torch.bincount(keys, minlength=bins**3)
        return int((counts >= max(1, 0.05 * len(c))).sum())

    # ------------------------------------------------------------ checkpoints

    STATE = ("pos", "vel", "energy", "age", "generation", "alive", "aggression", "memory",
             "food", "food_alive", "patch_pos", "patch_heading")

    def save(self, path: str) -> None:
        from dataclasses import asdict

        torch.save(
            {
                "config": asdict(self.cfg),
                "state": {k: getattr(self, k) for k in self.STATE},
                "genome": self.genome.tensors(),
                "counters": (self.step_count, self.births, self.deaths),
                "rng": self.gen.get_state(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "World":
        data = torch.load(path, map_location=device)
        cfg = Config(**{**data["config"], "device": device})
        world = cls.__new__(cls)
        world.cfg = cfg
        world.device = torch.device(device)
        world.gen = torch.Generator(device=world.device)
        world.gen.set_state(data["rng"])
        for k, v in data["state"].items():
            setattr(world, k, v)
        world.genome = Genome(**data["genome"])
        world.step_count, world.births, world.deaths = data["counters"]
        return world
