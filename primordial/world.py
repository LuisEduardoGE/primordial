"""The simulation: a torus full of particles with evolving bodies and brains.

Each step:
  1. sense       food, kin, strangers, signals, scents, bonds, threat, season, memory
  2. think       recurrent + plastic neural network -> thrust, attack, signal, scent, bond
  3. physics     particle-life forces + bond springs + thrust move everything
  4. metabolism  living, body upkeep, moving, attacking, signalling, scenting, bonds
  5. predation   bigger, aggressive particles drain touching non-kin
  6. eating      food under a particle becomes its energy
  7. learning    reward = energy change vs. own baseline; Hebbian update of plastic weights
  8. bonds       kin that both want to link up bond; strained or unwanted bonds snap
  9. death       out of energy or too old -> slot freed, corpse becomes food
 10. birth       energy above threshold -> child (with a mate's genes, if one is near)
 11. world       food patches wander, food regrows with the climate, scents diffuse
"""

import math

import torch

from .config import Config
from .genome import (AGGRESSION, BOND, DEPOSIT, MEMORY, N_INPUTS, N_MEMORY, N_OUTPUTS, N_SENSES,
                     SIGNAL, THRUST, Genome, crossover, mutate, think)
from .lineage import Lineage
from .spatial import neighbours, wrap
from .species import SpeciesTracker, features

N_EMA = 5  # behaviour averages: moving, attacking, signalling, scenting, bonds


def kinship(cq: torch.Tensor, ct: torch.Tensor, cand: torch.Tensor, sigma: float) -> torch.Tensor:
    """Colour similarity in (0, 1] between each query and its candidates (per channel: fast on CPU)."""
    d2 = sum((cq[:, c].contiguous()[:, None] - ct[:, c].contiguous()[cand]) ** 2 for c in range(3))
    return torch.exp(-d2 / sigma**2)


def particle_life_force(r: torch.Tensor, a: torch.Tensor, beta: float) -> torch.Tensor:
    """Classic particle-life kernel on normalised distance r in [0, 1).

    Below `beta` everything repels (keeps particles from overlapping);
    between `beta` and 1 a triangular bump scaled by attraction `a`.
    """
    repulse = r / beta - 1
    attract = a * (1 - (2 * r - 1 - beta).abs() / (1 - beta))
    return torch.where(r < beta, repulse, torch.where(r < 1, attract, torch.zeros_like(r)))


class World:
    STATE = ("pos", "vel", "energy", "age", "generation", "alive", "uid", "aggression", "signal",
             "memory", "plastic", "reward_avg", "bonds", "ema", "food_gained", "prey_gained",
             "species_id", "last_inputs", "last_hidden", "last_outputs",
             "food", "food_alive", "patch_pos", "patch_heading", "pheromone")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        c, d, h = cfg.capacity, self.device, cfg.hidden

        def z(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=d)

        self.pos, self.vel = z(c, 2), z(c, 2)
        self.energy = z(c)
        self.age = z(c, dtype=torch.long)
        self.generation = z(c, dtype=torch.long)
        self.alive = z(c, dtype=torch.bool)
        self.uid = torch.full((c,), -1, dtype=torch.long, device=d)
        self.aggression, self.signal = z(c), z(c)       # last outputs, sensed by others / rendered
        self.memory = z(c, N_MEMORY)                     # recurrent brain state
        self.plastic = z(c, h, N_OUTPUTS)                # lifetime-learned weight offsets
        self.reward_avg = z(c)                           # per-particle reward baseline
        self.bonds = torch.full((c, cfg.max_bonds), -1, dtype=torch.long, device=d)
        self.ema = z(c, N_EMA)
        self.food_gained, self.prey_gained = z(c), z(c)
        self.species_id = torch.full((c,), -1, dtype=torch.long, device=d)
        self.last_inputs, self.last_hidden, self.last_outputs = z(c, N_INPUTS), z(c, h), z(c, N_OUTPUTS)
        self.genome = Genome.empty(cfg)

        self.food = self._rand(cfg.food_capacity, 2) * cfg.world_size
        self.food_alive = torch.ones(cfg.food_capacity, dtype=torch.bool, device=d)
        self.patch_pos = self._rand(cfg.food_patches, 2) * cfg.world_size
        self.patch_heading = self._rand(cfg.food_patches) * 2 * math.pi
        self.pheromone = z(2, cfg.pheromone_res, cfg.pheromone_res)
        self.food_multiplier = 1.0                       # god-tool climate override

        self.lineage = Lineage()
        self.species = SpeciesTracker(cfg.species_k, cfg.seed)
        self.step_count = 0
        self.births = self.deaths = self.mated_births = self.bonds_formed = 0
        self.spawn(cfg.initial_particles)

    # ------------------------------------------------------------------ helpers

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def _wrap_delta(self, delta: torch.Tensor) -> torch.Tensor:
        return wrap(delta, self.cfg.world_size)

    def body(self, idx=None):
        """(size, speed, sense) multipliers for particles `idx` (default: all slots)."""
        b = self.genome.body if idx is None else self.genome.body[idx]
        return self.cfg.size(b[:, 0]), self.cfg.speed(b[:, 1]), self.cfg.sense(b[:, 2])

    def season(self) -> float:
        p = self.cfg.climate_period
        return math.sin(2 * math.pi * self.step_count / p) if p > 0 else 0.0

    @property
    def population(self) -> int:
        return int(self.alive.sum())

    def _reset_slots(self, idx: torch.Tensor) -> None:
        """Clear per-life state for slots about to hold a new particle."""
        for t in (self.vel, self.age, self.aggression, self.signal, self.memory, self.plastic,
                  self.reward_avg, self.ema, self.food_gained, self.prey_gained,
                  self.last_inputs, self.last_hidden, self.last_outputs):
            t[idx] = 0
        self.bonds[idx] = -1
        self.species_id[idx] = -1

    def _new_uids(self, idx, parent_uids, mate_uids) -> None:
        sizes = self.body(idx)[0].tolist()
        uids = self.lineage.register(parent_uids, mate_uids, self.step_count,
                                     self.genome.color[idx].tolist(), sizes)
        self.uid[idx] = torch.tensor(uids, dtype=torch.long, device=self.device)

    def spawn(self, n: int) -> None:
        """Abiogenesis: drop `n` brand-new random particles into free slots."""
        idx = (~self.alive).nonzero().squeeze(1)[:n]
        k = len(idx)
        if k == 0:
            return
        self._reset_slots(idx)
        self.pos[idx] = self._rand(k, 2) * self.cfg.world_size
        self.energy[idx] = self.cfg.initial_energy
        self.generation[idx] = 0
        self.alive[idx] = True
        self.genome.randomize(idx, self.cfg, self.gen)
        self._new_uids(idx, [-1] * k, [-1] * k)

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
        self._diffuse_pheromone()
        if self.population < cfg.min_population:
            self.spawn(cfg.min_population - self.population)
        self.step_count += 1
        if cfg.species_interval and self.step_count % cfg.species_interval == 0:
            self.classify_species()

    def _update_living(self, idx: torch.Tensor) -> None:
        cfg = self.cfg
        n, W = len(idx), cfg.world_size
        pos, vel, energy = self.pos[idx], self.vel[idx], self.energy[idx]
        g = self.genome.take(idx)
        size, speed, sense = self.body(idx)
        local = torch.full((cfg.capacity,), -1, dtype=torch.long, device=self.device)
        local[idx] = torch.arange(n, device=self.device)

        # Neighbourhood (grid-accelerated). Everything pairwise is (n, K) from here on.
        radius = max(cfg.interaction_radius, cfg.sense_radius * cfg.sense_max)
        nb = neighbours(pos, pos, radius, W, exclude_self=True)
        cand, dist = nb.cand, nb.dist
        inv = 1 / dist.clamp_min(1e-9)                              # zero where dist is inf
        ux, uy = nb.dx * inv, nb.dy * inv
        csize = size[cand]
        sim = kinship(g.color, g.color, cand, cfg.kin_sigma)
        seen = dist < (cfg.sense_radius * sense)[:, None]

        # Bonded partners (global slots -> local rows).
        partners = self.bonds[idx]
        pl = local[partners.clamp_min(0)]
        bvalid = (partners >= 0) & (pl >= 0)
        pl = pl.clamp_min(0)
        px, py = pos[:, 0].contiguous(), pos[:, 1].contiguous()
        bdx = self._wrap_delta(px[pl] - px[:, None])
        bdy = self._wrap_delta(py[pl] - py[:, None])
        bdist = torch.sqrt(bdx * bdx + bdy * bdy)
        binv = 1 / bdist.clamp_min(1e-9)
        bond_count = bvalid.sum(1)

        # 1. sense + 2. think
        senses = self._sense(idx, pos, vel, energy, size, sense, nb, ux, uy, sim, seen, csize, bond_count)
        inputs = torch.cat([senses, self.memory[idx]], dim=1)
        hidden, out = think(g, inputs, self.plastic[idx])
        thrust = out[:, THRUST] * cfg.max_thrust * speed[:, None]
        aggression = out[:, AGGRESSION].clamp_min(0)
        signal = out[:, SIGNAL]
        deposit = out[:, DEPOSIT].clamp_min(0)
        bond_intent = out[:, BOND]

        # 3. physics: particle-life forces, bond springs, thrust (heavier bodies accelerate less)
        a = torch.gather(g.attraction, 1, g.ptype[cand])
        near = dist < cfg.interaction_radius
        f = particle_life_force(dist / cfg.interaction_radius, a, cfg.repulsion_beta) * near
        rest = cfg.bond_rest * (size[:, None] + size[pl]) / 2
        spring = cfg.bond_strength * (bdist - rest) * bvalid * binv
        k = cfg.interaction_radius * cfg.force_factor
        acc = torch.stack([(f * ux).sum(1) * k + (spring * bdx).sum(1),
                           (f * uy).sum(1) * k + (spring * bdy).sum(1)], dim=1)
        acc = acc + thrust / size[:, None]
        friction = 0.5 ** (cfg.dt / cfg.friction_half_life)
        vel = vel * friction + acc * cfg.dt
        pos = (pos + vel * cfg.dt) % W

        # 4. metabolism
        energy_before = energy
        energy = energy - (
            cfg.base_metabolism * size**0.75
            + cfg.speed_upkeep * speed
            + cfg.sense_upkeep * sense
            + cfg.move_cost * (out[:, THRUST] ** 2).sum(1) * speed**2 * size
            + cfg.aggression_cost * aggression
            + cfg.signal_cost * signal.abs()
            + cfg.deposit_cost * deposit.sum(1)
            + cfg.bond_cost * bond_count
        )

        # Bonded organisms share energy: flow along each bond toward the poorer side.
        if bvalid.any():
            flow = cfg.bond_share * (energy[pl] - energy[:, None]) / 2 * bvalid
            energy = energy + flow.sum(1)

        # 5. predation: big-enough, aggressive particles drain touching non-kin
        contact = ((dist < cfg.contact_radius * (size[:, None] + csize) / 2)
                   & (sim < cfg.kin_threshold) & (size[:, None] >= cfg.prey_size_ratio * csize))
        drain = contact * (aggression * size * cfg.attack_rate)[:, None]
        demanded = torch.zeros(n, device=self.device).index_add_(0, cand.reshape(-1), drain.reshape(-1))
        scale = (energy.clamp_min(0) / demanded.clamp_min(1e-9)).clamp(max=1)
        taken = drain * scale[cand]
        lost = torch.zeros(n, device=self.device).index_add_(0, cand.reshape(-1), taken.reshape(-1))
        gained = cfg.attack_efficiency * taken.sum(1)
        energy = energy - lost + gained
        self.prey_gained[idx] += gained

        # 6. eating
        food_gain = self._eat(pos, size)
        energy = energy + food_gain
        self.food_gained[idx] += food_gain

        # 7. lifetime learning: reward-modulated Hebbian update, with a per-particle baseline
        delta_e = energy - energy_before
        if cfg.plasticity > 0:
            m = torch.tanh(cfg.reward_scale * (delta_e - self.reward_avg[idx]))
            eta = (10 ** g.log_eta) * cfg.plasticity
            p = self.plastic[idx] * (1 - cfg.plastic_decay)
            p = p + (eta * m)[:, None, None] * hidden[:, :, None] * out[:, None, :]
            self.plastic[idx] = p.clamp(-cfg.plastic_clip, cfg.plastic_clip)
        self.reward_avg[idx] = 0.95 * self.reward_avg[idx] + 0.05 * delta_e

        # Scent: deposit into the pheromone grid cell under each particle.
        if deposit.any():
            r = cfg.pheromone_res
            cell = (pos / W * r).long().clamp(0, r - 1)
            flat = cell[:, 0] * r + cell[:, 1]
            self.pheromone.view(2, -1).index_add_(1, flat, (deposit * cfg.pheromone_deposit).T.contiguous())

        # Write back.
        self.pos[idx], self.vel[idx], self.energy[idx] = pos, vel, energy
        self.aggression[idx], self.signal[idx] = aggression, signal
        self.memory[idx] = out[:, MEMORY]
        self.last_inputs[idx], self.last_hidden[idx], self.last_outputs[idx] = inputs, hidden, out
        behaviour = torch.stack([out[:, THRUST].norm(dim=1), aggression, signal.abs(),
                                 deposit.sum(1), bond_count.float()], dim=1)
        self.ema[idx] = 0.98 * self.ema[idx] + 0.02 * behaviour
        self.age[idx] += 1

        # 8. bonds
        self._update_bonds(idx, bond_intent, nb, sim, size, csize, partners, bvalid, bdist)

    def _sense(self, idx, pos, vel, energy, size, sense, nb, ux, uy, sim, seen, csize, bond_count):
        """Build the (n, N_SENSES) sensory vector; memory is appended by the caller.

        [0:3]   nearest food: direction * closeness, closeness
        [3:7]   average direction toward nearby kin, toward nearby strangers
        [7]     crowding          [8] energy vs. reproduction threshold   [9:11] own velocity
        [11:14] neighbours' signals: mean level, direction toward the loudest
        [14:20] two scent channels: gradient x, gradient y, level
        [20]    bonds in use      [21] threat: log size ratio of nearby strangers
        [22]    season (climate phase)
        """
        cfg = self.cfg
        n, dev = len(pos), self.device

        food_vec = torch.zeros(n, 2, device=dev)
        closeness = torch.zeros(n, device=dev)
        fidx = self.food_alive.nonzero().squeeze(1)
        if len(fidx):
            fr = cfg.food_sense_radius * sense
            fnb = neighbours(pos, self.food[fidx], cfg.food_sense_radius * cfg.sense_max, cfg.world_size)
            fd = fnb.dist.masked_fill(fnb.dist > fr[:, None], math.inf)
            if fd.shape[1]:
                d_min, j = fd.min(1)
                closeness = (1 - d_min / fr).clamp_min(0)
                ar = torch.arange(n, device=dev)
                direction = torch.stack([fnb.dx[ar, j], fnb.dy[ar, j]], dim=1) / d_min.clamp_min(1e-9)[:, None]
                food_vec = torch.nan_to_num(direction, 0.0, 0.0, 0.0) * closeness[:, None]

        def toward(weights):
            return torch.stack([(weights * ux).sum(1), (weights * uy).sum(1)], dim=1)

        w = seen.float()
        count = w.sum(1, keepdim=True).clamp_min(1)
        kin_dir = toward(w * sim) / count
        stranger_w = w * (1 - sim)
        stranger_dir = toward(stranger_w) / count

        prox = (1 - nb.dist / (cfg.sense_radius * sense)[:, None]).clamp_min(0) * w
        sig = self.signal[idx][nb.cand] * prox
        mean_signal = sig.sum(1) / prox.sum(1).clamp_min(1e-6)
        signal_dir = toward(sig) / count

        r = cfg.pheromone_res
        cell = (pos / cfg.world_size * r).long().clamp(0, r - 1)
        cx, cy = cell[:, 0], cell[:, 1]
        ph = self.pheromone
        level = ph[:, cx, cy]
        gx = ph[:, (cx + 1) % r, cy] - ph[:, (cx - 1) % r, cy]
        gy = ph[:, cx, (cy + 1) % r] - ph[:, cx, (cy - 1) % r]
        scent = torch.stack([torch.tanh(gx * 5), torch.tanh(gy * 5), torch.tanh(level)], dim=2)  # (2, n, 3)
        scent = scent.permute(1, 0, 2).reshape(n, 6)

        threat = (stranger_w * torch.log(csize / size[:, None])).sum(1) / stranger_w.sum(1).clamp_min(1e-6)

        return torch.cat([
            food_vec, closeness[:, None], kin_dir, stranger_dir,
            w.sum(1, keepdim=True) / 10,
            (energy / (cfg.reproduce_threshold * size))[:, None],
            vel * 5,
            mean_signal[:, None], signal_dir,
            scent,
            (bond_count.float() / cfg.max_bonds)[:, None],
            threat[:, None],
            torch.full((n, 1), self.season(), device=dev),
        ], dim=1)

    def _eat(self, pos: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        """Each food item within reach goes to its single closest particle."""
        cfg = self.cfg
        n = len(pos)
        gain = torch.zeros(n, device=self.device)
        fidx = self.food_alive.nonzero().squeeze(1)
        if len(fidx) == 0:
            return gain
        enb = neighbours(pos, self.food[fidx], cfg.eat_radius * cfg.size_max, cfg.world_size)
        ok = enb.dist < (cfg.eat_radius * size)[:, None]
        if not ok.any():
            return gain
        p, k = ok.nonzero(as_tuple=True)
        f = enb.cand[p, k]
        d = enb.dist[p, k]
        best = torch.full((len(fidx),), math.inf, device=self.device).scatter_reduce(0, f, d, "amin")
        win = d == best[f]
        p, f = p[win], f[win]
        order = torch.arange(len(f), device=self.device)
        first = torch.full((len(fidx),), len(f), device=self.device, dtype=torch.long).scatter_reduce(
            0, f, order, "amin")
        keep = order == first[f]                                   # break exact ties
        gain.index_add_(0, p[keep], torch.full((int(keep.sum()),), cfg.food_energy, device=self.device))
        self.food_alive[fidx[f[keep]]] = False
        return gain

    def _update_bonds(self, idx, intent, nb, sim, size, csize, partners, bvalid, bdist) -> None:
        cfg = self.cfg
        changed = False
        bonds = None

        # Break: overstretched, or either side no longer wants it.
        local = torch.full((cfg.capacity,), -1, dtype=torch.long, device=self.device)
        local[idx] = torch.arange(len(idx), device=self.device)
        partner_intent = intent[local[partners.clamp_min(0)].clamp_min(0)]
        snap = bvalid & ((bdist > cfg.bond_break) | (intent[:, None] < -0.5) | (partner_intent < -0.5))
        if snap.any():
            bonds = self.bonds.tolist()
            rows, cols = snap.nonzero(as_tuple=True)
            for a, b in zip(idx[rows].tolist(), partners[rows, cols].tolist()):
                for x, y in ((a, b), (b, a)):
                    if y in bonds[x]:
                        bonds[x][bonds[x].index(y)] = -1
            changed = True

        # Form: both sides want it, touching, kin, free slots, not already bonded.
        want = intent > 0.5
        free = (self.bonds[idx] < 0).any(1)
        touch = ((nb.dist < cfg.contact_radius * (size[:, None] + csize)) & (sim > cfg.kin_threshold)
                 & want[:, None] & want[nb.cand] & free[:, None] & free[nb.cand])
        if touch.any():
            d = nb.dist.masked_fill(~touch, math.inf)
            d_min, j = d.min(1)
            has = torch.isfinite(d_min)
            a_list = idx[has].tolist()
            b_list = idx[nb.cand[has, j[has]]].tolist()
            if bonds is None:
                bonds = self.bonds.tolist()
            for a, b in zip(a_list, b_list):
                if b in bonds[a] or -1 not in bonds[a] or -1 not in bonds[b]:
                    continue
                bonds[a][bonds[a].index(-1)] = b
                bonds[b][bonds[b].index(-1)] = a
                self.bonds_formed += 1
            changed = True

        if changed:
            self.bonds = torch.tensor(bonds, dtype=torch.long, device=self.device)

    # ------------------------------------------------------- population changes

    def kill(self, dead: torch.Tensor) -> None:
        """Remove particles (bool mask over slots); corpses become food where there's room."""
        dead = dead & self.alive
        n_dead = int(dead.sum())
        if n_dead == 0:
            return
        slots_dead = dead.nonzero().squeeze(1)
        self.alive &= ~dead
        self.deaths += n_dead
        self.lineage.record_deaths(self.uid[slots_dead].tolist(), self.step_count)
        self.bonds[slots_dead] = -1
        self.bonds[torch.isin(self.bonds, slots_dead)] = -1
        free_food = (~self.food_alive).nonzero().squeeze(1)
        corpses = slots_dead[: len(free_food)]
        free_food = free_food[: len(corpses)]
        self.food[free_food] = self.pos[corpses]
        self.food_alive[free_food] = True

    def _deaths(self) -> None:
        self.kill(self.alive & ((self.energy <= 0) | (self.age > self.cfg.max_age)))

    def _births(self) -> None:
        cfg = self.cfg
        size = self.body()[0]
        parents = (self.alive & (self.energy >= cfg.reproduce_threshold * size)).nonzero().squeeze(1)
        free = (~self.alive).nonzero().squeeze(1)
        k = min(len(parents), len(free))
        if k == 0:
            return
        parents = parents[torch.randperm(len(parents), generator=self.gen, device=self.device)[:k]]
        children = free[:k]

        # Find a compatible mate near each parent.
        mates = torch.full((k,), -1, dtype=torch.long, device=self.device)
        if cfg.mate_prob > 0:
            living = self.alive.nonzero().squeeze(1)
            mnb = neighbours(self.pos[parents], self.pos[living], cfg.mate_radius, cfg.world_size)
            mslot = living[mnb.cand]
            msim = kinship(self.genome.color[parents], self.genome.color, mslot, cfg.kin_sigma)
            ok = ((mnb.dist < cfg.mate_radius) & (mslot != parents[:, None]) & (msim > cfg.mate_similarity))
            d = mnb.dist.masked_fill(~ok, math.inf)
            if d.shape[1]:
                d_min, j = d.min(1)
                chosen = torch.isfinite(d_min) & (self._rand(k) < cfg.mate_prob)
                mates = torch.where(chosen, mslot[torch.arange(k, device=self.device), j], mates)
        mated = mates >= 0

        share = self.energy[parents] * (1 - cfg.birth_cost) / 2
        gift = torch.zeros(k, device=self.device)
        if mated.any():
            gift[mated] = self.energy[mates[mated]] * cfg.mate_contribution
            self.energy.index_add_(0, mates[mated], -gift[mated])

        self._reset_slots(children)
        self.species_id[children] = self.species_id[parents]   # until the next clustering pass
        self.energy[parents] = share
        self.energy[children] = share + gift
        self.pos[children] = (self.pos[parents] + self._randn(k, 2) * 0.01) % cfg.world_size
        self.generation[children] = torch.maximum(self.generation[parents],
                                                  torch.where(mated, self.generation[mates.clamp_min(0)], 0)) + 1
        self.alive[children] = True
        bud = self._rand(k) < self.genome.bud[parents]
        self.genome.copy(children, parents)
        crossover(self.genome, children[mated], mates[mated], self.gen)
        mutate(self.genome, children, cfg, self.gen)

        mate_uids = torch.where(mated, self.uid[mates.clamp_min(0)], -1).tolist()
        self._new_uids(children, self.uid[parents].tolist(), mate_uids)
        self.births += k
        self.mated_births += int(mated.sum())

        # Budding: some children stay attached to their parent, growing multi-particle bodies.
        if bud.any():
            bonds = self.bonds.tolist()
            for p, c in zip(parents[bud].tolist(), children[bud].tolist()):
                if -1 in bonds[p]:
                    bonds[p][bonds[p].index(-1)] = c
                    bonds[c][0] = p
                    self.bonds_formed += 1
            self.bonds = torch.tensor(bonds, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------- world

    def _regrow_food(self) -> None:
        cfg = self.cfg
        rate = cfg.food_regrow_prob * (1 + cfg.climate_amplitude * self.season()) * self.food_multiplier
        empty = ~self.food_alive
        sprout = empty & (self._rand(cfg.food_capacity) < rate)
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

    def _move_patches(self) -> None:
        """Fertile zones wander: a slowly turning heading, constant speed."""
        cfg = self.cfg
        if cfg.food_patches == 0:
            return
        self.patch_heading += self._randn(cfg.food_patches) * 0.05
        step = torch.stack([self.patch_heading.cos(), self.patch_heading.sin()], dim=1) * cfg.patch_speed
        self.patch_pos = (self.patch_pos + step) % cfg.world_size

    def _diffuse_pheromone(self) -> None:
        cfg = self.cfg
        p = self.pheromone
        if not p.any():
            return
        blur = (p.roll(1, 1) + p.roll(-1, 1) + p.roll(1, 2) + p.roll(-1, 2)) / 4
        self.pheromone = ((1 - cfg.pheromone_diffusion) * p + cfg.pheromone_diffusion * blur) * (1 - cfg.pheromone_decay)

    # ------------------------------------------------------------- god tools

    def drop_food(self, x: float, y: float, n: int = 12, spread: float = 0.02) -> int:
        """Scatter up to `n` food items around (x, y). Returns how many were placed."""
        slots = (~self.food_alive).nonzero().squeeze(1)[:n]
        k = len(slots)
        if k:
            centre = torch.tensor([x, y], device=self.device)
            self.food[slots] = (centre + self._randn(k, 2) * spread) % self.cfg.world_size
            self.food_alive[slots] = True
        return k

    def meteor(self, x: float, y: float, radius: float = 0.08) -> int:
        """Kill every particle within `radius` of (x, y); the crater fills with food. Returns kills."""
        d = self._wrap_delta(self.pos - torch.tensor([x, y], device=self.device)).norm(dim=1)
        hit = self.alive & (d < radius)
        n = int(hit.sum())
        self.kill(hit)
        self.drop_food(x, y, n=40, spread=radius / 2)
        return n

    # -------------------------------------------------------------- analysis

    def classify_species(self) -> None:
        idx = self.alive.nonzero().squeeze(1)
        self.species_id[:] = -1
        if len(idx) < self.cfg.species_k:
            return
        labels = self.species.update(features(self, idx))
        self.species_id[idx] = labels.to(self.device)

    def species_counts(self) -> dict[str, int]:
        ids = self.species_id[self.alive]
        ids = ids[ids >= 0]
        if len(ids) == 0 or not self.species.names:
            return {}
        counts = torch.bincount(ids, minlength=len(self.species.names)).tolist()
        return {name: c for name, c in zip(self.species.names, counts) if c > 0}

    def lineage_count(self, bins: int = 4) -> float:
        """Effective number of families: exp(Shannon entropy) of the population over colour
        cells (bins^3 cubes). Equals k for k equal-sized families; robust at any population."""
        c = self.genome.color[self.alive]
        if len(c) == 0:
            return 0.0
        cell = (c * bins).long().clamp(max=bins - 1)
        keys = cell[:, 0] * bins * bins + cell[:, 1] * bins + cell[:, 2]
        p = torch.bincount(keys, minlength=bins**3).float() / len(c)
        p = p[p > 0]
        return round(float(torch.exp(-(p * p.log()).sum())), 1)

    def stats(self) -> dict:
        a = self.alive
        pop = int(a.sum())
        if pop == 0:
            return {"step": self.step_count, "population": 0}
        size, speed, sense = self.body()
        g = self.genome
        bonded = (self.bonds[a] >= 0).any(1).float().mean()
        diet = self.prey_gained[a].sum() / (self.prey_gained[a].sum() + self.food_gained[a].sum() + 1e-6)
        r = lambda x, k=4: round(float(x), k)  # noqa: E731
        return {
            "step": self.step_count,
            "population": pop,
            "food": int(self.food_alive.sum()),
            "births": self.births,
            "deaths": self.deaths,
            "mated_births": self.mated_births,
            "bonds_formed": self.bonds_formed,
            "mean_energy": r(self.energy[a].mean()),
            "mean_generation": r(self.generation[a].float().mean(), 2),
            "max_generation": int(self.generation[a].max()),
            "mean_aggression": r(self.aggression[a].mean()),
            "mean_signal": r(self.signal[a].abs().mean()),
            "predation_share": r(diet),
            "bonded_fraction": r(bonded),
            "mean_size": r(size[a].mean()),
            "mean_speed": r(speed[a].mean()),
            "mean_sense": r(sense[a].mean()),
            "mean_log_eta": r(g.log_eta[a].mean()),
            "color_diversity": r(g.color[a].std(0).mean()),
            "lineages": self.lineage_count(),
            "season": r(self.season(), 3),
            "sigma_perception": r(g.sigma[a, 0].mean()),
            "sigma_behaviour": r(g.sigma[a, 1].mean()),
            "sigma_physics": r(g.sigma[a, 2].mean()),
            "sigma_traits": r(g.sigma[a, 3].mean()),
            "species": "|".join(f"{k}:{v}" for k, v in self.species_counts().items()),
        }

    # ------------------------------------------------------------ checkpoints

    def save(self, path: str) -> None:
        from dataclasses import asdict

        torch.save(
            {
                "config": asdict(self.cfg),
                "state": {k: getattr(self, k) for k in self.STATE},
                "genome": self.genome.tensors(),
                "counters": [self.step_count, self.births, self.deaths, self.mated_births, self.bonds_formed],
                "food_multiplier": self.food_multiplier,
                "lineage": self.lineage.state_dict(),
                "species": self.species.state_dict(),
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
        (world.step_count, world.births, world.deaths, world.mated_births, world.bonds_formed) = data["counters"]
        world.food_multiplier = data["food_multiplier"]
        world.lineage = Lineage.from_state(data["lineage"])
        world.species = SpeciesTracker(cfg.species_k, cfg.seed)
        world.species.load_state(data["species"])
        return world
