"""Heritable traits: a recurrent, plastic neural-network brain plus body traits.

Every field is a tensor with a leading `capacity` dimension, so slot `i`
holds the genome of particle `i`. All brains run in one batched einsum.
"""

from dataclasses import dataclass, fields

import torch

from .config import Config

N_SENSES = 23   # see World._sense for the layout
N_MEMORY = 2    # recurrent channels: outputs fed back as next step's inputs
N_INPUTS = N_SENSES + N_MEMORY

# Output layout
THRUST = slice(0, 2)
AGGRESSION = 2
SIGNAL = 3
DEPOSIT = slice(4, 6)
BOND = 6
N_ACTIONS = 7
MEMORY = slice(N_ACTIONS, N_ACTIONS + N_MEMORY)
N_OUTPUTS = N_ACTIONS + N_MEMORY

SENSE_NAMES = [
    "food dx", "food dy", "food near", "kin dx", "kin dy", "stranger dx", "stranger dy",
    "crowding", "energy", "vel x", "vel y", "signal", "signal dx", "signal dy",
    "scent A dx", "scent A dy", "scent A", "scent B dx", "scent B dy", "scent B",
    "bonds", "threat", "season", "memory 1", "memory 2",
]
OUTPUT_NAMES = ["thrust x", "thrust y", "attack", "signal", "drop A", "drop B", "bond", "memory 1", "memory 2"]

# Mutation groups, each with its own evolvable step size (genome.sigma[:, group]).
PERCEPTION, BEHAVIOUR, PHYSICS, TRAITS = 0, 1, 2, 3
GROUPS = {PERCEPTION: ("w1", "b1"), BEHAVIOUR: ("w2", "b2"), PHYSICS: ("attraction",),
          TRAITS: ("body", "bud", "log_eta")}
BOUNDS = {"attraction": (-1, 1), "body": (0, 1), "bud": (0, 1), "log_eta": (-4, -0.5)}


@dataclass
class Genome:
    w1: torch.Tensor          # (C, N_INPUTS, hidden)
    b1: torch.Tensor          # (C, hidden)
    w2: torch.Tensor          # (C, hidden, N_OUTPUTS)
    b2: torch.Tensor          # (C, N_OUTPUTS)
    attraction: torch.Tensor  # (C, n_types)  pull toward each physical type
    color: torch.Tensor       # (C, 3)        kinship marker, drifts slowly down lineages
    ptype: torch.Tensor       # (C,) long     physical type, used by others' attraction rows
    sigma: torch.Tensor       # (C, 4)        mutation step size per group, itself evolving
    body: torch.Tensor        # (C, 3)        size, speed, sense genes in [0, 1]
    bud: torch.Tensor         # (C,)          chance a child stays bonded to its parent
    log_eta: torch.Tensor     # (C,)          log10 lifetime learning rate

    @classmethod
    def empty(cls, cfg: Config) -> "Genome":
        c, h, d = cfg.capacity, cfg.hidden, torch.device(cfg.device)
        z = lambda *s: torch.zeros(c, *s, device=d)  # noqa: E731
        return cls(
            w1=z(N_INPUTS, h), b1=z(h), w2=z(h, N_OUTPUTS), b2=z(N_OUTPUTS),
            attraction=z(cfg.n_types), color=z(3),
            ptype=torch.zeros(c, dtype=torch.long, device=d),
            sigma=z(len(GROUPS)), body=z(3), bud=z(), log_eta=z(),
        )

    def tensors(self) -> dict[str, torch.Tensor]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def take(self, idx: torch.Tensor) -> "Genome":
        return Genome(**{k: v[idx] for k, v in self.tensors().items()})

    def copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        for v in self.tensors().values():
            v[dst] = v[src]

    def randomize(self, idx: torch.Tensor, cfg: Config, gen: torch.Generator) -> None:
        n, d = len(idx), self.w1.device

        def randn(*shape):
            return torch.randn(n, *shape, generator=gen, device=d)

        def rand(*shape):
            return torch.rand(n, *shape, generator=gen, device=d)

        self.w1[idx] = randn(N_INPUTS, cfg.hidden) / N_INPUTS**0.5
        self.b1[idx] = randn(cfg.hidden) * 0.1
        self.w2[idx] = randn(cfg.hidden, N_OUTPUTS) / cfg.hidden**0.5
        self.b2[idx] = randn(N_OUTPUTS) * 0.1
        self.attraction[idx] = rand(cfg.n_types) * 2 - 1
        self.color[idx] = rand(3)
        self.ptype[idx] = torch.randint(0, cfg.n_types, (n,), generator=gen, device=d)
        self.sigma[idx] = cfg.mutation_scale
        self.body[idx] = 0.3 + rand(3) * 0.4   # start mid-range so evolution can push either way
        self.bud[idx] = rand() * 0.2
        self.log_eta[idx] = -3 + rand() * 1.5


def think(g: Genome, inputs: torch.Tensor, plastic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run every brain at once.

    inputs  (n, N_INPUTS); plastic (n, hidden, N_OUTPUTS) lifetime-learned weights.
    Returns (hidden activations, outputs in [-1, 1]).
    """
    h = torch.tanh(torch.einsum("ni,nih->nh", inputs, g.w1) + g.b1)
    out = torch.tanh(torch.einsum("nh,nho->no", h, g.w2 + plastic) + g.b2)
    return h, out


def crossover(g: Genome, child: torch.Tensor, mate: torch.Tensor, gen: torch.Generator) -> None:
    """Mix a mate's genes into children already holding a copy of their first parent.

    Hidden neurons are inherited as units (incoming + outgoing weights together) so
    a working feature detector is never torn in half; everything else is per-gene.
    """
    n, d = len(child), g.w1.device
    if n == 0:
        return
    hidden = g.w1.shape[-1]
    take = torch.rand(n, hidden, generator=gen, device=d) < 0.5          # per neuron
    g.w1[child] = torch.where(take[:, None, :], g.w1[mate], g.w1[child])
    g.b1[child] = torch.where(take, g.b1[mate], g.b1[child])
    g.w2[child] = torch.where(take[:, :, None], g.w2[mate], g.w2[child])
    for name in ("b2", "attraction", "sigma", "body", "bud", "log_eta", "ptype"):
        t = getattr(g, name)
        a, b = t[child], t[mate]
        mask = torch.rand(a.shape, generator=gen, device=d) < 0.5
        t[child] = torch.where(mask, b, a)
    g.color[child] = (g.color[child] + g.color[mate]) / 2


def mutate(g: Genome, idx: torch.Tensor, cfg: Config, gen: torch.Generator) -> None:
    """Mutate the freshly-copied genomes of newborn particles at slots `idx`.

    Three optional mechanisms stack on plain Gaussian mutation (all off by default,
    since experiments/mutation.json found no benefit):
      1. self-adaptation  each group's step size sigma is log-normally perturbed first,
                          then used, so good step sizes hitch-hike with good weights
      2. per-layer rates  perception mutates rarely, behaviour often
      3. neuron resets    occasionally one hidden neuron is re-randomised outright
    """
    n, d = len(idx), g.w1.device
    if n == 0:
        return

    def rand(*shape):
        return torch.rand(*shape, generator=gen, device=d)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=d)

    # Visible traits: colour drifts so lineages stay recognisable, type rarely flips.
    g.color[idx] = (g.color[idx] + randn(n, 3) * cfg.color_drift).clamp(0, 1)
    flip = rand(n) < cfg.type_flip_prob
    new_types = torch.randint(0, cfg.n_types, (n,), generator=gen, device=d)
    g.ptype[idx] = torch.where(flip, new_types, g.ptype[idx])

    # 1. Self-adaptation (mutate the mutator).
    if cfg.sigma_tau > 0:
        g.sigma[idx] = (g.sigma[idx] * torch.exp(cfg.sigma_tau * randn(n, len(GROUPS)))).clamp(
            cfg.sigma_min, cfg.sigma_max)
    sigma = g.sigma[idx]

    # 2. Per-layer rates, each group scaled by its own evolved sigma.
    rates = {
        PERCEPTION: cfg.mutation_rate * cfg.perception_rate_mult,
        BEHAVIOUR: cfg.mutation_rate * cfg.behaviour_rate_mult,
        PHYSICS: cfg.mutation_rate * cfg.attraction_rate_mult,
        TRAITS: cfg.mutation_rate * cfg.traits_rate_mult,
    }
    for group, names in GROUPS.items():
        for name in names:
            t = getattr(g, name)
            values = t[idx]
            scale = sigma[:, group].view(n, *([1] * (values.dim() - 1)))
            if group == TRAITS and name != "log_eta":
                scale = scale * 0.25   # [0, 1] genes need smaller steps than weights
            mask = rand(*values.shape) < rates[group]
            values = values + mask * randn(*values.shape) * scale
            if name in BOUNDS:
                values = values.clamp(*BOUNDS[name])
            t[idx] = values

    # 3. Sparse, large jumps: wipe and redraw one hidden neuron's in/out weights.
    reset = idx[rand(n) < cfg.neuron_reset_prob]
    k = len(reset)
    if k:
        hidden = g.w1.shape[-1]
        h = torch.randint(0, hidden, (k,), generator=gen, device=d)
        g.w1[reset, :, h] = randn(k, N_INPUTS) / N_INPUTS**0.5
        g.b1[reset, h] = randn(k) * 0.1
        g.w2[reset, h, :] = randn(k, N_OUTPUTS) / hidden**0.5
