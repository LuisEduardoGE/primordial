"""Heritable traits: a tiny recurrent neural network brain plus physical traits.

Every field is a tensor with a leading `capacity` dimension, so slot `i`
holds the genome of particle `i`. All brains run in one batched einsum.
"""

from dataclasses import dataclass, fields

import torch

from .config import Config

N_MEMORY = 2                  # recurrent channels: outputs fed back as next step's inputs
N_SENSES = 11                 # see World._sense for the layout
N_INPUTS = N_SENSES + N_MEMORY
N_ACTIONS = 3                 # thrust x, thrust y, aggression
N_OUTPUTS = N_ACTIONS + N_MEMORY

# Mutation groups, each with its own evolvable step size (genome.sigma[:, group]).
PERCEPTION, BEHAVIOUR, PHYSICS = 0, 1, 2
GROUPS = {PERCEPTION: ("w1", "b1"), BEHAVIOUR: ("w2", "b2"), PHYSICS: ("attraction",)}


@dataclass
class Genome:
    w1: torch.Tensor          # (C, N_INPUTS, hidden)
    b1: torch.Tensor          # (C, hidden)
    w2: torch.Tensor          # (C, hidden, N_OUTPUTS)
    b2: torch.Tensor          # (C, N_OUTPUTS)
    attraction: torch.Tensor  # (C, n_types)  how strongly this particle is pulled toward each type
    color: torch.Tensor       # (C, 3)        kinship marker, drifts slowly down lineages
    ptype: torch.Tensor       # (C,) long     physical type, used by others' attraction rows
    sigma: torch.Tensor       # (C, 3)        mutation step size per group, itself evolving

    @classmethod
    def empty(cls, cfg: Config) -> "Genome":
        c, h, d = cfg.capacity, cfg.hidden, torch.device(cfg.device)
        return cls(
            w1=torch.zeros(c, N_INPUTS, h, device=d),
            b1=torch.zeros(c, h, device=d),
            w2=torch.zeros(c, h, N_OUTPUTS, device=d),
            b2=torch.zeros(c, N_OUTPUTS, device=d),
            attraction=torch.zeros(c, cfg.n_types, device=d),
            color=torch.zeros(c, 3, device=d),
            ptype=torch.zeros(c, dtype=torch.long, device=d),
            sigma=torch.zeros(c, 3, device=d),
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

        self.w1[idx] = randn(N_INPUTS, cfg.hidden) / N_INPUTS**0.5
        self.b1[idx] = randn(cfg.hidden) * 0.1
        self.w2[idx] = randn(cfg.hidden, N_OUTPUTS) / cfg.hidden**0.5
        self.b2[idx] = randn(N_OUTPUTS) * 0.1
        self.attraction[idx] = torch.rand(n, cfg.n_types, generator=gen, device=d) * 2 - 1
        self.color[idx] = torch.rand(n, 3, generator=gen, device=d)
        self.ptype[idx] = torch.randint(0, cfg.n_types, (n,), generator=gen, device=d)
        self.sigma[idx] = cfg.mutation_scale


def think(g: Genome, inputs: torch.Tensor) -> torch.Tensor:
    """Run every brain at once. inputs (n, N_INPUTS) -> outputs (n, N_OUTPUTS) in [-1, 1]."""
    h = torch.tanh(torch.einsum("ni,nih->nh", inputs, g.w1) + g.b1)
    return torch.tanh(torch.einsum("nh,nho->no", h, g.w2) + g.b2)


def mutate(g: Genome, idx: torch.Tensor, cfg: Config, gen: torch.Generator) -> None:
    """Mutate the freshly-copied genomes of newborn particles at slots `idx`.

    Three mechanisms stack:
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
        g.sigma[idx] = (g.sigma[idx] * torch.exp(cfg.sigma_tau * randn(n, 3))).clamp(cfg.sigma_min, cfg.sigma_max)
    sigma = g.sigma[idx]

    # 2. Per-layer rates, each group scaled by its own evolved sigma.
    rates = {
        PERCEPTION: cfg.mutation_rate * cfg.perception_rate_mult,
        BEHAVIOUR: cfg.mutation_rate * cfg.behaviour_rate_mult,
        PHYSICS: cfg.mutation_rate * cfg.attraction_rate_mult,
    }
    for group, names in GROUPS.items():
        for name in names:
            t = getattr(g, name)
            values = t[idx]
            scale = sigma[:, group].view(n, *([1] * (values.dim() - 1)))
            mask = rand(*values.shape) < rates[group]
            t[idx] = values + mask * randn(*values.shape) * scale
    g.attraction[idx] = g.attraction[idx].clamp(-1, 1)

    # 3. Sparse, large jumps: wipe and redraw one hidden neuron's in/out weights.
    reset = idx[rand(n) < cfg.neuron_reset_prob]
    k = len(reset)
    if k:
        hidden = g.w1.shape[-1]
        h = torch.randint(0, hidden, (k,), generator=gen, device=d)
        g.w1[reset, :, h] = randn(k, N_INPUTS) / N_INPUTS**0.5
        g.b1[reset, h] = randn(k) * 0.1
        g.w2[reset, h, :] = randn(k, N_OUTPUTS) / hidden**0.5
