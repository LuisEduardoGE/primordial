"""Head-to-head contests between evolved populations.

A fresh arena world is seeded with `n_each` particles cloned from population A and
`n_each` from population B (or random brains). Each side is tagged with one colour
(red / blue) so kinship keeps the sides apart, and every particle's side is decided
exactly by tracing its lineage back to a founder. The side with more survivors wins.

    python scripts/compete.py run.pt                 # evolved vs random brains
    python scripts/compete.py new.pt old.pt          # two evolved populations
"""

from __future__ import annotations

import argparse
import math

import torch

from .config import Config
from .world import World

RED = (1.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


def clone_into(w: World, slots: torch.Tensor, source: World) -> None:
    """Overwrite the genomes in `slots` with genomes sampled from `source`'s living particles."""
    donors = source.alive.nonzero().squeeze(1)
    if len(donors) == 0:
        raise ValueError("source population is extinct")
    pick = donors[torch.randint(0, len(donors), (len(slots),), generator=w.gen)]
    src = source.genome.tensors()
    for name, t in w.genome.tensors().items():
        t[slots] = src[name][pick].to(t.device)


def arena_config(trial_seed: int, n_each: int, overrides: dict | None = None) -> Config:
    """World settings for a contest: shared world overrides, no abiogenesis."""
    cfg = Config(seed=trial_seed)
    for k, v in (overrides or {}).items():
        cfg.override([f"{k}={v}"])
    cfg.initial_particles = 2 * n_each
    cfg.capacity = max(cfg.capacity, 4 * n_each)
    cfg.min_population = 0          # a side that dies out must stay dead
    return cfg


def side_counts(w: World, n_each: int) -> tuple[int, int]:
    """Survivors per side, by tracing each living particle to its founder uid."""
    founder: dict[int, int] = {}
    parent = w.lineage.parent

    def root(u: int) -> int:
        path = []
        while u not in founder and parent[u] >= 0:
            path.append(u)
            u = parent[u]
        r = founder.get(u, u)
        for v in path:
            founder[v] = r
        founder[u] = r
        return r

    n_a = n_b = 0
    for u in w.uid[w.alive].tolist():
        r = root(u)
        if r < n_each:
            n_a += 1
        elif r < 2 * n_each:
            n_b += 1
    return n_a, n_b


def head_to_head(side_a: World, side_b: World | None, trial_seed: int, steps: int,
                 n_each: int = 100, overrides: dict | None = None) -> tuple[int, int]:
    """One contest; returns (survivors of A, survivors of B). `side_b=None` means random brains."""
    w = World(arena_config(trial_seed, n_each, overrides))
    idx = w.alive.nonzero().squeeze(1)
    a_slots, b_slots = idx[:n_each], idx[n_each:2 * n_each]
    clone_into(w, a_slots, side_a)
    if side_b is not None:
        clone_into(w, b_slots, side_b)
    w.genome.color[a_slots] = torch.tensor(RED, device=w.device)
    w.genome.color[b_slots] = torch.tensor(BLUE, device=w.device)
    for _ in range(steps):
        w.step()
        if w.population == 0:
            break
    return side_counts(w, n_each)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial (sign) test of wins vs losses under p = 0.5; ties are dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Head-to-head contest between two checkpoints.")
    p.add_argument("a", help="checkpoint for side A (red)")
    p.add_argument("b", nargs="?", help="checkpoint for side B (blue); omit for random brains")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--each", type=int, default=100, help="particles per side")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="arena config overrides")
    args = p.parse_args(argv)

    side_a = World.load(args.a)
    side_b = World.load(args.b) if args.b else None
    overrides = dict(item.split("=", 1) for item in args.set)
    label_b = "B" if side_b else "random"
    wins = losses = 0
    for trial in range(args.trials):
        n_a, n_b = head_to_head(side_a, side_b, 100 + trial, args.steps, args.each, overrides)
        wins += n_a > n_b
        losses += n_b > n_a
        print(f"trial {trial}: A {n_a:>4}   {label_b} {n_b:>4}", flush=True)
    print(f"A won {wins}/{args.trials} (lost {losses}); sign test p = {sign_test(wins, losses):.3g}")
