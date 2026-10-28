"""Head-to-head: do evolved brains beat random ones, or each other?

    python scripts/compete.py run.pt                 # evolved vs random brains
    python scripts/compete.py new.pt old.pt          # two evolved populations

Seeds a fresh world with 100 particles cloned from population A (tagged red) and
100 from population B, or random (tagged blue), runs it, and counts survivors of each side.
"""

import argparse

import torch

from primordial import Config, World


def clone_into(w: World, slots: torch.Tensor, source: World) -> None:
    donors = source.alive.nonzero().squeeze(1)
    pick = donors[torch.randint(0, len(donors), (len(slots),), generator=w.gen)]
    for name, t in w.genome.tensors().items():
        t[slots] = source.genome.tensors()[name][pick]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a", help="checkpoint for side A (red)")
    p.add_argument("b", nargs="?", help="checkpoint for side B (blue); omit for random brains")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--steps", type=int, default=3000)
    args = p.parse_args()

    side_a = World.load(args.a)
    side_b = World.load(args.b) if args.b else None
    red, blue = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])
    label_b = "B" if side_b else "random"
    wins = 0

    for trial in range(args.trials):
        w = World(Config(seed=100 + trial, initial_particles=200, min_population=0))
        idx = w.alive.nonzero().squeeze(1)
        clone_into(w, idx[:100], side_a)
        if side_b:
            clone_into(w, idx[100:], side_b)
        w.genome.color[idx[:100]] = red
        w.genome.color[idx[100:]] = blue
        for _ in range(args.steps):
            w.step()
        c = w.genome.color[w.alive]
        n_a = int((c[:, 0] > c[:, 2]).sum())
        n_b = len(c) - n_a
        wins += n_a > n_b
        print(f"trial {trial}: A {n_a:>4}   {label_b} {n_b:>4}")

    print(f"A won {wins}/{args.trials}")


if __name__ == "__main__":
    main()
