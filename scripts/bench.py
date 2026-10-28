"""Steps/second from a fixed mid-run state, best of several repeats (robust to background load).

    python scripts/bench.py [--pop 1500] [--steps 200] [--repeats 3]
"""

import argparse
import copy
import os
import tempfile
import time

from primordial import Config, World


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pop", type=int, default=0, help="initial particles (0 = evolve 1200 steps from defaults)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    if args.pop:
        base = World(Config(seed=1, initial_particles=args.pop, capacity=max(2000, args.pop * 2)))
        base.step()
    else:
        base = World(Config(seed=1))
        for _ in range(1200):
            base.step()
    path = os.path.join(tempfile.gettempdir(), "primordial_bench.pt")
    base.save(path)

    best = 0.0
    for _ in range(args.repeats):
        w = World.load(path)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            w.step()
        best = max(best, args.steps / (time.perf_counter() - t0))
    print(f"population {base.population} -> {w.population}   best {best:.1f} steps/s")


if __name__ == "__main__":
    main()
