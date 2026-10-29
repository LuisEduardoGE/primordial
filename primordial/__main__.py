"""Command line: `python -m primordial {view,run} ...`"""

import argparse
import csv
import importlib
import os
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # keep CLI output clean

from .config import Config
from .world import World


def build_world(args) -> World:
    if args.load:
        world = World.load(args.load)
        world.cfg.override(args.set)
        return world
    cfg = Config(seed=args.seed).override(args.set)
    return World(cfg)


def cmd_view(args) -> None:
    from .viewer import run_viewer

    run_viewer(build_world(args), size=args.size, steps_per_frame=args.speed, save_path=args.save)


def cmd_run(args) -> None:
    world = build_world(args)
    writer, log_file = None, None
    if args.log:
        log_file = open(args.log, "w", newline="")
    t0 = time.perf_counter()
    try:
        for i in range(1, args.steps + 1):
            world.step()
            if i % args.every == 0 or i == args.steps:
                s = world.stats()
                if log_file:
                    if writer is None:
                        writer = csv.DictWriter(log_file, fieldnames=list(s))
                        writer.writeheader()
                    writer.writerow(s)
                rate = i / (time.perf_counter() - t0)
                print("  ".join(f"{k}={v}" for k, v in s.items()) + f"  ({rate:.0f} steps/s)", flush=True)
    finally:
        if log_file:
            log_file.close()
    if args.save:
        world.save(args.save)
        print(f"saved {args.save}")


def main() -> None:
    p = argparse.ArgumentParser(prog="primordial", description="Particles with evolving neural brains.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--load", help="resume from a checkpoint")
        sp.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")

    v = sub.add_parser("view", help="watch the simulation live")
    common(v)
    v.add_argument("--size", type=int, default=820)
    v.add_argument("--speed", type=int, default=2, help="simulation steps per frame")
    v.add_argument("--save", default="checkpoint.pt", help="where the S key saves to")
    v.set_defaults(func=cmd_view)

    r = sub.add_parser("run", help="run headless and log stats")
    common(r)
    r.add_argument("--steps", type=int, default=10_000)
    r.add_argument("--every", type=int, default=500, help="stats interval")
    r.add_argument("--log", help="CSV file for stats")
    r.add_argument("--save", help="checkpoint to write at the end")
    r.set_defaults(func=cmd_run)

    # Optional tools register their own subcommands (timelapse, experiment, discover).
    for module in ("timelapse", "experiments", "discovery"):
        try:
            mod = importlib.import_module(f".{module}", __package__)
        except ImportError:
            continue
        mod.add_subparser(sub)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
