"""World discovery: MAP-Elites over world/config parameters.

Instead of hunting for one "best" world, MAP-Elites keeps the most interesting world
found in each cell of a behaviour grid, so the archive ends up as a catalogue of
qualitatively *different* worlds that each work well.

Behaviour descriptor (the grid axes), both averaged over the last third of a run:
  x  predation_share  share of energy that came from killing (grazers -> hunters)
  y  bonded_fraction  share of particles in a bond (solitary -> colonial organisms)
These are the two main axes of ecological organisation the engine supports (what you
eat, who you live with), they are independent of how "interesting" a world is, and
both live in [0, 1]. Bins are sqrt-spaced because most worlds sit near zero on both.

Interestingness (fitness, in [0, 1]) = autonomy x weighted geometric mean of
  pop      sustained population: half mean, half minimum late population, vs 2x the floor
  gen      generational progress: 1 - exp(-g / 5), g = mean generation per 1000 steps
  species  distinct named species (hunter, grazer, colonial...) holding >= 5%, / 4
  lineage  colour lineages (among particles of generation >= 2) holding >= 5% of them, / 6
autonomy = late births / (late births + late abiogenesis spawns): a world kept alive by the
`min_population` floor scores near zero however busy it looks. The geometric mean means a
world must do all of these, not just one, to rank well.

    python -m primordial discover --evals 24 --workers 6 --steps 1500 --out runs/discovery
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .config import Config
from .experiments import jsonable, make_config, pool_map, set_args, simulate

# name: (low, high, log-scaled). Searched in a unit cube, decoded into these ranges.
PARAMS: dict[str, tuple[float, float, bool]] = {
    "food_regrow_prob": (0.001, 0.012, True),
    "food_energy": (0.3, 1.2, False),
    "attack_rate": (0.0, 0.1, False),
    "attack_efficiency": (0.2, 0.95, False),
    "patch_speed": (0.0, 0.002, False),
    "climate_amplitude": (0.0, 0.9, False),
    "climate_period": (500, 8000, True),
    "mate_prob": (0.0, 1.0, False),
    "bond_strength": (5.0, 100.0, True),
    "bond_share": (0.0, 0.2, False),
    "bond_cost": (0.0, 0.001, False),
    "kin_threshold": (0.2, 0.9, False),
    "mutation_rate": (0.02, 0.3, True),
    "pheromone_decay": (0.002, 0.05, True),
}
FIXED = {"capacity": 800, "initial_particles": 300}
AXES = ("predation_share", "bonded_fraction")
WEIGHTS = {"pop": 0.3, "gen": 0.25, "species": 0.25, "lineage": 0.2}


def decode(unit: list[float]) -> dict[str, float]:
    """Unit-cube genotype -> config values (rounded so --set lines stay readable)."""
    out = {}
    for u, (name, (lo, hi, log)) in zip(unit, PARAMS.items()):
        u = min(1.0, max(0.0, u))
        v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if log else lo + u * (hi - lo)
        out[name] = int(round(v)) if isinstance(getattr(Config, name), int) else float(f"{v:.4g}")
    return out


def encode(params: dict[str, float]) -> list[float]:
    """Config values -> unit cube (inverse of decode, for defaults/comparison)."""
    out = []
    for name, (lo, hi, log) in PARAMS.items():
        v = params[name]
        out.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if log else (v - lo) / (hi - lo))
    return out


def cell_of(desc: tuple[float, float], grid: int) -> tuple[int, int]:
    return tuple(min(grid - 1, int(math.sqrt(max(0.0, min(1.0, d))) * grid)) for d in desc)


def bin_edges(grid: int) -> list[float]:
    return [(i / grid) ** 2 for i in range(grid + 1)]


def score(rows: list[dict], steps: int, min_population: int) -> tuple[float, dict, tuple[float, float]]:
    """Interestingness, its parts, and the behaviour descriptor from a run's stats rows."""
    half = [r for r in rows if r["step"] > steps / 2] or rows[-1:]
    third = [r for r in rows if r["step"] > 2 * steps / 3] or rows[-1:]
    before = max((r for r in rows if r["step"] <= steps / 2), key=lambda r: r["step"],
                 default={"births": 0, "spawned": 0})
    last = rows[-1]

    def late(key: str) -> list[float]:
        return [float(r.get(key, 0) or 0) for r in half]

    births = last.get("births", 0) - before.get("births", 0)
    spawned = last.get("spawned", 0) - before.get("spawned", 0)
    autonomy = births / (births + spawned) if births > 0 else 0.0
    pops = late("population")
    target = 2 * max(min_population, 1)
    parts = {
        "pop": 0.5 * min(1, sum(pops) / len(pops) / target) + 0.5 * min(1, min(pops) / target),
        "gen": 1 - math.exp(-float(last.get("mean_generation", 0)) * 1000 / steps / 5),
        "species": min(1.0, sum(late("n_species")) / len(half) / 4),
        "lineage": min(1.0, sum(late("evolved_lineages")) / len(half) / 6),
    }
    geo = math.exp(sum(w * math.log(max(0.02, parts[k])) for k, w in WEIGHTS.items()) / sum(WEIGHTS.values()))
    parts["autonomy"] = autonomy
    desc = tuple(sum(float(r.get(a, 0) or 0) for r in third) / len(third) for a in AXES)
    return autonomy * geo, parts, desc


def evaluate(job: dict) -> dict:
    """Worker: run one candidate world, save its checkpoint, return its archive record."""
    from .world import World

    t0 = time.perf_counter()
    cfg = make_config(job["seed"], job["fixed"], job["params"])
    world = World(cfg)
    rows = simulate(world, job["steps"], job["every"])
    world.save(job["ckpt"])
    fitness, parts, desc = score(rows, job["steps"], cfg.min_population)
    keep = ("step", "population", "mean_generation", "spawned", "predation_share", "bonded_fraction")
    return {
        "id": job["id"], "seed": job["seed"], "steps": job["steps"], "unit": job["unit"], "params": job["params"],
        "fixed": job["fixed"], "fitness": fitness, "parts": parts, "descriptor": list(desc),
        "final": dict(rows[-1]), "ckpt": job["ckpt"],
        "series": {k: [r.get(k, 0) for r in rows] for k in keep},
        "seconds": round(time.perf_counter() - t0, 1),
    }


def command_line(rec: dict) -> str:
    """Reproduce a world from scratch with the CLI."""
    return (f"python -m primordial run --seed {rec['seed']} --steps {rec['steps']} --save world.pt "
            f"{set_args({**rec['fixed'], **rec['params']})}")


# ------------------------------------------------------------------ MAP-Elites


class Archive:
    """Best record per behaviour cell."""

    def __init__(self, grid: int):
        self.grid = grid
        self.cells: dict[tuple[int, int], dict] = {}

    def add(self, rec: dict) -> tuple[bool, dict | None]:
        """Insert if the cell is empty or the record beats its elite. Returns (inserted, displaced)."""
        c = cell_of(tuple(rec["descriptor"]), self.grid)
        rec["cell"] = list(c)
        old = self.cells.get(c)
        if old is None or rec["fitness"] > old["fitness"]:
            self.cells[c] = rec
            return True, old
        return False, None

    def elites(self) -> list[dict]:
        return sorted(self.cells.values(), key=lambda r: -r["fitness"])


def propose(archive: Archive, rng: random.Random, n_random: bool, sigma: float = 0.12) -> list[float]:
    """A new genotype: uniform random while seeding, else a mutated (sometimes crossed-over) elite."""
    d = len(PARAMS)
    if n_random or not archive.cells:
        return [rng.random() for _ in range(d)]
    elites = list(archive.cells.values())
    child = list(rng.choice(elites)["unit"])
    if len(elites) > 1 and rng.random() < 0.3:   # crossover helps combine traits from different niches
        other = rng.choice(elites)["unit"]
        child = [a if rng.random() < 0.5 else b for a, b in zip(child, other)]
    return [min(1.0, max(0.0, u + rng.gauss(0, sigma))) for u in child]


def read_log(path: Path, steps: int) -> list[dict]:
    """Records from a previous (possibly interrupted) run with the same step count."""
    if not path.exists():
        return []
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue   # a torn last line from a crash
        if r.get("steps") == steps:
            recs.append(r)
    return recs


def discover(evals: int = 24, workers: int = 6, steps: int = 1500, out: str | Path = "runs/discovery",
             seed: int = 0, grid: int = 6, every: int = 50, n_init: int | None = None,
             fixed: dict | None = None, log: Callable[[str], None] = print) -> Path:
    """Run (or resume) MAP-Elites, then write archive.json, thumbnails and gallery.html."""
    out = Path(out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    log_path = out / "evals.jsonl"
    archive = Archive(grid)
    done = read_log(log_path, steps)
    for r in done:
        archive.add(r)
    n_init = n_init if n_init is not None else max(workers, evals // 3)
    next_id = max((r["id"] for r in done), default=-1) + 1
    log(f"discovery: {len(done)} evaluations already done, {max(0, evals - len(done))} to go")
    t0 = time.perf_counter()

    while len(done) < evals:
        batch = min(max(1, workers), evals - len(done))
        rng = random.Random(seed * 1_000_003 + next_id)
        jobs = []
        for k in range(batch):
            unit = propose(archive, rng, n_random=len(done) + k < n_init)
            i = next_id + k
            jobs.append({"id": i, "seed": seed * 10_000 + i, "unit": unit, "params": decode(unit),
                         "fixed": {**FIXED, **(fixed or {})}, "steps": steps, "every": every,
                         "ckpt": str(out / "ckpt" / f"e{i:04d}.pt")})
        next_id += batch
        with open(log_path, "a", encoding="utf-8") as fh:
            for rec in pool_map(evaluate, jobs, workers):
                fh.write(json.dumps(jsonable(rec)) + "\n")
                fh.flush()
                done.append(rec)
                inserted, displaced = archive.add(rec)
                for loser in ([displaced] if displaced else []) + ([] if inserted else [rec]):
                    Path(loser["ckpt"]).unlink(missing_ok=True)   # keep checkpoints for elites only
                log(f"[eval {len(done)}/{evals}] e{rec['id']} fitness {rec['fitness']:.3f} "
                    f"cell {tuple(rec['cell'])} {'NEW ELITE' if inserted else ''} "
                    f"({time.perf_counter() - t0:.0f}s elapsed)")

    write_archive(archive, out, steps, seed, len(done))
    path = render_gallery(archive, out, len(done))
    log(f"gallery: {path}")
    return out


def write_archive(archive: Archive, out: Path, steps: int, seed: int, n_evals: int) -> Path:
    elites = []
    for r in archive.elites():
        e = {k: r[k] for k in ("id", "cell", "fitness", "parts", "descriptor", "params", "fixed", "seed", "steps", "final")}
        e["checkpoint"] = Path(r["ckpt"]).as_posix()
        e["command"] = command_line(r)
        e["view"] = f"python -m primordial view --load {Path(r['ckpt']).as_posix()}"
        elites.append(e)
    data = {
        "evaluations": n_evals, "steps": steps, "seed": seed, "grid": archive.grid,
        "axes": list(AXES), "bin_edges": bin_edges(archive.grid), "search_space": PARAMS, "fixed": FIXED,
        "fitness": "autonomy x weighted geometric mean of pop/gen/species/lineage (see primordial/discovery.py)",
        "weights": WEIGHTS, "coverage": len(archive.cells) / archive.grid**2, "elites": elites,
    }
    path = out / "archive.json"
    path.write_text(json.dumps(jsonable(data), indent=1), encoding="utf-8")
    return path


# ------------------------------------------------------------------ gallery


def thumbnail(ckpt: str | Path, path: Path, size_in: float = 3.2) -> None:
    """Snapshot of a world: particles in genome colour (sized by body), bonds as lines, food as dots."""
    from .experiments import _plt
    from .world import World

    plt = _plt()
    w = World.load(str(ckpt))
    W = w.cfg.world_size
    alive = w.alive.nonzero().squeeze(1)
    pos = w.pos[alive].numpy()
    fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_facecolor("#111418")
    food = w.food[w.food_alive].numpy()
    ax.scatter(food[:, 0], food[:, 1], s=1.5, color="#6b8f5e", linewidths=0)
    segs = []
    for i, partners in zip(alive.tolist(), w.bonds[alive].tolist()):
        for j in partners:
            if j > i and w.alive[j]:
                a, b = w.pos[i].tolist(), w.pos[j].tolist()
                if abs(a[0] - b[0]) < W / 2 and abs(a[1] - b[1]) < W / 2:   # skip lines across the wrap
                    segs.append([a, b])
    if segs:
        from matplotlib.collections import LineCollection

        ax.add_collection(LineCollection(segs, colors="#e8e6df", linewidths=0.6, alpha=0.7))
    size = w.body(alive)[0].numpy()
    ax.scatter(pos[:, 0], pos[:, 1], s=9 * size**2, c=w.genome.color[alive].clamp(0, 1).numpy(), linewidths=0)
    ax.set_xlim(0, W)
    ax.set_ylim(0, W)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(path, facecolor="#111418")
    plt.close(fig)


def sparkline(rec: dict, path: Path) -> None:
    """Population over the run, with the abiogenesis floor marked."""
    from .experiments import INK_2, PALETTE, _plt

    plt = _plt()
    s = rec["series"]
    fig, ax = plt.subplots(figsize=(3.2, 0.9), dpi=100)
    floor = Config().min_population if "min_population" not in rec["fixed"] else rec["fixed"]["min_population"]
    ax.axhline(floor, color=INK_2, linewidth=0.8, linestyle=":")
    ax.plot(s["step"], s["population"], color=PALETTE[0], linewidth=1.6)
    ax.set_ylim(0, max(max(s["population"]), floor) * 1.1)
    ax.tick_params(labelsize=6.5, length=2)
    ax.set_title("population (dotted: spawn floor)", fontsize=7, fontweight="normal", color=INK_2)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def heatmap(archive: Archive, path: Path) -> None:
    """Archive coverage: fitness of the elite in each behaviour cell."""
    import numpy as np

    from .experiments import _plt

    plt = _plt()
    g = archive.grid
    m = np.full((g, g), np.nan)
    for (x, y), r in archive.cells.items():
        m[y, x] = r["fitness"]
    fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=110)
    im = ax.imshow(m, origin="lower", cmap="Blues", vmin=0, vmax=max(0.05, np.nanmax(m) if archive.cells else 1),
                   extent=(0, 1, 0, 1))
    for (x, y), r in archive.cells.items():
        ax.text((x + 0.5) / g, (y + 0.5) / g, f"e{r['id']}", ha="center", va="center", fontsize=6.5,
                color="#0b0b0b" if r["fitness"] < 0.6 * np.nanmax(m) else "#fcfcfb")
    edges = bin_edges(g)
    ticks = [i / g for i in range(g + 1)]
    ax.set_xticks(ticks, [f"{e:.2g}" for e in edges], fontsize=7)
    ax.set_yticks(ticks, [f"{e:.2g}" for e in edges], fontsize=7)
    ax.set_xlabel("predation share of diet (sqrt-spaced bins)")
    ax.set_ylabel("bonded fraction")
    ax.grid(False)
    ax.set_title("Archive: best fitness per niche")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="interestingness")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _distinctive(params: dict, k: int = 4) -> list[tuple[str, float, float]]:
    """The k parameters furthest from the engine defaults (in unit-cube distance)."""
    defaults = {n: getattr(Config, n) for n in PARAMS}
    du, pu = encode(defaults), encode(params)
    order = sorted(range(len(PARAMS)), key=lambda i: -abs(pu[i] - du[i]))[:k]
    names = list(PARAMS)
    return [(names[i], params[names[i]], defaults[names[i]]) for i in order]


def render_gallery(archive: Archive, out: Path, n_evals: int) -> Path:
    from .experiments import _esc, _img, fmt, page

    tdir = out / "thumbs"
    tdir.mkdir(exist_ok=True)
    heatmap(archive, tdir / "archive.png")
    elites = archive.elites()
    g = archive.grid
    h = [f"<h1>World discovery</h1><p class='sub'>MAP-Elites · {n_evals} evaluations · "
         f"{elites[0]['steps'] if elites else 0:,} steps each · {len(elites)}/{g * g} niches filled · "
         f"generated {time.strftime('%Y-%m-%d %H:%M')}</p>",
         "<div class='card'><p><b>Grid:</b> predation share of diet (grazers → hunters) × bonded fraction "
         "(solitary → colonial), averaged over the last third of each run. <b>Interestingness</b> = autonomy × "
         "weighted geometric mean of sustained population (0.3), generational progress (0.25), distinct named "
         "species (0.25) and lineage diversity (0.2). Autonomy is the share of late newcomers that were born "
         "rather than spawned by the <code>min_population</code> floor, so worlds that only survive through "
         "abiogenesis score near zero.</p>" + _img(tdir / "archive.png", "archive heatmap") + "</div>",
         "<h2>Elites, most interesting first</h2><div class='charts'>"]
    for r in elites:
        thumb, spark = tdir / f"e{r['id']:04d}.png", tdir / f"e{r['id']:04d}_pop.png"
        if Path(r["ckpt"]).exists():
            thumbnail(r["ckpt"], thumb)
        sparkline(r, spark)
        p = r["parts"]
        f = r["final"]
        dist = "".join(f"<tr><td>{_esc(n)}</td><td>{fmt(v)}</td><td>{fmt(d)}</td></tr>" for n, v, d in _distinctive(r["params"]))
        species = _esc((f.get("species") or "-").replace("|", ", "))
        h.append(
            "<div class='card'>" + (_img(thumb, f"world e{r['id']}") if thumb.exists() else "") + _img(spark, "population")
            + f"<h3 style='margin:10px 0 2px'>e{r['id']} · interestingness {r['fitness']:.3f}</h3>"
            f"<p class='sub' style='margin:0 0 6px'>niche {tuple(r['cell'])}: predation {r['descriptor'][0]:.3f}, "
            f"bonded {r['descriptor'][1]:.3f}</p>"
            f"<table><tr><th>autonomy</th><th>pop</th><th>gen</th><th>species</th><th>lineage</th></tr>"
            f"<tr><td>{p['autonomy']:.2f}</td><td>{p['pop']:.2f}</td><td>{p['gen']:.2f}</td><td>{p['species']:.2f}</td>"
            f"<td>{p['lineage']:.2f}</td></tr></table>"
            f"<p class='sub' style='margin:4px 0'>final: population {f.get('population')}, max generation "
            f"{f.get('max_generation')}, evolved lineages {f.get('evolved_lineages')}, spawned {f.get('spawned')}<br>species: {species}</p>"
            f"<table><tr><th>most distinctive</th><th>value</th><th>default</th></tr>{dist}</table>"
            f"<details><summary>reproduce</summary><pre>{_esc(command_line(r))}\n\n"
            f"python -m primordial view --load {_esc(Path(r['ckpt']).as_posix())}</pre></details></div>")
    h.append("</div>")
    path = out / "gallery.html"
    path.write_text(page("World discovery", h), encoding="utf-8")
    return path


def cmd_discover(args) -> None:
    discover(evals=args.evals, workers=args.workers, steps=args.steps, out=args.out, seed=args.seed, grid=args.grid)


def add_subparser(subparsers) -> None:
    """Register `discover [--evals N] [--workers N] [--steps N] [--out DIR] [--seed S]`."""
    p = subparsers.add_parser("discover", help="MAP-Elites search for interesting world settings")
    p.add_argument("--evals", type=int, default=24, help="total world evaluations (resumes if --out has some)")
    p.add_argument("--workers", type=int, default=6, help="parallel processes")
    p.add_argument("--steps", type=int, default=1500, help="steps per evaluation")
    p.add_argument("--out", default="runs/discovery", help="output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", type=int, default=6, help="bins per descriptor axis")
    p.set_defaults(func=cmd_discover)
