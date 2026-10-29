"""Multi-seed experiments: run variants x seeds in parallel, aggregate, compare, report.

A spec is a JSON file:

    {"name": "mutation", "steps": 6000, "every": 250, "seeds": [1, 2, 3],
     "base": {"capacity": 800},                      # overrides shared by every variant
     "variants": {"old": {"sigma_tau": 0}, "new": {}},
     "tournament": {"trials": 3, "steps": 2000}}     # optional head-to-head of final populations

    python -m primordial experiment experiments/mutation.json --workers 6
    python -m primordial experiment experiments/mutation.json --report-only

Outputs land in runs/<name>/: per-run CSV + checkpoint, charts, summary.json, report.html.
Finished runs are skipped on re-launch, so an interrupted experiment resumes where it stopped.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import time
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .config import Config

# ------------------------------------------------------------------ statistics


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularised incomplete beta (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def t_cdf(t: float, df: float) -> float:
    """CDF of Student's t distribution."""
    tail = 0.5 * betainc(df / 2, 0.5, df / (df + t * t))
    return 1 - tail if t >= 0 else tail


def t_ppf(q: float, df: float) -> float:
    """Inverse CDF of Student's t (bisection; plenty fast for report-sized work)."""
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mean_ci(values: list[float], level: float = 0.95) -> tuple[float, float]:
    """(mean, half-width of the t-based confidence interval); half-width is NaN for n < 2."""
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    m = sum(values) / n
    if n < 2:
        return m, math.nan
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    return m, t_ppf(0.5 + level / 2, n - 1) * sd / math.sqrt(n)


def welch_p(a: list[float], b: list[float]) -> float:
    """Two-sided Welch t-test p-value for a difference in means."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return math.nan
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se2 = va / na + vb / nb
    if se2 == 0:
        return 1.0 if ma == mb else 0.0
    t = (ma - mb) / math.sqrt(se2)
    df = se2**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return 2 * (1 - t_cdf(abs(t), df))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs pass through untouched)."""
    idx = sorted((i for i, p in enumerate(pvals) if not math.isnan(p)), key=lambda i: pvals[i])
    out, running, m = list(pvals), 0.0, len(idx)
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = running
    return out


# ------------------------------------------------------------ running worlds


def species_kinds(species: str, min_share: float = 0.05) -> set[str]:
    """Distinct species *names* ("grazer 2" counts as grazer) holding >= min_share of the population."""
    counts: dict[str, int] = {}
    for part in filter(None, (species or "").split("|")):
        name, _, n = part.rpartition(":")
        base = name.rstrip("0123456789 ").strip()
        counts[base] = counts.get(base, 0) + int(n)
    total = sum(counts.values())
    return {k for k, v in counts.items() if total and v >= min_share * total}


def spawn_count(world) -> int:
    """Particles created by abiogenesis after step 0 (the `min_population` floor)."""
    return sum(1 for p, b in zip(world.lineage.parent, world.lineage.born) if p < 0 and b > 0)


def evolved_lineages(world, min_generation: int = 2, bins: int = 4) -> int:
    """Like world.lineage_count(), but only over particles at least `min_generation` deep,
    so the colour spread of random founders and fresh spawns doesn't count as diversity."""
    keep = world.alive & (world.generation >= min_generation)
    c = world.genome.color[keep]
    if len(c) == 0:
        return 0
    cell = (c * bins).long().clamp(max=bins - 1)
    keys = cell[:, 0] * bins * bins + cell[:, 1] * bins + cell[:, 2]
    counts = keys.bincount(minlength=bins**3)
    return int((counts >= max(1, 0.05 * len(c))).sum())


def record(world) -> dict:
    """world.stats() plus the columns experiments care about."""
    s = world.stats()
    s["spawned"] = spawn_count(world)
    s["n_species"] = len(species_kinds(s.get("species", "")))
    s["evolved_lineages"] = evolved_lineages(world)
    return s


def simulate(world, steps: int, every: int, csv_path: str | Path | None = None,
             on_record: Callable[[dict], None] | None = None) -> list[dict]:
    """Step `world` and record stats every `every` steps (and at the end)."""
    rows: list[dict] = []
    fh = open(csv_path, "w", newline="") if csv_path else None
    try:
        writer = None
        for i in range(1, steps + 1):
            world.step()
            if i % every == 0 or i == steps:
                row = record(world)
                rows.append(row)
                if fh:
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=list(row), extrasaction="ignore", restval="")
                        writer.writeheader()
                    writer.writerow(row)
                    fh.flush()   # lets you watch progress with tail
                if on_record:
                    on_record(row)
    finally:
        if fh:
            fh.close()
    return rows


def make_config(seed: int, *override_dicts: dict) -> Config:
    """Config for `seed` with override dicts applied in order (CLI-style type coercion)."""
    cfg = Config(seed=seed)
    for d in override_dicts:
        for k, v in (d or {}).items():
            cfg.override([f"{k}={v}"])
    return cfg


def set_args(overrides: dict) -> str:
    """`--set k=v ...` fragment reproducing a dict of overrides."""
    return ("--set " + " ".join(f"{k}={v}" for k, v in overrides.items())) if overrides else ""


def _init_worker(threads: int = 1) -> None:
    # One torch thread per process: with several processes sharing the CPU, 2+ intra-op threads
    # spend most of their time waiting on each other (measured: 2.8 vs 21.7 steps/s per world).
    import torch

    torch.set_num_threads(threads)


def pool_map(fn: Callable, jobs: list, workers: int) -> Iterator:
    """imap_unordered over a spawn pool (or inline when workers <= 1)."""
    if not jobs:
        return
    if workers <= 1 or len(jobs) == 1:
        for j in jobs:
            yield fn(j)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, len(jobs)), initializer=_init_worker) as pool:
        yield from pool.imap_unordered(fn, jobs)


# ------------------------------------------------------------------ spec + jobs


def load_spec(path: str | Path) -> dict:
    """Read and validate an experiment spec, filling defaults."""
    spec = json.loads(Path(path).read_text())
    for key in ("name", "steps", "seeds", "variants"):
        if key not in spec:
            raise ValueError(f"spec is missing '{key}'")
    if not spec["variants"]:
        raise ValueError("spec needs at least one variant")
    spec.setdefault("every", max(1, spec["steps"] // 40))
    spec.setdefault("base", {})
    known = set(Config.__dataclass_fields__)
    for label, d in [("base", spec["base"]), *spec["variants"].items()]:
        bad = set(d or {}) - known
        if bad:
            raise ValueError(f"unknown config keys in {label}: {sorted(bad)}")
    t = spec.get("tournament")
    if t:
        t.setdefault("trials", 3)
        t.setdefault("steps", 2000)
        t.setdefault("each", 100)
        t.setdefault("overrides", {})
    return spec


def run_paths(out: Path, variant: str, seed: int) -> dict[str, Path]:
    stem = out / "runs" / f"{variant}_s{seed}"
    return {"csv": stem.with_suffix(".csv"), "ckpt": stem.with_suffix(".pt"), "meta": stem.with_suffix(".json")}


def run_jobs(spec: dict, out: Path) -> list[dict]:
    """Every variant x seed as a job description (picklable dict)."""
    jobs = []
    for variant, overrides in spec["variants"].items():
        for seed in spec["seeds"]:
            paths = run_paths(out, variant, seed)
            jobs.append({
                "kind": "run", "variant": variant, "seed": seed,
                "overrides": {**spec["base"], **(overrides or {})},
                "steps": spec["steps"], "every": spec["every"],
                **{k: str(v) for k, v in paths.items()},
            })
    return jobs


def _job_key(job: dict) -> dict:
    """The parts of a job that determine its result (a finished run is reused only if these match)."""
    return {k: job[k] for k in ("variant", "seed", "overrides", "steps", "every")}


def run_done(job: dict) -> bool:
    meta = Path(job["meta"])
    if not (meta.exists() and Path(job["csv"]).exists() and Path(job["ckpt"]).exists()):
        return False
    try:
        return json.loads(meta.read_text()).get("job") == json.loads(json.dumps(_job_key(job)))
    except (OSError, json.JSONDecodeError):
        return False


def execute_run(job: dict) -> dict:
    """Worker: one variant x seed run. Writes to temp names, then renames; meta file marks completion."""
    from .world import World

    t0 = time.perf_counter()
    world = World(make_config(job["seed"], job["overrides"]))
    tmp_csv, tmp_ckpt = job["csv"] + ".tmp", job["ckpt"] + ".tmp"
    rows = simulate(world, job["steps"], job["every"], tmp_csv)
    world.save(tmp_ckpt)
    os.replace(tmp_csv, job["csv"])
    os.replace(tmp_ckpt, job["ckpt"])
    elapsed = time.perf_counter() - t0
    Path(job["meta"]).write_text(json.dumps({"job": _job_key(job), "seconds": round(elapsed, 1),
                                             "final": rows[-1] if rows else {}}, indent=1))
    return {"job": job, "seconds": elapsed}


def duel_jobs(spec: dict, out: Path) -> list[dict]:
    """Head-to-head trials: every pair of variants, per seed pairing, `trials` contests each."""
    t = spec.get("tournament")
    if not t:
        return []
    jobs = []
    for a, b in combinations(spec["variants"], 2):
        for seed in spec["seeds"]:
            for trial in range(t["trials"]):
                jobs.append({
                    "kind": "duel", "a": a, "b": b, "seed": seed, "trial": trial,
                    "ckpt_a": str(run_paths(out, a, seed)["ckpt"]), "ckpt_b": str(run_paths(out, b, seed)["ckpt"]),
                    "steps": t["steps"], "each": t["each"],
                    "overrides": {**spec["base"], **t["overrides"]},
                    "arena_seed": 10_000 + 100 * seed + trial,
                    "result": str(out / "tournament" / f"{a}__{b}__s{seed}__t{trial}.json"),
                })
    return jobs


def execute_duel(job: dict) -> dict:
    """Worker: one head-to-head contest between the final populations of two runs."""
    from .tournament import head_to_head
    from .world import World

    n_a, n_b = head_to_head(World.load(job["ckpt_a"]), World.load(job["ckpt_b"]), job["arena_seed"],
                            job["steps"], job["each"], job["overrides"])
    res = {k: job[k] for k in ("a", "b", "seed", "trial", "steps", "each", "arena_seed")}
    res.update(n_a=n_a, n_b=n_b)
    Path(job["result"]).write_text(json.dumps(res))
    return {"job": job, "result": res}


def duel_done(job: dict) -> bool:
    p = Path(job["result"])
    if not p.exists():
        return False
    try:
        r = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return r.get("steps") == job["steps"] and r.get("each") == job["each"]


def execute(job: dict) -> dict:
    return execute_run(job) if job["kind"] == "run" else execute_duel(job)


def run_all(jobs: list[dict], workers: int, label: str, log: Callable[[str], None] = print) -> None:
    """Run jobs in a pool, printing progress as each finishes."""
    t0 = time.perf_counter()
    for i, res in enumerate(pool_map(execute, jobs, workers), 1):
        j = res["job"]
        what = f"{j['variant']} seed {j['seed']}" if j["kind"] == "run" else \
            f"{j['a']} vs {j['b']} seed {j['seed']} trial {j['trial']}: {res['result']['n_a']}-{res['result']['n_b']}"
        log(f"[{label} {i}/{len(jobs)}] {what} done ({time.perf_counter() - t0:.0f}s elapsed)")


# ------------------------------------------------------------------ aggregation

# Metrics compared at the end of the run (in report order) and charted over time.
KEY_METRICS = ["population", "max_generation", "mean_generation", "predation_share", "bonded_fraction",
               "lineages", "n_species", "mean_size", "mean_aggression", "color_diversity", "spawned"]
CHART_METRICS = KEY_METRICS[:-1] + ["mean_speed", "mean_sense", "mean_log_eta", "sigma_perception",
                                    "sigma_behaviour", "sigma_physics", "sigma_traits", "spawned"]
LABELS = {
    "population": "Population", "max_generation": "Max generation", "mean_generation": "Mean generation",
    "predation_share": "Predation share of diet", "bonded_fraction": "Bonded fraction",
    "lineages": "Lineages (colour clusters)", "n_species": "Distinct named species", "mean_size": "Mean body size",
    "mean_aggression": "Mean aggression", "color_diversity": "Colour diversity", "spawned": "Abiogenesis spawns (cum.)",
    "mean_speed": "Mean speed", "mean_sense": "Mean sense", "mean_log_eta": "Mean log10 learning rate",
    "sigma_perception": "Sigma: perception", "sigma_behaviour": "Sigma: behaviour",
    "sigma_physics": "Sigma: physics", "sigma_traits": "Sigma: traits",
}
END_WINDOW = 0.2   # "end-of-run" value = mean over the last 20% of the run (less noisy than one sample)


def read_series(csv_path: str | Path) -> dict[str, list[float]]:
    """Numeric columns of a stats CSV as lists (non-numeric columns are dropped)."""
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, list[float]] = {}
    for key in rows[0] if rows else []:
        try:
            out[key] = [float(r[key]) for r in rows]
        except (TypeError, ValueError):
            continue
    return out


def end_value(series: dict[str, list[float]], metric: str) -> float:
    steps, vals = series["step"], series.get(metric)
    if not vals:
        return math.nan
    cut = steps[-1] - END_WINDOW * steps[-1]
    tail = [v for s, v in zip(steps, vals) if s >= cut]
    return sum(tail) / len(tail)


def aggregate(runs: list[dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    """Per metric: step, mean, lo, hi (95% t CI across runs) on the steps every run has."""
    if not runs:
        return {}
    common = sorted(set.intersection(*(set(r["step"]) for r in runs)))
    metrics = set.intersection(*(set(r) for r in runs)) - {"step"}
    out = {}
    for m in sorted(metrics):
        lookup = [dict(zip(r["step"], r[m])) for r in runs]
        mean, lo, hi = [], [], []
        for s in common:
            mu, half = mean_ci([lk[s] for lk in lookup])
            half = 0.0 if math.isnan(half) else half
            mean.append(mu)
            lo.append(mu - half)
            hi.append(mu + half)
        out[m] = {"step": common, "mean": mean, "lo": lo, "hi": hi, "n": len(runs)}
    return out


def tournament_summary(results: list[dict], variants: list[str]) -> list[dict]:
    """Per variant pair: trial wins/losses/ties, per-seed majorities, and sign-test p-values."""
    from .tournament import sign_test

    out = []
    for a, b in combinations(variants, 2):
        rs = [r for r in results if r["a"] == a and r["b"] == b]
        if not rs:
            continue
        wins = sum(r["n_a"] > r["n_b"] for r in rs)
        losses = sum(r["n_b"] > r["n_a"] for r in rs)
        per_seed = {}
        for r in rs:
            d = per_seed.setdefault(r["seed"], {"seed": r["seed"], "a_wins": 0, "b_wins": 0, "ties": 0,
                                                "a_survivors": 0, "b_survivors": 0})
            d["a_wins" if r["n_a"] > r["n_b"] else "b_wins" if r["n_b"] > r["n_a"] else "ties"] += 1
            d["a_survivors"] += r["n_a"]
            d["b_survivors"] += r["n_b"]
        seeds = sorted(per_seed.values(), key=lambda d: d["seed"])
        seed_a = sum(d["a_wins"] > d["b_wins"] for d in seeds)
        seed_b = sum(d["b_wins"] > d["a_wins"] for d in seeds)
        share = [d["a_survivors"] / max(1, d["a_survivors"] + d["b_survivors"]) for d in seeds]
        mu, half = mean_ci(share)
        out.append({
            "a": a, "b": b, "trials": len(rs), "a_wins": wins, "b_wins": losses, "ties": len(rs) - wins - losses,
            "p_trials": sign_test(wins, losses),
            "seed_a_wins": seed_a, "seed_b_wins": seed_b, "p_seeds": sign_test(seed_a, seed_b),
            "a_share_mean": mu, "a_share_ci": half, "per_seed": seeds,
        })
    return out


def summarise(spec: dict, out: Path) -> dict:
    """Read every finished run (and duel) under `out` and compute the summary dict."""
    variants = list(spec["variants"])
    series: dict[str, dict[int, dict]] = {v: {} for v in variants}
    for job in run_jobs(spec, out):
        if run_done(job):   # finished *and* produced by the current spec
            series[job["variant"]][job["seed"]] = read_series(job["csv"])

    finals: dict[str, dict] = {}
    for v in variants:
        finals[v] = {}
        for m in KEY_METRICS:
            vals = [end_value(s, m) for s in series[v].values()]
            vals = [x for x in vals if not math.isnan(x)]
            mu, half = mean_ci(vals)
            finals[v][m] = {"mean": mu, "ci": half, "n": len(vals), "values": vals}

    comparisons = []
    for a, b in combinations(variants, 2):
        rows = []
        for m in KEY_METRICS:
            fa, fb = finals[a][m], finals[b][m]
            rows.append({"a": a, "b": b, "metric": m, "diff": fa["mean"] - fb["mean"],
                         "p": welch_p(fa["values"], fb["values"])})
        for r, adj in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = adj
        comparisons += rows

    duels = [json.loads(Path(j["result"]).read_text()) for j in duel_jobs(spec, out) if duel_done(j)]
    summary = {
        "name": spec["name"], "spec": spec,
        "completed": {v: list(series[v]) for v in variants},   # same order as final[v][m]["values"]
        "end_window": END_WINDOW,
        "final": finals, "comparisons": comparisons,
        "tournament": tournament_summary(duels, variants) if spec.get("tournament") else [],
    }
    summary["findings"] = findings(summary)
    summary["_series"] = {v: aggregate(list(series[v].values())) for v in variants}
    return summary


# ------------------------------------------------------------------ findings


def fmt(x: float, ci: float | None = None) -> str:
    """Compact number (with optional ± CI) for prose and tables."""
    def f(v: float) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        a = abs(v)
        return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}" if a >= 0.1 else f"{v:.3f}"
    return f(x) if ci is None or math.isnan(ci) else f"{f(x)} ± {f(ci)}"


def fmt_p(p: float) -> str:
    return "n/a" if math.isnan(p) else "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def findings(summary: dict) -> list[str]:
    """Plain-language bullet points computed from the numbers (verdict first)."""
    spec, finals = summary["spec"], summary["final"]
    variants = list(spec["variants"])
    n_done = {v: len(summary["completed"][v]) for v in variants}
    lines: list[str] = []
    evidence = False
    by_pair: dict[tuple, list[dict]] = {}
    for r in summary["comparisons"]:
        by_pair.setdefault((r["a"], r["b"]), []).append(r)

    for (a, b), rows in by_pair.items():
        rows = [r for r in rows if r["metric"] != "spawned" and not math.isnan(r["p"])]
        if not rows:
            lines.append(f"{a} vs {b}: not enough finished seeds to compare (need ≥ 2 per variant).")
            continue
        strong = [r for r in rows if r["p_holm"] < 0.05]
        hints = [r for r in rows if r["p"] < 0.05 and r["p_holm"] >= 0.05]
        for r in strong:
            evidence = True
            fa, fb = finals[a][r["metric"]], finals[b][r["metric"]]
            lines.append(
                f"{LABELS[r['metric']]} ended {'higher' if r['diff'] > 0 else 'lower'} under <b>{a}</b> "
                f"({fmt(fa['mean'], fa['ci'])}) than under <b>{b}</b> ({fmt(fb['mean'], fb['ci'])}); "
                f"p = {fmt_p(r['p'])}, still significant after Holm correction for {len(rows)} metrics.")
        for r in hints:
            fa, fb = finals[a][r["metric"]], finals[b][r["metric"]]
            lines.append(
                f"Hint only: {LABELS[r['metric']].lower()} looked {'higher' if r['diff'] > 0 else 'lower'} under "
                f"<b>{a}</b> ({fmt(fa['mean'], fa['ci'])} vs {fmt(fb['mean'], fb['ci'])}, p = {fmt_p(r['p'])}), "
                f"but not after correcting for testing {len(rows)} metrics.")
        if not strong and not hints:
            best = min(rows, key=lambda r: r["p"])
            lines.append(f"No end-of-run metric differed reliably between <b>{a}</b> and <b>{b}</b> "
                         f"(smallest p = {fmt_p(best['p'])}, for {LABELS[best['metric']].lower()}).")

    for t in summary["tournament"]:
        a, b = t["a"], t["b"]
        n_seeds = len(t["per_seed"])
        lines.append(
            f"Head-to-head, {t['trials']} contests over {n_seeds} seed pairings: <b>{a}</b> won {t['a_wins']}, "
            f"<b>{b}</b> won {t['b_wins']}, {t['ties']} tied (sign test over contests p = {fmt_p(t['p_trials'])}). "
            f"Per seed pairing, {a} took the majority in {t['seed_a_wins']} and {b} in {t['seed_b_wins']} "
            f"(p = {fmt_p(t['p_seeds'])}); {a}'s share of survivors averaged "
            f"{fmt(100 * t['a_share_mean'], 100 * t['a_share_ci'] if not math.isnan(t['a_share_ci']) else math.nan)}%.")
        if t["p_seeds"] < 0.05:
            evidence = True

    for v in variants:
        sp = finals[v]["spawned"]
        if sp["n"] and sp["mean"] > 0:
            lines.append(f"Under <b>{v}</b> the population dipped below the <code>min_population</code> floor "
                         f"and needed on average {fmt(sp['mean'])} spontaneously spawned particles; "
                         f"part of its numbers came from abiogenesis, not reproduction.")
    pop = [finals[v]["population"] for v in variants if finals[v]["population"]["n"] >= 2]
    if pop:
        cv = max(math.sqrt(sum((x - p["mean"]) ** 2 for x in p["values"]) / (p["n"] - 1)) / max(p["mean"], 1e-9)
                 for p in pop)
        lines.append(f"Seed-to-seed noise: end-of-run population has a coefficient of variation of up to "
                     f"{100 * cv:.0f}% between seeds of the same variant"
                     + (", so only large effects are detectable with this many seeds." if cv > 0.1 else "."))

    seeds_txt = ", ".join(f"{v}: {n_done[v]}" for v in variants)
    if evidence:
        verdict = (f"<b>Verdict: there is evidence of a difference</b> (seeds finished per variant: {seeds_txt}). "
                   f"See the specific results below; they survive multiple-comparison correction or a seed-level sign test.")
    else:
        leans = [f"<b>{t['a'] if t['a_wins'] > t['b_wins'] else t['b']}</b> won {max(t['a_wins'], t['b_wins'])} of "
                 f"{t['a_wins'] + t['b_wins']} decisive contests (p = {fmt_p(t['p_trials'])}), but only "
                 f"{max(t['seed_a_wins'], t['seed_b_wins'])} of {len(t['per_seed'])} seed pairings "
                 f"(p = {fmt_p(t['p_seeds'])})" for t in summary["tournament"] if t["p_trials"] < 0.05]
        verdict = (f"<b>Verdict: inconclusive.</b> With the seeds finished so far ({seeds_txt}), no difference between "
                   f"variants clears the bar of seed-level replication. "
                   + (f"There is a lean: {'; '.join(leans)}, so the effect may depend on which populations happened "
                      f"to evolve. More seeds would settle it." if leans else
                      "More seeds or longer runs would be needed to detect a small effect."))
    if summary["tournament"]:
        lines.append("Contest p-values treat repeated contests as independent, but contests from one seed pairing "
                     "share the same two populations; the seed-level sign test is the more honest one.")
    return [verdict] + lines


# ------------------------------------------------------------------ charts + report

# Categorical slots in fixed order (validated palette); identity is also carried by legends and labels.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    return plt


def png_b64(path: str | Path) -> str:
    import base64

    return base64.b64encode(Path(path).read_bytes()).decode()


def chart_metric(metric: str, agg: dict[str, dict], colors: dict[str, str], path: Path) -> bool:
    """Mean line + 95% CI band per variant. Returns False if no variant has the metric."""
    plt = _plt()
    present = {v: a[metric] for v, a in agg.items() if metric in a and a[metric]["step"]}
    if not present:
        return False
    fig, ax = plt.subplots(figsize=(5.6, 3.1), dpi=120)
    for v, s in present.items():
        ax.fill_between(s["step"], s["lo"], s["hi"], color=colors[v], alpha=0.16, linewidth=0)
        ax.plot(s["step"], s["mean"], color=colors[v], linewidth=2, label=f"{v} (n={s['n']})")
    if len(present) <= 4:   # direct labels at line ends, nudged apart so they never overlap
        lo, hi = ax.get_ylim()
        gap = 0.07 * (hi - lo)
        ends = sorted(((s["mean"][-1], v, s["step"][-1]) for v, s in present.items()))
        placed: list[float] = []
        for y, v, x in ends:
            y = max(y, placed[-1] + gap) if placed else y
            placed.append(y)
            ax.annotate(v, (x, y), xytext=(4, 0), textcoords="offset points", color=INK_2, fontsize=8, va="center")
    ax.set_title(LABELS.get(metric, metric))
    ax.set_xlabel("step")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def chart_tournament(t: dict, colors: dict[str, str], path: Path) -> None:
    """Share of survivors for side A in each seed pairing (50% = even)."""
    plt = _plt()
    seeds = t["per_seed"]
    shares = [d["a_survivors"] / max(1, d["a_survivors"] + d["b_survivors"]) for d in seeds]
    fig, ax = plt.subplots(figsize=(5.6, 2.6), dpi=120)
    ax.axhline(0.5, color=INK_2, linewidth=1, linestyle="--")
    ax.scatter(range(len(seeds)), shares, s=48, color=colors[t["a"]], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_xticks(range(len(seeds)), [f"seed {d['seed']}" for d in seeds])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(f"{t['a']} share of survivors")
    ax.set_title(f"{t['a']} vs {t['b']}: survivors per seed pairing")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


CSS = """
:root{--bg:#f6f5f2;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;--good:#1d7a4f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2e2e2b;--accent:#3987e5;--good:#5cc28f}}
:root[data-theme="dark"]{--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2e2e2b;--accent:#3987e5;--good:#5cc28f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 10px}
.sub{color:var(--ink2);margin:0 0 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:12px 0}
.findings{padding-left:20px;margin:0}.findings li{margin:7px 0}
.findings li:first-child{list-style:none;margin-left:-20px;padding:10px 12px;border-left:4px solid var(--accent);background:var(--bg);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums;margin-bottom:10px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--ink2);font-weight:600}.scroll{overflow-x:auto}.sig{color:var(--good);font-weight:600}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(340px,100%),1fr));gap:12px}
.charts img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}
code,pre{font:12.5px/1.45 ui-monospace,Consolas,monospace}
pre{background:var(--bg);padding:10px;border-radius:6px;overflow-x:auto;white-space:pre-wrap;margin:0}
.swatch{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
"""


def _esc(s) -> str:
    import html

    return html.escape(str(s))


def _kv(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items())


def page(title: str, body: list[str]) -> str:
    """Wrap body HTML in a standalone page with the shared stylesheet."""
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_esc(title)}</title><style>{CSS}</style></head><body><main>"
            + "\n".join(body) + "</main></body></html>")


def _img(path: Path, alt: str) -> str:
    return f"<img alt='{_esc(alt)}' src='data:image/png;base64,{png_b64(path)}'>"


def render_report(summary: dict, out: Path) -> Path:
    """Write charts and the self-contained report.html; returns its path."""
    spec = summary["spec"]
    variants = list(spec["variants"])
    colors = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(variants)}
    cdir = out / "charts"
    cdir.mkdir(parents=True, exist_ok=True)
    imgs = [cdir / f"{m}.png" for m in CHART_METRICS
            if chart_metric(m, summary["_series"], colors, cdir / f"{m}.png")]
    timgs = []
    for t in summary["tournament"]:
        p = cdir / f"tournament_{t['a']}_vs_{t['b']}.png"
        chart_tournament(t, colors, p)
        timgs.append(p)

    h = [f"<h1>Experiment: {_esc(spec['name'])}</h1>"
         f"<p class='sub'>{len(variants)} variants × {len(spec['seeds'])} seeds ({', '.join(map(str, spec['seeds']))}) · "
         f"{spec['steps']:,} steps each · stats every {spec['every']} steps · generated {time.strftime('%Y-%m-%d %H:%M')}</p>",
         "<h2>What this shows</h2><div class='card'><ul class='findings'>"
         + "".join(f"<li>{line}</li>" for line in summary["findings"]) + "</ul></div>",
         "<h2>Variants</h2><div class='card scroll'><table><tr><th>variant</th><th>overrides (on top of base)</th>"
         "<th>seeds finished</th></tr>"]
    for v in variants:
        h.append(f"<tr><td><span class='swatch' style='background:{colors[v]}'></span>{_esc(v)}</td>"
                 f"<td><code>{_esc(_kv(spec['variants'][v] or {}) or '(defaults)')}</code></td>"
                 f"<td>{len(summary['completed'][v])}/{len(spec['seeds'])}</td></tr>")
    h.append(f"</table><p class='sub'>Base overrides: <code>{_esc(_kv(spec['base']) or '(none)')}</code></p></div>")

    pairs = list(combinations(variants, 2))
    h.append("<h2>End of run</h2><div class='card scroll'><p class='sub'>Mean ± 95% CI across seeds (t-distribution) of "
             f"each run's average over its last {int(100 * END_WINDOW)}% of steps. p: Welch t-test. "
             "p (Holm): corrected for testing every metric; green = significant after correction.</p>"
             "<table><tr><th>metric</th>" + "".join(f"<th>{_esc(v)}</th>" for v in variants)
             + "".join(f"<th>{_esc(a)} − {_esc(b)}</th><th>p</th><th>p (Holm)</th>" for a, b in pairs) + "</tr>")
    comp = {(r["a"], r["b"], r["metric"]): r for r in summary["comparisons"]}
    for m in KEY_METRICS:
        row = f"<tr><td>{_esc(LABELS[m])}</td>" + "".join(
            f"<td>{fmt(summary['final'][v][m]['mean'], summary['final'][v][m]['ci'])}</td>" for v in variants)
        for a, b in pairs:
            r = comp[(a, b, m)]
            cls = " class='sig'" if r["p_holm"] < 0.05 else ""
            row += f"<td>{fmt(r['diff'])}</td><td{cls}>{fmt_p(r['p'])}</td><td{cls}>{fmt_p(r['p_holm'])}</td>"
        h.append(row + "</tr>")
    h.append("</table></div>")

    if summary["tournament"]:
        ts = spec["tournament"]
        forced = {"initial_particles", "min_population", "capacity"}   # set by the arena itself
        arena = _kv({k: v for k, v in {**spec["base"], **ts["overrides"]}.items() if k not in forced}) or "none"
        h.append(f"<h2>Head-to-head tournament</h2><div class='card scroll'><p class='sub'>For each seed, {ts['each']} "
                 f"particles cloned from each variant's final population share a fresh arena for {ts['steps']:,} steps "
                 f"(no abiogenesis; arena overrides: <code>{_esc(arena)}</code>). The side with more survivors wins. "
                 f"Repeated {ts['trials']}× per seed pairing with different arenas.</p>"
                 "<table><tr><th>pairing</th><th>contests</th><th>A wins</th><th>B wins</th><th>ties</th>"
                 "<th>p (contests)</th><th>seed majorities A–B</th><th>p (seeds)</th><th>A share of survivors</th></tr>")
        for t in summary["tournament"]:
            ci = 100 * t["a_share_ci"] if not math.isnan(t["a_share_ci"]) else math.nan
            h.append(f"<tr><td>A={_esc(t['a'])}, B={_esc(t['b'])}</td><td>{t['trials']}</td><td>{t['a_wins']}</td>"
                     f"<td>{t['b_wins']}</td><td>{t['ties']}</td><td>{fmt_p(t['p_trials'])}</td>"
                     f"<td>{t['seed_a_wins']}–{t['seed_b_wins']}</td><td>{fmt_p(t['p_seeds'])}</td>"
                     f"<td>{fmt(100 * t['a_share_mean'], ci)}%</td></tr>")
        h.append("</table><table><tr><th>pairing</th><th>seed</th><th>A wins</th><th>B wins</th><th>ties</th>"
                 "<th>A survivors (sum)</th><th>B survivors (sum)</th></tr>")
        for t in summary["tournament"]:
            for d in t["per_seed"]:
                h.append(f"<tr><td>{_esc(t['a'])} vs {_esc(t['b'])}</td><td>{d['seed']}</td><td>{d['a_wins']}</td>"
                         f"<td>{d['b_wins']}</td><td>{d['ties']}</td><td>{d['a_survivors']}</td><td>{d['b_survivors']}</td></tr>")
        h.append("</table><div class='charts'>" + "".join(_img(p, "tournament survivors per seed") for p in timgs)
                 + "</div></div>")

    h.append("<h2>Over time</h2><p class='sub'>Line: mean across seeds. Band: 95% confidence interval of that mean.</p>"
             "<div class='charts'>" + "".join(_img(p, LABELS.get(p.stem, p.stem)) for p in imgs) + "</div>")

    h.append("<h2>Per-run end values</h2><div class='card scroll'><table><tr><th>run</th>"
             + "".join(f"<th>{_esc(LABELS[m])}</th>" for m in KEY_METRICS) + "</tr>")
    for v in variants:
        for i, seed in enumerate(summary["completed"][v]):
            cells = []
            for m in KEY_METRICS:
                vals = summary["final"][v][m]["values"]
                cells.append(f"<td>{fmt(vals[i])}</td>" if i < len(vals) else "<td></td>")
            h.append(f"<tr><td>{_esc(v)} s{seed}</td>{''.join(cells)}</tr>")
    h.append("</table></div>")

    cmds = [f"python -m primordial experiment {spec.get('_path', 'SPEC.json')} --workers 6", "", "# one run by hand:"]
    for v in variants:
        overrides = {**spec["base"], **(spec["variants"][v] or {})}
        cmds.append(f"python -m primordial run --seed {spec['seeds'][0]} --steps {spec['steps']} {set_args(overrides)}"
                    .rstrip() + f"   # {v}")
    h.append("<h2>Reproduce</h2><div class='card'><pre>" + _esc("\n".join(cmds)) + "</pre></div>")
    path = out / "report.html"
    path.write_text(page(f"{spec['name']} experiment", h), encoding="utf-8")
    return path


# ------------------------------------------------------------------ orchestration


def jsonable(x):
    """NaN/inf -> None, recursively, so the output is strict JSON."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def write_outputs(spec: dict, out: Path) -> dict:
    summary = summarise(spec, out)
    render_report(summary, out)
    public = {k: v for k, v in summary.items() if not k.startswith("_")}
    (out / "summary.json").write_text(json.dumps(jsonable(public), indent=1), encoding="utf-8")
    return summary


def run_experiment(spec_path: str | Path, workers: int = 6, report_only: bool = False,
                   runs_dir: str | Path = "runs", log: Callable[[str], None] = print) -> Path:
    """Run (or resume) an experiment, then write summary.json and report.html. Returns the output dir."""
    spec = load_spec(spec_path)
    spec["_path"] = Path(spec_path).as_posix()
    out = Path(runs_dir) / spec["name"]
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    if not report_only:
        all_jobs = run_jobs(spec, out)
        jobs = [j for j in all_jobs if not run_done(j)]
        log(f"{spec['name']}: {len(jobs)} runs to do ({len(all_jobs) - len(jobs)} already finished)")
        run_all(jobs, workers, "run", log)
        if spec.get("tournament"):
            (out / "tournament").mkdir(exist_ok=True)
            duels = [j for j in duel_jobs(spec, out) if not duel_done(j)]
            log(f"{spec['name']}: {len(duels)} head-to-head contests to do")
            run_all(duels, workers, "duel", log)
    write_outputs(spec, out)
    log(f"report: {out / 'report.html'}")
    return out


def cmd_experiment(args) -> None:
    run_experiment(args.spec, workers=args.workers, report_only=args.report_only, runs_dir=args.runs_dir)


def add_subparser(subparsers) -> None:
    """Register `experiment SPEC [--workers N] [--report-only]`."""
    p = subparsers.add_parser("experiment", help="run a multi-seed experiment spec and build a report")
    p.add_argument("spec", help="JSON spec file (see experiments/*.json)")
    p.add_argument("--workers", type=int, default=6, help="parallel processes")
    p.add_argument("--report-only", action="store_true", help="skip running; rebuild the report from finished runs")
    p.add_argument("--runs-dir", default="runs", help="parent directory for outputs")
    p.set_defaults(func=cmd_experiment)
