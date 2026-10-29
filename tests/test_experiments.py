"""Experiment harness, tournament and discovery: stats math plus tiny end-to-end runs."""

import json
import math

import pytest
import torch

from primordial import Config, World
from primordial import discovery, experiments as ex
from primordial.tournament import head_to_head, side_counts, sign_test

TINY = {"capacity": 120, "initial_particles": 40, "food_capacity": 80, "min_population": 10, "species_interval": 20}


# ------------------------------------------------------------------ statistics

@pytest.mark.parametrize("df, expected", [(1, 12.706), (5, 2.571), (30, 2.042)])
def test_t_quantiles_match_tables(df, expected):
    assert ex.t_ppf(0.975, df) == pytest.approx(expected, abs=1e-3)
    assert ex.t_cdf(0.0, df) == pytest.approx(0.5)


def test_mean_ci_uses_t_distribution():
    m, half = ex.mean_ci([1, 2, 3, 4, 5])
    assert m == 3
    assert half == pytest.approx(2.776 * math.sqrt(2.5) / math.sqrt(5), abs=1e-3)
    assert math.isnan(ex.mean_ci([4.0])[1])


def test_welch_and_holm():
    assert ex.welch_p([1, 2, 3, 4, 5], [3, 4, 5, 6, 7]) == pytest.approx(0.0805, abs=1e-3)  # t = -2, df = 8
    assert ex.welch_p([1, 1, 1], [1, 1, 1]) == 1.0
    assert ex.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_sign_test():
    assert sign_test(5, 0) == pytest.approx(2 / 32)
    assert sign_test(0, 0) == 1.0
    assert sign_test(3, 3) == 1.0


def test_aggregate_aligns_steps_and_bands_contain_mean():
    runs = [{"step": [10, 20, 30], "population": [1, 2, 3]},
            {"step": [10, 20], "population": [3, 4]}]
    agg = ex.aggregate(runs)["population"]
    assert agg["step"] == [10, 20] and agg["mean"] == [2, 3] and agg["n"] == 2
    assert all(lo < m < hi for lo, m, hi in zip(agg["lo"], agg["mean"], agg["hi"]))


def test_species_kinds_merges_numbered_clusters_and_drops_tiny_ones():
    assert ex.species_kinds("grazer:50|grazer 2:30|hunter:18|scout:2") == {"grazer", "hunter"}
    assert ex.species_kinds("") == set()


# ------------------------------------------------------------------ experiment end to end

@pytest.fixture(scope="module", autouse=True)
def few_charts():
    """Rendering every chart is the slow part; three are enough to exercise the code."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ex, "CHART_METRICS", ["population", "max_generation", "spawned"])
        yield


@pytest.fixture(scope="module")
def experiment(tmp_path_factory, few_charts):
    root = tmp_path_factory.mktemp("exp")
    spec = {"name": "tiny", "steps": 40, "every": 20, "seeds": [1, 2], "base": TINY,
            "variants": {"old": {"sigma_tau": 0, "neuron_reset_prob": 0}, "new": {}},
            "tournament": {"trials": 1, "steps": 20, "each": 10}}
    path = root / "tiny.json"
    path.write_text(json.dumps(spec))
    logs: list[str] = []
    out = ex.run_experiment(path, workers=1, runs_dir=root / "runs", log=logs.append)
    return path, out, logs


def test_experiment_writes_runs_summary_and_report(experiment):
    _, out, _ = experiment
    for v in ("old", "new"):
        for s in (1, 2):
            p = ex.run_paths(out, v, s)
            assert p["csv"].exists() and p["ckpt"].exists() and p["meta"].exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["completed"] == {"old": [1, 2], "new": [1, 2]}
    assert summary["final"]["new"]["population"]["n"] == 2
    assert summary["findings"][0].startswith("<b>Verdict")
    assert summary["tournament"][0]["trials"] == 2
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "What this shows" in html and html.count("data:image/png;base64,") >= 4   # 3 charts + tournament


def test_experiment_resumes_without_rerunning(experiment):
    path, out, _ = experiment
    csv = ex.run_paths(out, "old", 1)["csv"]
    before = csv.stat().st_mtime_ns
    ex.run_paths(out, "new", 2)["meta"].unlink()      # pretend one run was interrupted
    logs: list[str] = []
    ex.run_experiment(path, workers=1, runs_dir=out.parent, log=logs.append)
    assert any("1 runs to do (3 already finished)" in line for line in logs)
    assert any("0 head-to-head contests" in line for line in logs)
    assert csv.stat().st_mtime_ns == before


def test_report_only_does_not_run(experiment):
    path, out, _ = experiment
    logs: list[str] = []
    ex.run_experiment(path, workers=1, report_only=True, runs_dir=out.parent, log=logs.append)
    assert not any("runs to do" in line for line in logs)
    assert (out / "report.html").exists()


def test_spec_rejects_unknown_keys(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"name": "x", "steps": 1, "seeds": [1], "variants": {"a": {"nope": 1}}}))
    with pytest.raises(ValueError, match="nope"):
        ex.load_spec(p)


# ------------------------------------------------------------------ tournament

def test_head_to_head_assigns_every_survivor_to_a_side():
    src = World(Config(seed=3, **TINY))
    a, b = head_to_head(src, None, trial_seed=5, steps=15, n_each=12, overrides=TINY)
    assert 0 < a + b and a <= 40 and b <= 40

    w = World(Config(seed=1, **{**TINY, "initial_particles": 20, "min_population": 0}))
    w.energy[w.alive] = 10.0   # force a round of births: children inherit their parent's side
    w._births()
    assert side_counts(w, 10) == (20, 20)
    assert torch.all(w.generation[w.alive] <= 1)


# ------------------------------------------------------------------ discovery

def test_score_penalises_worlds_kept_alive_by_spawning():
    rows = [{"step": s, "population": 100, "births": 0, "spawned": s, "mean_generation": 0, "n_species": 3,
             "evolved_lineages": 5, "predation_share": 0.1, "bonded_fraction": 0.2} for s in (50, 100)]
    fitness, parts, desc = discovery.score(rows, 100, min_population=80)
    assert fitness == 0 and parts["autonomy"] == 0
    assert desc == pytest.approx((0.1, 0.2))
    for r in rows:
        r["births"], r["spawned"] = r["step"], 0
    assert discovery.score(rows, 100, min_population=80)[0] > 0


def test_decode_encode_roundtrip_and_cells():
    unit = [0.25] * len(discovery.PARAMS)
    back = discovery.encode(discovery.decode(unit))
    assert back == pytest.approx(unit, abs=0.01)
    assert discovery.cell_of((0.0, 1.0), 6) == (0, 5)
    assert discovery.cell_of((0.04, 0.03), 6) == (1, 1)   # sqrt-spaced bins


def test_discovery_runs_archives_and_resumes(tmp_path):
    out = tmp_path / "disc"
    kw = dict(workers=1, steps=30, every=15, out=out, grid=3, fixed=TINY, log=lambda s: None)
    discovery.discover(evals=3, **kw)
    archive = json.loads((out / "archive.json").read_text())
    assert archive["evaluations"] == 3 and archive["elites"]
    best = archive["elites"][0]
    assert "--set" in best["command"] and "capacity=120" in best["command"]
    assert (tmp_path / best["checkpoint"]).exists() or (out / "ckpt" / f"e{best['id']:04d}.pt").exists()
    html = (out / "gallery.html").read_text(encoding="utf-8")
    assert html.count("data:image/png;base64,") >= 1 + 2 * len(archive["elites"])
    # only elites keep checkpoints
    assert len(list((out / "ckpt").glob("*.pt"))) == len(archive["elites"])

    logs: list[str] = []
    discovery.discover(evals=3, **{**kw, "log": logs.append})
    assert any("3 evaluations already done, 0 to go" in line for line in logs)
