"""Tests for the ecosystem mechanics: grid search, bodies, bonds, mating, learning, lineage, species."""

import torch

from primordial import Config, World
from primordial.genome import crossover
from primordial.lineage import Lineage
from primordial.spatial import neighbours


def tiny(**kw) -> Config:
    return Config(**{"capacity": 300, "initial_particles": 60, "food_capacity": 100, "min_population": 0, **kw})


def brute_force(q, t, radius, world, exclude_self):
    d = t[None] - q[:, None]
    d = d - torch.round(d / world) * world
    hit = d.norm(dim=-1) <= radius
    if exclude_self:
        hit.fill_diagonal_(False)
    return hit


def test_grid_neighbours_match_brute_force():
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(3000, 2, generator=g) * 2.0    # big enough to take the grid path
    nb = neighbours(pts, pts, 0.1, 2.0, exclude_self=True)
    brute = brute_force(pts, pts, 0.1, 2.0, True)
    assert torch.equal(nb.valid.sum(1), brute.sum(1))
    for i in (0, 17, 1234):
        assert set(nb.cand[i][nb.valid[i]].tolist()) == set(brute[i].nonzero().squeeze(1).tolist())
    assert torch.isinf(nb.dist[~nb.valid]).all()


def test_dense_neighbours_match_brute_force_across_the_wrap():
    q = torch.tensor([[0.01, 0.01], [0.5, 0.5]])
    t = torch.tensor([[0.99, 0.99], [0.03, 0.01], [0.5, 0.58], [0.2, 0.2]])
    nb = neighbours(q, t, 0.05, 1.0)
    assert set(nb.cand[0][nb.valid[0]].tolist()) == {0, 1}      # 0 is across the corner
    assert nb.valid[1].sum() == 0
    i = (nb.cand[0] == 0) & nb.valid[0]
    assert torch.isclose(nb.dx[0][i], torch.tensor([-0.02])).all()


def test_neighbours_handle_empty_targets():
    nb = neighbours(torch.rand(5, 2), torch.zeros(0, 2), 0.1, 1.0)
    assert nb.cand.shape == (5, 0)


def test_body_genes_map_to_ranges():
    w = World(tiny())
    w.genome.body[:3] = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    size, speed, sense = w.body(torch.arange(3))
    assert torch.allclose(size, torch.tensor([w.cfg.size_min, w.cfg.size_max, (w.cfg.size_min + w.cfg.size_max) / 2]))
    assert speed[1] == w.cfg.speed_max and sense[0] == w.cfg.sense_min


def _duel(big_attacker: bool) -> float:
    w = World(tiny(initial_particles=2, plasticity=0))
    a, b = w.alive.nonzero().squeeze(1).tolist()
    w.pos[a], w.pos[b] = torch.tensor([0.5, 0.5]), torch.tensor([0.505, 0.5])
    w.genome.color[a], w.genome.color[b] = torch.tensor([1.0, 0, 0]), torch.tensor([0, 0, 1.0])
    w.genome.body[a, 0], w.genome.body[b, 0] = (1.0, 0.0) if big_attacker else (0.0, 1.0)
    w.genome.w2[a] = 0
    w.genome.b2[a] = 0
    w.genome.b2[a, 2] = 10    # a: maximal aggression
    w.genome.w2[b] = 0
    w.genome.b2[b] = 0
    w.genome.b2[b, 2] = -10   # b: passive
    w.step()
    return float(w.prey_gained[a])


def test_small_attackers_cannot_eat_big_victims():
    assert _duel(big_attacker=False) == 0
    assert _duel(big_attacker=True) > 0


def test_bonds_are_symmetric_and_cleared_on_death():
    w = World(tiny(initial_particles=40))
    w.genome.bud[:] = 0.5
    for _ in range(60):
        w.step()
        for i in w.alive.nonzero().squeeze(1).tolist():
            for j in w.bonds[i].tolist():
                if j >= 0:
                    assert w.alive[j] and i in w.bonds[j].tolist()
    i, j = w.alive.nonzero().squeeze(1)[:2].tolist()
    w.bonds[i, -1], w.bonds[j, -1] = j, i
    w.kill(torch.arange(w.cfg.capacity) == j)
    assert j not in w.bonds[i].tolist()


def test_budding_child_is_bonded_to_parent():
    w = World(tiny(initial_particles=1))
    p = int(w.alive.nonzero()[0])
    w.genome.bud[p] = 1.0
    w.energy[p] = 10.0
    w._births()
    child = [i for i in w.alive.nonzero().squeeze(1).tolist() if i != p][0]
    assert child in w.bonds[p].tolist() and p in w.bonds[child].tolist()


def test_mating_mixes_genes_and_records_mate():
    w = World(tiny(initial_particles=2, mate_prob=1.0))
    a, b = w.alive.nonzero().squeeze(1).tolist()
    w.pos[a], w.pos[b] = torch.tensor([0.5, 0.5]), torch.tensor([0.52, 0.5])
    w.genome.color[a] = torch.tensor([0.5, 0.5, 0.5])
    w.genome.color[b] = torch.tensor([0.5, 0.5, 0.5])
    w.energy[a], w.energy[b] = 10.0, 1.0
    w._births()
    assert w.mated_births == 1
    child = [i for i in w.alive.nonzero().squeeze(1).tolist() if i not in (a, b)][0]
    cu = int(w.uid[child])
    assert w.lineage.parent[cu] == int(w.uid[a]) and w.lineage.mate[cu] == int(w.uid[b])
    assert w.energy[b] < 1.0      # the mate contributed energy


def test_crossover_inherits_whole_neurons():
    w = World(tiny(initial_particles=3))
    a, b, c = w.alive.nonzero().squeeze(1)[:3]
    w.genome.copy(c[None], a[None])
    crossover(w.genome, c[None], b[None], w.gen)
    for h in range(w.cfg.hidden):
        from_a = torch.equal(w.genome.w1[c, :, h], w.genome.w1[a, :, h])
        from_b = torch.equal(w.genome.w1[c, :, h], w.genome.w1[b, :, h])
        assert from_a or from_b
        src = a if from_a else b    # outgoing weights travel with the incoming ones
        assert torch.equal(w.genome.w2[c, h], w.genome.w2[src, h])


def test_plastic_weights_learn_and_stay_bounded():
    w = World(tiny(initial_particles=40, plasticity=1.0))
    for _ in range(100):
        w.step()
    p = w.plastic[w.alive]
    assert p.abs().sum() > 0 and p.abs().max() <= w.cfg.plastic_clip
    w2 = World(tiny(initial_particles=40, plasticity=0.0))
    for _ in range(20):
        w2.step()
    assert w2.plastic.abs().sum() == 0


def test_pheromone_spreads_and_decays():
    w = World(tiny(initial_particles=0))
    w.pheromone[0, 10, 10] = 1.0
    w._diffuse_pheromone()
    assert w.pheromone[0, 11, 10] > 0 and w.pheromone[0, 10, 10] < 1.0
    total = float(w.pheromone.sum())
    w._diffuse_pheromone()
    assert float(w.pheromone.sum()) < total


def test_meteor_kills_area_and_leaves_food():
    w = World(tiny(initial_particles=100))
    w.food_alive[:] = False
    killed = w.meteor(0.7, 0.7, radius=0.3)
    d = w._wrap_delta(w.pos - torch.tensor([0.7, 0.7])).norm(dim=1)
    assert killed > 0 and not (w.alive & (d < 0.3)).any()
    assert w.food_alive.sum() > 0


def test_lineage_living_tree_collapses_chains():
    lin = Lineage()
    root = lin.register([-1], [-1], 0, [(0, 0, 0)], [1.0])[0]
    a = lin.register([root], [-1], 1, [(0, 0, 0)], [1.0])[0]
    b, c = lin.register([a, a], [-1, -1], 2, [(0, 0, 0)] * 2, [1.0] * 2)
    assert {n["uid"] for n in lin.living_tree([b, c])} == {root, a, b, c}   # a is a branch point
    lin.record_deaths([c], 3)
    assert {n["uid"] for n in lin.living_tree([b])} == {root, b}            # chain collapsed
    assert lin.ancestors(b) == [b, a, root]


def test_species_are_detected_and_named():
    w = World(tiny(initial_particles=120, species_interval=20))
    for _ in range(40):
        w.step()
    counts = w.species_counts()
    assert counts and sum(counts.values()) == w.population


def test_long_run_is_stable():
    w = World(Config(capacity=600, initial_particles=200, food_capacity=200, seed=5))
    for _ in range(400):
        w.step()
    a = w.alive
    for t in (w.pos, w.vel, w.energy, w.plastic, w.pheromone):
        assert torch.isfinite(t).all()
    assert ((w.pos[a] >= 0) & (w.pos[a] < w.cfg.world_size)).all()
