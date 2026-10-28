import torch

from primordial import Config, World
from primordial.genome import N_INPUTS, N_OUTPUTS, think
from primordial.world import particle_life_force


def small(**kw) -> Config:
    return Config(**{"capacity": 200, "initial_particles": 50, "food_capacity": 100, **kw})


def test_brains_have_expected_shapes():
    w = World(small())
    idx = w.alive.nonzero().squeeze(1)
    hidden, out = think(w.genome.take(idx), torch.zeros(len(idx), N_INPUTS), w.plastic[idx])
    assert hidden.shape == (len(idx), w.cfg.hidden)
    assert out.shape == (len(idx), N_OUTPUTS)
    assert out.abs().max() <= 1


def test_force_kernel_repels_inside_core_and_vanishes_outside():
    r = torch.tensor([0.0, 0.1, 0.65, 1.0, 1.5])
    f = particle_life_force(r, torch.ones_like(r), beta=0.3)
    assert f[0] == -1 and f[1] < 0      # core repulsion
    assert torch.isclose(f[2], torch.tensor(1.0))  # peak attraction at (1 + beta) / 2
    assert f[3] == 0 and f[4] == 0      # out of range


def test_positions_stay_on_torus_and_values_finite():
    w = World(small())
    for _ in range(200):
        w.step()
    p = w.pos[w.alive]
    assert ((p >= 0) & (p < w.cfg.world_size)).all()
    assert torch.isfinite(w.energy).all() and torch.isfinite(w.vel).all()


def test_same_seed_same_history():
    a, b = World(small(seed=7)), World(small(seed=7))
    for _ in range(100):
        a.step()
        b.step()
    assert torch.equal(a.alive, b.alive)
    assert torch.allclose(a.pos, b.pos)


def test_birth_splits_energy_and_mutates_only_the_child():
    w = World(small(initial_particles=1, min_population=0))
    parent = int(w.alive.nonzero()[0])
    w.energy[parent] = 4.0
    before = w.genome.take(torch.tensor([parent]))
    w._births()

    assert w.population == 2
    alive = w.alive.nonzero().squeeze(1)
    child = int(alive[alive != parent][0])
    expected = 4.0 * (1 - w.cfg.birth_cost) / 2
    assert torch.isclose(w.energy[parent], torch.tensor(expected))
    assert torch.isclose(w.energy[child], torch.tensor(expected))
    assert w.generation[child] == 1
    assert torch.equal(w.genome.w1[parent], before.w1[0])           # parent untouched
    assert not torch.equal(w.genome.color[child], w.genome.color[parent])  # child drifted


def test_eating_moves_food_energy_into_particle():
    w = World(small(initial_particles=1, min_population=0, base_metabolism=0, move_cost=0, aggression_cost=0,
                    speed_upkeep=0, sense_upkeep=0, signal_cost=0, deposit_cost=0, plasticity=0))
    i = int(w.alive.nonzero()[0])
    w.food_alive[:] = False
    w.food[0] = w.pos[i]
    w.food_alive[0] = True
    w.vel[i] = 0
    e0 = float(w.energy[i])
    w._update_living(torch.tensor([i]))
    assert not w.food_alive[0]
    assert torch.isclose(w.energy[i], torch.tensor(e0 + w.cfg.food_energy))


def test_starved_particles_die_and_become_food():
    w = World(small(min_population=0))
    w.food_alive[:] = False
    w.energy[w.alive] = -1
    n = w.population
    w._deaths()
    assert w.population == 0
    assert int(w.food_alive.sum()) == n


def test_population_never_drops_below_minimum():
    w = World(small(min_population=10, base_metabolism=1.0))  # everyone starves instantly
    w.step()
    assert w.population >= 10


def test_checkpoint_roundtrip(tmp_path):
    w = World(small())
    for _ in range(20):
        w.step()
    path = tmp_path / "ck.pt"
    w.save(str(path))
    r = World.load(str(path))
    for _ in range(10):
        w.step()
        r.step()
    assert torch.equal(w.alive, r.alive)
    assert torch.allclose(w.pos, r.pos)
