import torch

from primordial import Config, World
from primordial.genome import mutate


def world_with_twins(**kw):
    """A world with one parent and a cloned child slot, ready for mutate()."""
    cfg = Config(capacity=10, initial_particles=1, min_population=0, food_capacity=10, **kw)
    w = World(cfg)
    parent, child = torch.tensor([0]), torch.tensor([1])
    w.genome.copy(child, parent)
    return w, parent, child


def off(**kw):
    """All mutation mechanisms disabled unless re-enabled via kw."""
    base = dict(mutation_rate=0.0, sigma_tau=0.0, neuron_reset_prob=0.0)
    return {**base, **kw}


def test_fixed_sigma_when_self_adaptation_off():
    w, p, c = world_with_twins(**off(mutation_rate=0.5))
    mutate(w.genome, c, w.cfg, w.gen)
    assert torch.allclose(w.genome.sigma[c], torch.full((1, 3), w.cfg.mutation_scale))


def test_self_adaptation_moves_sigma_within_bounds():
    w, p, c = world_with_twins(**off(sigma_tau=0.5, sigma_min=0.05, sigma_max=0.3))
    changed = False
    for _ in range(50):
        mutate(w.genome, c, w.cfg, w.gen)
        s = w.genome.sigma[c]
        assert (s >= 0.05 - 1e-6).all() and (s <= 0.3 + 1e-6).all()
        changed |= not torch.allclose(s, w.genome.sigma[p])
    assert changed


def test_per_layer_rates_can_freeze_perception():
    w, p, c = world_with_twins(**off(mutation_rate=0.5, perception_rate_mult=0.0, behaviour_rate_mult=1.0))
    mutate(w.genome, c, w.cfg, w.gen)
    assert torch.equal(w.genome.w1[c], w.genome.w1[p])        # perception frozen
    assert not torch.equal(w.genome.w2[c], w.genome.w2[p])    # behaviour mutated


def test_neuron_reset_rewrites_exactly_one_hidden_neuron():
    w, p, c = world_with_twins(**off(neuron_reset_prob=1.0))
    mutate(w.genome, c, w.cfg, w.gen)
    diff_cols = (w.genome.w1[c] != w.genome.w1[p]).any(dim=1).squeeze(0)   # per hidden unit
    assert int(diff_cols.sum()) == 1
    h = int(diff_cols.nonzero())
    other = torch.ones(w.cfg.hidden, dtype=torch.bool)
    other[h] = False
    assert torch.equal(w.genome.w2[c][0, other], w.genome.w2[p][0, other])
    assert not torch.equal(w.genome.w2[c][0, h], w.genome.w2[p][0, h])


def test_memory_is_bounded_and_reset_for_newborns():
    w = World(Config(capacity=100, initial_particles=40, food_capacity=50))
    for _ in range(30):
        w.step()
    assert w.memory.abs().max() <= 1
    assert w.memory[w.alive].abs().sum() > 0
    parent = int(w.alive.nonzero()[0])
    w.energy[parent] = 5.0
    before = set(w.alive.nonzero().squeeze(1).tolist())
    w._births()
    born = set(w.alive.nonzero().squeeze(1).tolist()) - before
    assert born and all(w.memory[i].abs().sum() == 0 for i in born)


def test_food_patches_wander_and_stay_on_torus():
    w = World(Config(capacity=50, initial_particles=10, food_capacity=50, patch_speed=0.01))
    start = w.patch_pos.clone()
    for _ in range(200):
        w.step()
    assert not torch.allclose(start, w.patch_pos)
    assert ((w.patch_pos >= 0) & (w.patch_pos < 1)).all()
