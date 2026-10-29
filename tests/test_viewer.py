"""Headless smoke tests for the renderer, viewer panels, god tools and timelapse."""

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import pytest

from primordial import Config, World
from primordial.render import COLOR_MODES, Renderer, TreePanel, draw_brain
from primordial.timelapse import add_subparser, run_timelapse
from primordial.viewer import SLIDERS, TOOLS, Viewer


def small(**kw) -> Config:
    return Config(**{"capacity": 300, "initial_particles": 80, "food_capacity": 150,
                     "species_interval": 20, **kw})


@pytest.fixture(scope="module")
def world():
    pygame.init()
    w = World(small(seed=1))
    for _ in range(120):
        w.step()
    # Make sure the optional marks have something to draw.
    a = w.alive.nonzero().squeeze(1)
    w.bonds[a[0], 0], w.bonds[a[1], 0] = a[1], a[0]
    w.aggression[a[:5]] = 0.8
    w.signal[a[5:10]] = -0.9
    w.pheromone[0, 10:20, 10:20] = 1.0
    return w


def key(v: Viewer, k: int) -> None:
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=k))


def test_renderer_every_mode_and_overlay(world):
    r = Renderer(240, world.cfg.world_size)
    surf = pygame.Surface((240, 240))
    for mode in COLOR_MODES:
        r.color_mode = mode
        for trails in (True, False):
            r.trails = trails
            r.show_pheromone = r.show_bonds = r.show_patches = not trails
            r.draw(surf, world, selected=int(world.alive.nonzero()[0, 0]))
    assert surf.get_bounding_rect().w > 0
    # Pheromone overlay actually lights the deposited corner.
    r.trails, r.show_pheromone = False, True
    r.draw(surf, world)
    lit = sum(surf.get_at((int((15 / 96) * 240), int((15 / 96) * 240)))[:3])
    assert lit > 30


def test_viewer_panels_and_keys(world):
    v = Viewer(world, size=640, steps_per_frame=1, save_path="unused.pt")
    screen = pygame.Surface(v.window_size)
    v.update()
    v.draw(screen)
    v.select(int(world.alive.nonzero()[0, 0]))
    v.draw(screen)                                  # inspector + brain
    key(v, pygame.K_f)
    v.draw(screen)                                  # family tree with ancestry highlight
    assert v.tree.where and v.selected_uid in v.tree.where
    key(v, pygame.K_g)
    v.draw(screen)                                  # sliders
    for k in (pygame.K_c, pygame.K_c, pygame.K_t, pygame.K_o, pygame.K_b, pygame.K_p, pygame.K_h,
              pygame.K_SPACE, pygame.K_UP, pygame.K_DOWN):
        key(v, k)
        v.update()
        v.draw(screen)
    assert v.paused and v.renderer.color_mode == COLOR_MODES[2]


def test_brain_and_tree_direct(world):
    surf = pygame.Surface((340, 420))
    i = int(world.alive.nonzero()[0, 0])
    draw_brain(surf, surf.get_rect(), world, i)
    tree = TreePanel()
    tree.draw(surf, surf.get_rect(), world, int(world.uid[i]))
    assert len(tree.where) >= world.population


def test_god_tools(world):
    v = Viewer(world, size=400, steps_per_frame=1)
    world.food_alive[:] = False
    v.tool = TOOLS[1]
    v.use_tool(0.5, 0.5)
    assert int(world.food_alive.sum()) > 0

    i = int(world.alive.nonzero()[0, 0])
    x, y = world.pos[i].tolist()
    key(v, pygame.K_3)
    assert v.tool == "meteor"
    v.use_tool(x, y)
    assert not world.alive[i] and v.effects

    key(v, pygame.K_1)
    j = int(world.alive.nonzero()[0, 0])
    sx, sy = v.renderer.to_screen(*world.pos[j].tolist())
    v.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(sx), int(sy)), button=1))
    assert v.selected is not None

    before = world.food_multiplier
    key(v, pygame.K_RIGHTBRACKET)
    assert world.food_multiplier == before + 0.25
    for _ in range(20):
        key(v, pygame.K_LEFTBRACKET)
    assert world.food_multiplier == 0.0

    # Drag every slider to its far right: config takes the slider's max.
    key(v, pygame.K_g)
    for k, s in enumerate(SLIDERS):
        t = v.sliders.track(k)
        v.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(t.x + 2, t.centery), button=1))
        v.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(t.right + 50, t.centery), rel=(0, 0), buttons=(1, 0, 0)))
        v.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(t.right + 50, t.centery), button=1))
        assert getattr(world.cfg, s.key) == pytest.approx(s.hi)
    world.step()                                     # still runs with extreme settings


def test_selection_dropped_when_slot_recycled(world):
    v = Viewer(world, size=400, steps_per_frame=1)
    i = int(world.alive.nonzero()[0, 0])
    v.select(i)
    world.uid[i] = 10**9                             # simulate the slot being reused by a newborn
    v.paused = True
    v.update()
    assert v.selected is None


def test_timelapse_writes_video(tmp_path):
    w = World(small(seed=2))
    path = run_timelapse(w, steps=12, every=3, out_path=str(tmp_path / "lapse.mp4"), size=128)
    assert path.exists() and path.stat().st_size > 0
    assert w.step_count == 12

    gif = run_timelapse(w, steps=4, every=2, out_path=str(tmp_path / "lapse.gif"), size=64)
    assert gif.suffix == ".gif" and gif.stat().st_size > 0


def test_timelapse_subparser_parses():
    import argparse

    p = argparse.ArgumentParser()
    add_subparser(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["timelapse", "--steps", "10", "--every", "2", "--set", "capacity=100", "--size", "96"])
    assert a.cmd == "timelapse" and a.steps == 10 and a.set == ["capacity=100"] and callable(a.func)


def test_recording_toggle(world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    v = Viewer(world, size=200, steps_per_frame=1)
    screen = pygame.Surface(v.window_size)
    key(v, pygame.K_r)
    for _ in range(3):
        v.draw(screen)
    path = v.recorder.path
    key(v, pygame.K_r)
    assert v.recorder is None and path.exists() and path.stat().st_size > 0
