"""Real-time pygame window onto a World."""

import time
from collections import deque

import pygame
import torch

from .world import World

HELP = [
    "SPACE pause   UP/DOWN speed   C colours   T trails",
    "L-click inspect   R-click drop food   P patches",
    "H hide HUD   S save   Q quit",
]
BG = (12, 12, 18)
TEXT = (210, 210, 210)
DIM = (120, 120, 130)
TYPE_PALETTE = torch.tensor([[230, 80, 80], [80, 170, 240], [240, 200, 60], [170, 100, 230],
                             [90, 220, 140], [240, 130, 40], [220, 220, 220], [120, 120, 255]])


def lineage_rgb(color: torch.Tensor) -> torch.Tensor:
    """Genome colour -> vivid display colour (stretch away from grey so lineages pop)."""
    mean = color.mean(1, keepdim=True)
    vivid = (mean + (color - mean) * 2.2).clamp(0, 1)
    return (vivid * 200 + 55).int()


class Graph:
    """Tiny rolling line chart of a few stats."""

    def __init__(self, series: dict[str, tuple[int, int, int]], length: int = 240):
        self.series = series
        self.data = {k: deque(maxlen=length) for k in series}

    def push(self, stats: dict) -> None:
        for k in self.series:
            self.data[k].append(float(stats.get(k, 0)))

    def draw(self, screen, font, rect: pygame.Rect) -> None:
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill((20, 20, 30, 200))
        screen.blit(panel, rect.topleft)
        for i, (k, col) in enumerate(self.series.items()):
            values = self.data[k]
            if len(values) < 2:
                continue
            hi = max(values) or 1
            n = values.maxlen
            pts = [(rect.x + 4 + (rect.w - 8) * j / (n - 1),
                    rect.bottom - 4 - (rect.h - 22) * v / hi) for j, v in enumerate(values)]
            pygame.draw.lines(screen, col, False, pts, 2)
            screen.blit(font.render(f"{k} {values[-1]:g}", True, col), (rect.x + 6 + i * 120, rect.y + 3))


def run_viewer(world: World, size: int = 820, steps_per_frame: int = 2, save_path: str = "checkpoint.pt") -> None:
    pygame.init()
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("primordial")
    font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()
    scale = size / world.cfg.world_size

    # Particles draw onto their own layer so trails fade there without smearing the HUD.
    layer = pygame.Surface((size, size))
    layer.fill(BG)
    fade = pygame.Surface((size, size), pygame.SRCALPHA)
    fade.fill((*BG, 60))
    graph = Graph({"population": (230, 230, 230), "lineages": (90, 200, 255)})

    paused, color_by_type, trails, show_hud, show_patches = False, False, True, True, True
    flash = ("", 0.0)
    selected, selected_age = None, -1
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif key == pygame.K_SPACE:
                    paused = not paused
                elif key == pygame.K_UP:
                    steps_per_frame = min(steps_per_frame * 2, 64)
                elif key == pygame.K_DOWN:
                    steps_per_frame = max(steps_per_frame // 2, 1)
                elif key == pygame.K_c:
                    color_by_type = not color_by_type
                elif key == pygame.K_t:
                    trails = not trails
                elif key == pygame.K_h:
                    show_hud = not show_hud
                elif key == pygame.K_p:
                    show_patches = not show_patches
                elif key == pygame.K_s:
                    world.save(save_path)
                    flash = (f"saved {save_path}", time.time())
            elif event.type == pygame.MOUSEBUTTONDOWN:
                wx, wy = event.pos[0] / scale, event.pos[1] / scale
                if event.button == 1:
                    selected = _pick(world, wx, wy, radius=20 / scale)
                    selected_age = int(world.age[selected]) if selected is not None else -1
                elif event.button == 3:
                    placed = world.drop_food(wx, wy)
                    flash = (f"dropped {placed} food", time.time())

        if not paused:
            for _ in range(steps_per_frame):
                world.step()

        # A selected particle is "gone" if its slot died or was recycled for a newborn.
        if selected is not None:
            age = int(world.age[selected])
            if not world.alive[selected] or age < selected_age:
                flash, selected = ("selected particle died", time.time()), None
            else:
                selected_age = age

        if trails:
            layer.blit(fade, (0, 0))
        else:
            layer.fill(BG)

        a = world.alive
        pos = (world.pos[a] * scale).tolist()
        energy = world.energy[a].tolist()
        aggression = world.aggression[a].tolist()
        if color_by_type:
            colors = TYPE_PALETTE[world.genome.ptype[a] % len(TYPE_PALETTE)].tolist()
        else:
            colors = lineage_rgb(world.genome.color[a]).tolist()
        for (x, y), e, agg, col in zip(pos, energy, aggression, colors):
            r = 3 + min(e, 3.0) * 1.5
            pygame.draw.circle(layer, col, (x, y), r)
            if agg > 0.3:
                pygame.draw.circle(layer, (255, 60, 60), (x, y), r + 2, 1)

        # Overlays: crisp every frame.
        screen.blit(layer, (0, 0))
        if show_patches:
            for x, y in (world.patch_pos * scale).tolist():
                pygame.draw.circle(screen, (60, 52, 30), (x, y), world.cfg.patch_radius * scale * 2, 1)
        for x, y in (world.food[world.food_alive] * scale).tolist():
            screen.fill((150, 120, 50), (int(x) - 1, int(y) - 1, 2, 2))

        stats = world.stats()
        if not paused:
            graph.push(stats)

        if selected is not None:
            sx, sy = (world.pos[selected] * scale).tolist()
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 14, 1)
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), world.cfg.interaction_radius * scale, 1)

        if show_hud:
            _draw_hud(screen, font, world, stats, steps_per_frame, paused, color_by_type, flash)
            graph.draw(screen, font, pygame.Rect(size - 300, size - 110, 290, 100))
            if selected is not None:
                _draw_inspector(screen, font, world, selected, pygame.Rect(10, size - 170, 260, 160))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def _pick(world: World, x: float, y: float, radius: float):
    if not world.alive.any():
        return None
    delta = world._wrap_delta(world.pos - torch.tensor([x, y]))
    dist = delta.norm(dim=1)
    dist[~world.alive] = float("inf")
    d, i = dist.min(0)
    return int(i) if d < radius else None


def _draw_hud(screen, font, world, s, speed, paused, color_by_type, flash) -> None:
    lines = [
        f"step {s['step']:>7}   pop {s['population']:>4}   food {s.get('food', 0):>3}   x{speed}"
        + ("   PAUSED" if paused else ""),
        f"gen avg {s.get('mean_generation', 0):>6}  max {s.get('max_generation', 0):>4}   "
        f"lineages {s.get('lineages', 0)}   aggr {s.get('mean_aggression', 0):.3f}",
        f"mutation sigma  perception {s.get('sigma_perception', 0):.3f}  "
        f"behaviour {s.get('sigma_behaviour', 0):.3f}  physics {s.get('sigma_physics', 0):.3f}",
    ]
    backing = pygame.Surface((540, 6 + 17 * len(lines) + 16 * (len(HELP) + 1) + 4), pygame.SRCALPHA)
    backing.fill((12, 12, 18, 170))
    screen.blit(backing, (0, 0))
    y = 6
    for line in lines:
        screen.blit(font.render(line, True, TEXT), (8, y))
        y += 17
    for line in HELP + [f"[{'type' if color_by_type else 'lineage'} colours]"]:
        screen.blit(font.render(line, True, DIM), (8, y))
        y += 16
    if flash[0] and time.time() - flash[1] < 2:
        screen.blit(font.render(flash[0], True, (255, 220, 120)), (8, y + 4))


def _draw_inspector(screen, font, world: World, i: int, rect: pygame.Rect) -> None:
    g = world.genome
    sigma = g.sigma[i].tolist()
    memory = world.memory[i].tolist()
    lines = [
        f"particle #{i}",
        f"generation {int(world.generation[i])}   age {int(world.age[i])}",
        f"energy {float(world.energy[i]):.2f}   type {int(g.ptype[i])}",
        f"aggression {float(world.aggression[i]):.2f}",
        f"sigma  P {sigma[0]:.3f}  B {sigma[1]:.3f}  Ph {sigma[2]:.3f}",
        f"memory {memory[0]:+.2f} {memory[1]:+.2f}",
        "attraction " + " ".join(f"{v:+.1f}" for v in g.attraction[i].tolist()),
    ]
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    panel.fill((20, 20, 30, 210))
    screen.blit(panel, rect.topleft)
    swatch = lineage_rgb(g.color[i : i + 1])[0].tolist()
    pygame.draw.circle(screen, swatch, (rect.right - 18, rect.y + 16), 8)
    for k, line in enumerate(lines):
        screen.blit(font.render(line, True, TEXT), (rect.x + 8, rect.y + 6 + k * 21))
