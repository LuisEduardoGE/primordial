"""Real-time pygame window onto a World: world view on the left, side panel on the right."""

import time
from dataclasses import dataclass

import pygame
import torch

from . import render
from .render import ACCENT, BG, DIM, TEXT, Graph, Renderer, TreePanel, panel, text
from .world import World

SIDE = 340               # side panel width
TOOLS = ("inspect", "food", "meteor")
HELP = ("SPACE pause   UP/DOWN speed   1 inspect  2 food  3 meteor   L/R-click use / drop food   [ ] climate",
        "C colours   T trails   O scent   B bonds   P patches   F tree   G sliders   R record   S save   H hud   Q quit")


@dataclass
class Slider:
    key: str          # Config field edited live
    lo: float
    hi: float
    fmt: str = "{:.3f}"

    def frac(self, cfg) -> float:
        return (getattr(cfg, self.key) - self.lo) / (self.hi - self.lo)

    def set_frac(self, cfg, f: float) -> None:
        setattr(cfg, self.key, self.lo + min(1.0, max(0.0, f)) * (self.hi - self.lo))


SLIDERS = [
    Slider("food_regrow_prob", 0.0, 0.02, "{:.4f}"),
    Slider("attack_rate", 0.0, 0.1),
    Slider("mutation_rate", 0.0, 0.5),
    Slider("climate_amplitude", 0.0, 1.0, "{:.2f}"),
    Slider("pheromone_decay", 0.0, 0.05, "{:.4f}"),
    Slider("bond_strength", 0.0, 120.0, "{:.1f}"),
    Slider("mate_prob", 0.0, 1.0, "{:.2f}"),
    Slider("plasticity", 0.0, 3.0, "{:.2f}"),
]


class SliderPanel:
    """Pygame-drawn sliders that live-edit world.cfg; drag a track to change a value."""

    ROW = 34

    def __init__(self, sliders: list[Slider]):
        self.sliders = sliders
        self.rect = pygame.Rect(0, 0, 330, 30 + self.ROW * len(sliders))
        self.active: Slider | None = None

    def track(self, k: int) -> pygame.Rect:
        return pygame.Rect(self.rect.x + 14, self.rect.y + 26 + k * self.ROW + 18, self.rect.w - 28, 6)

    def press(self, pos, cfg) -> bool:
        """Start a drag if `pos` hits a track (with a forgiving hit box). True if consumed."""
        if not self.rect.collidepoint(pos):
            return False
        for k, s in enumerate(self.sliders):
            if self.track(k).inflate(8, 18).collidepoint(pos):
                self.active = s
                self.drag(pos, cfg)
                break
        return True

    def drag(self, pos, cfg) -> None:
        if self.active is not None:
            t = self.track(self.sliders.index(self.active))
            self.active.set_frac(cfg, (pos[0] - t.x) / t.w)

    def release(self) -> None:
        self.active = None

    def draw(self, surface, cfg) -> None:
        panel(surface, self.rect, alpha=230, title="world sliders  (drag)")
        for k, s in enumerate(self.sliders):
            t = self.track(k)
            f = min(1.0, max(0.0, s.frac(cfg)))
            hot = s is self.active
            text(surface, s.key, (t.x, t.y - 16), ACCENT if hot else TEXT, 12)
            text(surface, s.fmt.format(getattr(cfg, s.key)), (t.right, t.y - 16), ACCENT, 12, bold=True,
                 anchor="topright")
            pygame.draw.rect(surface, (34, 38, 52), t, border_radius=3)
            pygame.draw.rect(surface, ACCENT if hot else (200, 160, 90), (t.x, t.y, max(3, t.w * f), t.h),
                             border_radius=3)
            pygame.draw.circle(surface, (240, 240, 245), (t.x + t.w * f, t.centery), 7 if hot else 6)


class Viewer:
    """All interactive state; `handle` events, `update` the sim, `draw` onto any surface.

    Kept separate from the window loop so tests can drive it headlessly.
    """

    def __init__(self, world: World, size: int = 820, steps_per_frame: int = 2, save_path: str = "checkpoint.pt"):
        self.world = world
        self.size = size
        self.steps_per_frame = steps_per_frame
        self.save_path = save_path
        self.renderer = Renderer(size, world.cfg.world_size)
        self.graph = Graph({"population": (230, 230, 235), "lineages": (90, 200, 255), "food": render.FOOD})
        self.tree = TreePanel(interval=1.0)
        self.sliders = SliderPanel(SLIDERS)
        self.sliders.rect.bottomleft = (10, size - 46)
        self.tool = "inspect"
        self.paused = False
        self.show_hud = True
        self.show_tree = False
        self.show_sliders = False
        self.selected: int | None = None
        self.selected_uid = -1
        self.recorder = None
        self.flash = ("", 0.0)
        self.effects: list[tuple[float, float, float, float]] = []   # meteor shockwaves: x, y, r, t0
        self.stats: dict = world.stats()
        self._stats_at = 0.0
        self.running = True

    @property
    def window_size(self) -> tuple[int, int]:
        return self.size + SIDE, self.size

    def say(self, msg: str) -> None:
        self.flash = (msg, time.time())

    # ------------------------------------------------------------------ input

    def handle(self, event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            self.key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.click(event.pos, event.button)
        elif event.type == pygame.MOUSEMOTION and self.sliders.active is not None:
            self.sliders.drag(event.pos, self.world.cfg)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.sliders.release()

    def key(self, key: int) -> None:
        r, w = self.renderer, self.world
        if key in (pygame.K_q, pygame.K_ESCAPE):
            self.running = False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_UP:
            self.steps_per_frame = min(self.steps_per_frame * 2, 64)
        elif key == pygame.K_DOWN:
            self.steps_per_frame = max(self.steps_per_frame // 2, 1)
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3):
            self.tool = TOOLS[key - pygame.K_1]
        elif key == pygame.K_c:
            self.say(f"colour by {r.cycle_colors()}")
        elif key == pygame.K_t:
            r.trails = not r.trails
        elif key == pygame.K_o:
            r.show_pheromone = not r.show_pheromone
        elif key == pygame.K_b:
            r.show_bonds = not r.show_bonds
        elif key == pygame.K_p:
            r.show_patches = not r.show_patches
        elif key == pygame.K_f:
            self.show_tree = not self.show_tree
        elif key == pygame.K_g:
            self.show_sliders = not self.show_sliders
        elif key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            delta = 0.25 if key == pygame.K_RIGHTBRACKET else -0.25
            w.food_multiplier = round(min(3.0, max(0.0, w.food_multiplier + delta)), 2)
            self.say(f"food growth x{w.food_multiplier:.2f}")
        elif key == pygame.K_h:
            self.show_hud = not self.show_hud
        elif key == pygame.K_r:
            self.toggle_recording()
        elif key == pygame.K_s:
            w.save(self.save_path)
            self.say(f"saved {self.save_path}")

    def click(self, pos, button: int) -> None:
        if self.show_sliders and self.show_hud and button == 1 and self.sliders.press(pos, self.world.cfg):
            return
        if pos[0] >= self.size:
            if button == 1 and self.show_tree:
                self.pick_from_tree(pos)
            return
        x, y = self.renderer.to_world(*pos)
        if button == 3:
            self.say(f"dropped {self.world.drop_food(x, y)} food")
        elif button == 1:
            self.use_tool(x, y)

    def use_tool(self, x: float, y: float) -> None:
        """Apply the current god tool at world coordinates (x, y)."""
        w = self.world
        if self.tool == "inspect":
            self.select(_pick(w, x, y, radius=20 / self.renderer.scale))
        elif self.tool == "food":
            self.say(f"dropped {w.drop_food(x, y)} food")
        elif self.tool == "meteor":
            radius = 0.08
            self.say(f"meteor: {w.meteor(x, y, radius)} killed")
            self.effects.append((x, y, radius, time.time()))

    def select(self, slot: int | None) -> None:
        self.selected = slot
        self.selected_uid = int(self.world.uid[slot]) if slot is not None else -1

    def pick_from_tree(self, pos) -> None:
        """Click a living leaf of the family tree to select that particle."""
        best, best_d = None, 8.0
        for uid, (x, y) in self.tree.where.items():
            d = abs(pos[1] - y) + 0.02 * abs(pos[0] - x)
            if d < best_d:
                best, best_d = uid, d
        if best is not None:
            slots = ((self.world.uid == best) & self.world.alive).nonzero()
            if len(slots):
                self.select(int(slots[0, 0]))

    def toggle_recording(self) -> None:
        from .timelapse import VideoWriter

        if self.recorder is None:
            self.recorder = VideoWriter(f"recordings/{time.strftime('%Y%m%d_%H%M%S')}.mp4", fps=30)
            self.say(f"recording -> {self.recorder.path}")
        else:
            path = self.recorder.close()
            self.say(f"saved {path} ({self.recorder.frames} frames)")
            self.recorder = None

    # ------------------------------------------------------------------ update

    def update(self) -> None:
        w = self.world
        if not self.paused:
            for _ in range(self.steps_per_frame):
                w.step()
        # Slots are recycled for newborns, so identity is the uid, not the slot.
        if self.selected is not None:
            if not bool(w.alive[self.selected]) or int(w.uid[self.selected]) != self.selected_uid:
                self.say("selected particle died")
                self.select(None)
        now = time.time()
        if now - self._stats_at > 0.25:      # stats() does many reductions; 4 Hz is plenty for a HUD
            self.stats = w.stats()
            self._stats_at = now
            if not self.paused:
                self.graph.push(self.stats)

    # ------------------------------------------------------------------ draw

    def draw(self, screen: pygame.Surface) -> None:
        self.renderer.draw(screen, self.world, self.selected)
        self._draw_effects(screen)
        if self.show_hud:
            self._draw_hud(screen)
            if self.show_sliders:
                self.sliders.draw(screen, self.world.cfg)
        self._draw_side(screen, pygame.Rect(self.size, 0, SIDE, self.size))
        if self.recorder is not None:
            self.recorder.add(screen)

    def _draw_effects(self, screen) -> None:
        now, keep = time.time(), []
        for x, y, r, t0 in self.effects:
            age = now - t0
            if age < 0.8:
                keep.append((x, y, r, t0))
                sx, sy = self.renderer.to_screen(x, y)
                rad = r * self.renderer.scale * (0.4 + age * 1.2)
                k = 1 - age / 0.8
                pygame.draw.circle(screen, (int(255 * k), int(140 * k), int(40 * k)), (sx, sy), rad, 3)
        self.effects = keep

    def _draw_hud(self, screen) -> None:
        # Tool chips, top-left.
        x = 10
        for k, name in enumerate(TOOLS):
            on = name == self.tool
            label = f"{k + 1} {name}"
            w = render.get_font(12, True).size(label)[0] + 16
            chip = pygame.Rect(x, 10, w, 22)
            panel(screen, chip, alpha=235 if on else 170)
            if on:
                pygame.draw.rect(screen, ACCENT, chip, 1, border_radius=8)
            text(screen, label, chip.center, ACCENT if on else DIM, 12, bold=on, anchor="center")
            x = chip.right + 6
        flash, t0 = self.flash
        if flash and time.time() - t0 < 2.5:
            text(screen, flash, (x + 8, 13), ACCENT, 13)

        # Status, top-right: REC blinks, PAUSED is steady.
        right = self.size - 12
        if self.recorder is not None:
            if int(time.time() * 2) % 2 == 0:
                pygame.draw.circle(screen, (235, 50, 50), (right - 8, 21), 7)
            right = text(screen, f"REC {self.recorder.frames}", (right - 22, 21), (235, 80, 80), 13, True,
                         anchor="midright").x - 14
        if self.paused:
            text(screen, "PAUSED", (right, 21), TEXT, 13, True, anchor="midright")

        # Key help strip along the bottom.
        strip = pygame.Surface((self.size, 36), pygame.SRCALPHA)
        strip.fill((9, 10, 16, 210))
        screen.blit(strip, (0, self.size - 36))
        for k, line in enumerate(HELP):
            text(screen, line, (self.size // 2, self.size - 27 + k * 15), DIM, 11, anchor="center")

    def _draw_side(self, screen, rect: pygame.Rect) -> None:
        screen.fill(BG, rect)
        pygame.draw.line(screen, render.PANEL_EDGE, rect.topleft, rect.bottomleft)
        w, s = self.world, self.stats
        pad = 8
        x, width = rect.x + pad, rect.w - 2 * pad

        # Header: title, counters, climate.
        head = pygame.Rect(x, pad, width, 92)
        panel(screen, head)
        text(screen, "primordial", (head.x + 10, head.y + 7), ACCENT, 16, bold=True)
        text(screen, f"step {s['step']:,}   x{self.steps_per_frame}", (head.right - 10, head.y + 9), TEXT, 12,
             anchor="topright")
        text(screen, f"pop {s['population']}   food {s.get('food', 0)}   lineages {s.get('lineages', 0)}",
             (head.x + 10, head.y + 31), TEXT, 12)
        text(screen, f"gen {s.get('mean_generation', 0):.1f} / {s.get('max_generation', 0)}   "
                     f"bonded {s.get('bonded_fraction', 0):.0%}   prey {s.get('predation_share', 0):.0%}",
             (head.x + 10, head.y + 48), DIM, 12)
        season = w.season()
        text(screen, "climate", (head.x + 10, head.y + 68), DIM, 11)
        bar = pygame.Rect(head.x + 66, head.y + 72, 120, 6)
        pygame.draw.rect(screen, (34, 38, 52), bar, border_radius=3)
        pygame.draw.line(screen, (70, 75, 95), (bar.centerx, bar.y - 2), (bar.centerx, bar.bottom + 1))
        sx = bar.centerx + season * bar.w / 2
        pygame.draw.circle(screen, (120, 210, 120) if season >= 0 else (210, 150, 90), (sx, bar.centery), 5)
        text(screen, f"food x{w.food_multiplier:.2f}  [ ]", (head.right - 10, head.y + 67), ACCENT, 12,
             anchor="topright")

        # Species legend sized to its rows.
        rows = max(1, len(w.species_counts()))
        legend = pygame.Rect(x, head.bottom + 6, width, 32 + 17 * rows)
        render.draw_legend(screen, legend, w, self.renderer.color_mode == "species")
        graph = pygame.Rect(x, legend.bottom + 6, width, 92)
        self.graph.draw(screen, graph)

        bottom = pygame.Rect(x, graph.bottom + 6, width, rect.bottom - graph.bottom - 6 - pad)
        if bottom.h < 80:        # window too short for the bottom panel
            return
        if self.show_tree:
            self.tree.draw(screen, bottom, w, self.selected_uid if self.selected is not None else None)
        elif self.selected is not None:
            color = self.renderer.colors(w, torch.tensor([self.selected]))[0]
            render.draw_inspector(screen, bottom, w, self.selected, color)
        else:
            self._draw_idle(screen, bottom)

    def _draw_idle(self, screen, rect: pygame.Rect) -> None:
        """Inspector placeholder: hints plus a key to the marks on the world view."""
        panel(screen, rect, title="inspector")
        text(screen, f"colours: {self.renderer.color_mode}  (C to cycle)", (rect.x + 10, rect.y + 30), TEXT, 12)
        hints = ["Click a particle (tool 1) to see its", "body, diet, lineage and live brain.", "",
                 "F  family tree of everything alive", "G  sliders to bend the rules",
                 "2  drop food     3  meteor", "R  record the window to recordings/"]
        for k, line in enumerate(hints):
            text(screen, line, (rect.x + 10, rect.y + 56 + k * 17), DIM, 12)
        _draw_mark_key(screen, pygame.Rect(rect.x + 10, rect.y + 190, rect.w - 20, 60))


def _draw_mark_key(screen, rect: pygame.Rect) -> None:
    """What the marks on the world view mean."""
    items = [("ring", render.AGGRO, "attacking"), ("glow", (120, 200, 255), "signalling"),
             ("line", (160, 160, 200), "bonded"), ("dot", render.FOOD, "food"),
             ("circle", render.PATCH, "fertile patch"), ("cloud", (40, 200, 180), "scent A / B")]
    for k, (kind, col, label) in enumerate(items):
        cx = rect.x + 8 + (k % 2) * (rect.w // 2)
        cy = rect.y + 8 + (k // 2) * 20
        if kind == "ring":
            pygame.draw.circle(screen, (200, 200, 210), (cx, cy), 4)
            pygame.draw.circle(screen, col, (cx, cy), 7, 1)
        elif kind == "glow":
            for rr, a in ((9, 40), (6, 90), (4, 200)):
                pygame.draw.circle(screen, [c * a // 255 for c in col], (cx, cy), rr)
        elif kind == "line":
            pygame.draw.line(screen, col, (cx - 7, cy + 3), (cx + 7, cy - 3), 2)
        elif kind == "dot":
            pygame.draw.circle(screen, col, (cx, cy), 2)
        elif kind == "circle":
            pygame.draw.circle(screen, col, (cx, cy), 7, 1)
        else:
            pygame.draw.circle(screen, (20, 70, 70), (cx - 2, cy), 7)
            pygame.draw.circle(screen, (70, 25, 70), (cx + 3, cy), 6)
        text(screen, label, (cx + 16, cy), DIM, 12, anchor="midleft")


def _pick(world: World, x: float, y: float, radius: float) -> int | None:
    """Nearest living particle to (x, y) within `radius` (torus-aware), or None."""
    if not bool(world.alive.any()):
        return None
    delta = world._wrap_delta(world.pos - torch.tensor([x, y], device=world.device))
    dist = delta.norm(dim=1)
    dist[~world.alive] = float("inf")
    d, i = dist.min(0)
    return int(i) if d < radius else None


def run_viewer(world: World, size: int = 820, steps_per_frame: int = 2, save_path: str = "checkpoint.pt") -> None:
    pygame.init()
    viewer = Viewer(world, size, steps_per_frame, save_path)
    screen = pygame.display.set_mode(viewer.window_size)
    pygame.display.set_caption("primordial")
    clock = pygame.time.Clock()
    try:
        while viewer.running:
            for event in pygame.event.get():
                viewer.handle(event)
            viewer.update()
            viewer.draw(screen)
            pygame.display.flip()
            clock.tick(60)
    finally:
        if viewer.recorder is not None:
            print(f"saved {viewer.recorder.close()}")
        pygame.quit()
