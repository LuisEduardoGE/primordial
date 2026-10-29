"""Real-time pygame window onto a World: world view on the left, analysis side panel on the right."""

import time
from dataclasses import dataclass

import pygame
import pygame.gfxdraw
import torch

from . import render
from .render import Graph, Renderer, TreePanel, card, rule, section, text, theme
from .world import World

SIDE = 400               # side panel width
PAD = 16                 # side panel gutter
TOOLS = ("inspect", "add food", "remove area")
SHORT_HELP = "Space pause · 1–3 interventions · F family tree · G parameters · D theme"


@dataclass
class Slider:
    key: str          # Config field edited live
    label: str
    lo: float
    hi: float
    fmt: str = "{:.3f}"

    def frac(self, cfg) -> float:
        return (getattr(cfg, self.key) - self.lo) / (self.hi - self.lo)

    def set_frac(self, cfg, f: float) -> None:
        setattr(cfg, self.key, self.lo + min(1.0, max(0.0, f)) * (self.hi - self.lo))


SLIDERS = [
    Slider("food_regrow_prob", "Food regrowth probability", 0.0, 0.02, "{:.4f}"),
    Slider("attack_rate", "Attack rate", 0.0, 0.1),
    Slider("mutation_rate", "Mutation rate", 0.0, 0.5),
    Slider("climate_amplitude", "Climate amplitude", 0.0, 1.0, "{:.2f}"),
    Slider("pheromone_decay", "Pheromone decay", 0.0, 0.05, "{:.4f}"),
    Slider("bond_strength", "Bond strength", 0.0, 120.0, "{:.1f}"),
    Slider("mate_prob", "Mating probability", 0.0, 1.0, "{:.2f}"),
    Slider("plasticity", "Plasticity", 0.0, 3.0, "{:.2f}"),
]


class SliderPanel:
    """Pygame-drawn sliders that live-edit world.cfg; drag a track to change a value."""

    ROW = 36

    def __init__(self, sliders: list[Slider]):
        self.sliders = sliders
        self.rect = pygame.Rect(0, 0, 320, 52 + self.ROW * len(sliders))
        self.active: Slider | None = None

    def track(self, k: int) -> pygame.Rect:
        return pygame.Rect(self.rect.x + 16, self.rect.y + 46 + k * self.ROW + 18, self.rect.w - 32, 4)

    def press(self, pos, cfg) -> bool:
        """Start a drag if `pos` hits a track (with a forgiving hit box). True if consumed."""
        if not self.rect.collidepoint(pos):
            return False
        for k, s in enumerate(self.sliders):
            if self.track(k).inflate(10, 22).collidepoint(pos):
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
        th = theme()
        card(surface, self.rect)
        section(surface, pygame.Rect(self.rect.x + 16, self.rect.y + 12, self.rect.w - 32, 20), "Parameters",
                "drag to adjust · live")
        for k, s in enumerate(self.sliders):
            t = self.track(k)
            f = min(1.0, max(0.0, s.frac(cfg)))
            hot = s is self.active
            text(surface, s.label, (t.x, t.y - 17), th.text, 12)
            text(surface, s.fmt.format(getattr(cfg, s.key)), (t.right, t.y - 17), th.accent if hot else th.text, 12,
                 anchor="topright")
            pygame.draw.rect(surface, th.grid, t.inflate(0, 0), border_radius=2)
            pygame.draw.rect(surface, th.accent, (t.x, t.y, max(2, t.w * f), t.h), border_radius=2)
            knob = (int(t.x + t.w * f), t.centery)
            pygame.gfxdraw.filled_circle(surface, *knob, 6, th.panel)
            pygame.gfxdraw.aacircle(surface, *knob, 6, th.accent)
            if hot:
                pygame.gfxdraw.aacircle(surface, *knob, 5, th.accent)


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
        self.graph = Graph({"population": ("Population", "accent"), "food": ("Food items", "food")})
        self.tree = TreePanel(interval=1.0)
        self.sliders = SliderPanel(SLIDERS)
        self.sliders.rect.bottomleft = (12, size - 12)
        self.tool = "inspect"
        self.paused = False
        self.show_hud = True
        self.show_tree = False
        self.show_sliders = False
        self.selected: int | None = None
        self.selected_uid = -1
        self.recorder = None
        self._rec_start = 0.0
        self.flash = ("", 0.0)
        self.effects: list[tuple[float, float, float, float]] = []   # removed areas: x, y, r, t0
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
        toggles = {pygame.K_t: "trails", pygame.K_o: "show_pheromone", pygame.K_v: "show_signal",
                   pygame.K_b: "show_bonds", pygame.K_p: "show_patches"}
        names = {"trails": "Trails", "show_pheromone": "Pheromone field", "show_signal": "Signal display",
                 "show_bonds": "Bonds", "show_patches": "Fertile patches"}
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
            self.say(f"Colour by {render.COLOR_MODE_LABELS[r.cycle_colors()]}")
        elif key in toggles:
            attr = toggles[key]
            setattr(r, attr, not getattr(r, attr))
            self.say(f"{names[attr]} {'on' if getattr(r, attr) else 'off'}")
        elif key == pygame.K_f:
            self.show_tree = not self.show_tree
        elif key == pygame.K_g:
            self.show_sliders = not self.show_sliders
        elif key == pygame.K_d:
            render.set_theme("dark" if theme().name == "light" else "light")
        elif key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            delta = 0.25 if key == pygame.K_RIGHTBRACKET else -0.25
            w.food_multiplier = round(min(3.0, max(0.0, w.food_multiplier + delta)), 2)
            self.say(f"Food growth ×{w.food_multiplier:.2f}")
        elif key == pygame.K_h:
            self.show_hud = not self.show_hud
        elif key == pygame.K_r:
            self.toggle_recording()
        elif key == pygame.K_s:
            w.save(self.save_path)
            self.say(f"Saved {self.save_path}")

    def click(self, pos, button: int) -> None:
        if self.show_sliders and self.show_hud and button == 1 and self.sliders.press(pos, self.world.cfg):
            return
        if pos[0] >= self.size:
            if button == 1 and self.show_tree:
                self.pick_from_tree(pos)
            return
        x, y = self.renderer.to_world(*pos)
        if button == 3:
            self.say(f"Added {self.world.drop_food(x, y)} food items")
        elif button == 1:
            self.use_tool(x, y)

    def use_tool(self, x: float, y: float) -> None:
        """Apply the current intervention at world coordinates (x, y)."""
        w = self.world
        if self.tool == "inspect":
            self.select(_pick(w, x, y, radius=20 / self.renderer.scale))
        elif self.tool == "add food":
            self.say(f"Added {w.drop_food(x, y)} food items")
        elif self.tool == "remove area":
            radius = 0.08
            self.say(f"Removed {w.meteor(x, y, radius)} particles")
            self.effects.append((x, y, radius, time.time()))

    def select(self, slot: int | None) -> None:
        self.selected = slot
        self.selected_uid = int(self.world.uid[slot]) if slot is not None else -1

    def pick_from_tree(self, pos) -> None:
        """Click a living branch of the family tree to select that particle."""
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
            self._rec_start = time.time()
            self.say(f"Recording to {self.recorder.path}")
        else:
            path = self.recorder.close()
            self.say(f"Saved {path} ({self.recorder.frames} frames)")
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
                self.say("Selected particle died")
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
        th, now, keep = theme(), time.time(), []
        for x, y, r, t0 in self.effects:
            age = now - t0
            if age < 0.8:
                keep.append((x, y, r, t0))
                sx, sy = self.renderer.to_screen(x, y)
                col = render.mix(th.canvas, th.negative, 1 - age / 0.8)
                pygame.draw.circle(screen, col, (sx, sy), r * self.renderer.scale, 2)
        self.effects = keep

    def _draw_hud(self, screen) -> None:
        th = theme()
        # Interventions: a segmented control, top-left.
        labels = [f"{k + 1}  {name.capitalize()}" for k, name in enumerate(TOOLS)]
        widths = [render.get_font(12).size(s)[0] + 20 for s in labels]
        head = render.get_font(12).size("Interventions")[0] + 20
        bar = pygame.Rect(12, 12, head + sum(widths), 28)
        card(screen, bar)
        text(screen, "Interventions", (bar.x + 10, bar.centery), th.muted, 12, anchor="midleft")
        x = bar.x + head
        for name, label, w in zip(TOOLS, labels, widths):
            seg = pygame.Rect(x, bar.y + 3, w - 3, bar.h - 6)
            if name == self.tool:
                pygame.draw.rect(screen, th.accent, seg, border_radius=3)
            text(screen, label, seg.center, th.panel if name == self.tool else th.text, 12, anchor="center")
            x += w

        flash, t0 = self.flash
        if flash and time.time() - t0 < 2.5:
            img = render.get_font(12).size(flash)
            toast = pygame.Rect(12, bar.bottom + 8, img[0] + 20, 24)
            card(screen, toast)
            text(screen, flash, (toast.x + 10, toast.centery), th.text, 12, anchor="midleft")

        # Status, top-right.
        right = self.size - 12
        status = []
        if self.recorder is not None:
            secs = int(time.time() - self._rec_start)
            status.append((f"Recording {secs // 60}:{secs % 60:02d}", th.negative))
        if self.paused:
            status.append(("Paused", th.text))
        for label, col in status:
            w = render.get_font(12).size(label)[0] + (34 if col == th.negative else 20)
            pill = pygame.Rect(right - w, 12, w, 28)
            card(screen, pill)
            if col == th.negative:
                pygame.gfxdraw.filled_circle(screen, pill.x + 14, pill.centery, 4, th.negative)
                pygame.gfxdraw.aacircle(screen, pill.x + 14, pill.centery, 4, th.negative)
            text(screen, label, (pill.right - 10, pill.centery), col, 12, anchor="midright")
            right = pill.x - 8

    def _draw_side(self, screen, rect: pygame.Rect) -> None:
        th, w, s = theme(), self.world, self.stats
        screen.fill(th.panel, rect)
        pygame.draw.line(screen, th.border, rect.topleft, rect.bottomleft)
        x, width = rect.x + PAD, rect.w - 2 * PAD
        x1 = x + width

        text(screen, "Primordial — evolutionary particle simulation", (x, 14), th.text, 14, bold=True)
        state = "paused" if self.paused else "running"
        text(screen, f"Step {s['step']:,} · {self.steps_per_frame} steps per frame · {state}", (x, 36), th.muted, 11)
        rule(screen, x, x1, 60)

        y = section(screen, pygame.Rect(x, 70, width, 20), "Summary")
        season = w.season()
        grid = [
            (("Population", f"{s['population']:,}"), ("Food items", f"{s.get('food', 0):,}"),
             ("Effective lineages", f"{float(s.get('lineages', 0)):.1f}")),
            (("Mean generation", f"{s.get('mean_generation', 0):.1f}"), ("Max generation", f"{s.get('max_generation', 0)}"),
             ("Bonded", f"{s.get('bonded_fraction', 0):.0%}")),
            (("Predation share", f"{s.get('predation_share', 0):.0%}"), ("Season", f"{season:+.2f}"),
             ("Food growth", f"×{w.food_multiplier:.2f}")),
        ]
        gap = 14
        col_w = (width - 2 * gap) / 3
        for row in grid:
            for k, (label, value) in enumerate(row):
                cx = x + k * (col_w + gap)
                text(screen, label, (cx, y), th.muted, 11)
                text(screen, value, (cx + col_w, y), th.text, 11, anchor="topright")
            y += 16
        y += 6
        rule(screen, x, x1, y)

        sp = pygame.Rect(x, y + 10, width, render.species_height(w))
        render.draw_species(screen, sp, w, self.renderer.color_mode == "species")
        rule(screen, x, x1, sp.bottom + 6)
        chart = pygame.Rect(x, sp.bottom + 16, width, 108)
        self.graph.draw(screen, chart)
        rule(screen, x, x1, chart.bottom + 6)

        area = pygame.Rect(x, chart.bottom + 16, width, rect.bottom - chart.bottom - 16 - 30)
        if area.h < 80:          # window too short for the detail area
            return
        if self.show_tree or self.selected is not None:
            if self.show_tree:
                self.tree.draw(screen, area, w, self.selected_uid if self.selected is not None else None)
            else:
                color = self.renderer.colors(w, torch.tensor([self.selected]))[0]
                render.draw_inspector(screen, area, w, self.selected, color)
            rule(screen, x, x1, rect.bottom - 26)
            text(screen, SHORT_HELP, (x, rect.bottom - 20), th.muted, 11)
        else:
            self._draw_idle(screen, area.inflate(0, 30).move(0, 15))

    def _draw_idle(self, screen, rect: pygame.Rect) -> None:
        """Nothing selected: how to inspect, the legend, and the keyboard reference."""
        th = theme()
        y = section(screen, rect, "Particle inspector")
        for line in ("Choose Inspect (1) and click a particle to see its body,",
                     "diet, ancestry and neural network."):
            text(screen, line, (rect.x, y), th.muted, 12)
            y += 17
        rule(screen, rect.x, rect.right, y + 8)
        render.draw_legend(screen, pygame.Rect(rect.x, y + 18, rect.w, 82), self.renderer.color_mode)
        y += 18 + 86
        rule(screen, rect.x, rect.right, y)
        y = section(screen, pygame.Rect(rect.x, y + 10, rect.w, 20), "Keyboard")
        half = rect.w // 2
        for k, (keys, action) in enumerate(KEYS):
            cx, cy = rect.x + (k % 2) * half, y + (k // 2) * 16
            if cy > rect.bottom - 14:
                break
            text(screen, keys, (cx, cy), th.text, 11)
            text(screen, action, (cx + 62, cy), th.muted, 11)


KEYS = [("Space", "Pause"), ("C", "Colour mode"), ("↑ / ↓", "Speed"), ("T", "Trails"),
        ("1 – 3", "Interventions"), ("O", "Pheromone field"), ("Right-click", "Add food"),
        ("V", "Signal display"), ("[ / ]", "Food growth"), ("B", "Bonds"), ("F", "Family tree"),
        ("P", "Fertile patches"), ("G", "Parameters"), ("D", "Dark / light theme"), ("R", "Record video"),
        ("H", "Hide overlays"), ("S", "Save checkpoint"), ("Q", "Quit")]


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
    pygame.display.set_caption("Primordial — evolutionary particle simulation")
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
