"""Drawing a World onto any pygame Surface: shared by the live viewer and headless timelapses.

`Renderer` draws the world view (particles, food, patches, bonds, optional overlays).
The panel helpers (`Graph`, `draw_species`, `draw_legend`, `draw_inspector`, `TreePanel`)
draw side-panel sections into a given rect, so the viewer only decides layout.
Styling follows a sober light theme (with a dark variant) meant to read like an analysis tool.
"""

import math
import os
import time
from collections import deque
from dataclasses import dataclass

import pygame
import pygame.gfxdraw
import torch

from .genome import N_INPUTS, OUTPUT_NAMES, SENSE_NAMES

RGB = tuple[int, int, int]


def hex_rgb(h: str) -> RGB:
    return tuple(int(h[k:k + 2], 16) for k in (1, 3, 5))


@dataclass(frozen=True)
class Theme:
    name: str
    canvas: RGB      # world view background
    panel: RGB       # side panel and cards
    border: RGB      # 1px rules and card borders
    text: RGB
    muted: RGB
    accent: RGB      # also the colour of positive weights
    negative: RGB    # attacking outline, negative weights
    grid: RGB        # chart gridlines, bar tracks
    bond: RGB
    food: RGB
    patch: RGB
    heat: RGB        # pheromone field hue


LIGHT = Theme("light", hex_rgb("#fafaf9"), hex_rgb("#ffffff"), hex_rgb("#e2e2de"), hex_rgb("#1f2421"),
              hex_rgb("#5f6763"), hex_rgb("#2f5d8a"), hex_rgb("#c0392b"), hex_rgb("#efefec"),
              hex_rgb("#9aa09c"), hex_rgb("#7d9a5a"), hex_rgb("#c9ccc8"), hex_rgb("#2f5d8a"))
DARK = Theme("dark", hex_rgb("#16191b"), hex_rgb("#1e2226"), hex_rgb("#2c3237"), hex_rgb("#e6e8e6"),
             hex_rgb("#9aa29e"), hex_rgb("#6f9cc8"), hex_rgb("#d9604f"), hex_rgb("#262b30"),
             hex_rgb("#5d6561"), hex_rgb("#7d9a5a"), hex_rgb("#3a4146"), hex_rgb("#5b8fc4"))
THEMES = {"light": LIGHT, "dark": DARK}
_theme = LIGHT


def theme() -> Theme:
    return _theme


def set_theme(name: str) -> Theme:
    global _theme
    _theme = THEMES[name]
    return _theme


COLOR_MODES = ("lineage", "type", "species")
COLOR_MODE_LABELS = {"lineage": "lineage", "type": "physical type", "species": "behavioural species"}
TABLEAU10 = [hex_rgb(h) for h in ("#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                                  "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac")]
DARK2 = [hex_rgb(h) for h in ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
                              "#e6ab02", "#a6761d", "#666666")]
UNCLASSIFIED = (160, 164, 161)


def lineage_rgb(color: torch.Tensor) -> torch.Tensor:
    """Genome colour (n, 3) in [0, 1] -> muted display colour (n, 3) int.

    Keeps the hue (so lineages stay distinguishable) but pins saturation to 0.45-0.6 and value
    to 0.55-0.75, which reads on both a white and a dark canvas without going neon.
    """
    r, g, b = color.unbind(1)
    mx, mn = color.max(1).values, color.min(1).values
    d = mx - mn
    safe = d.clamp_min(1e-6)
    h = torch.where(mx == r, ((g - b) / safe) % 6, torch.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4))
    h = torch.where(d > 1e-6, h, torch.zeros_like(h)) / 6
    # Near-grey genomes have no meaningful hue: fade their saturation out rather than invent one.
    s = (0.45 + 0.15 * (d / mx.clamp_min(1e-6))) * (d / 0.12).clamp(max=1)
    v = 0.55 + 0.2 * mx
    k = (torch.stack([torch.full_like(h, 5.0), torch.full_like(h, 3.0), torch.ones_like(h)], 1) + h[:, None] * 6) % 6
    rgb = v[:, None] - (v * s)[:, None] * torch.clamp(torch.minimum(k, 4 - k), 0, 1)
    return (rgb * 255).round().int()


def muted(color) -> RGB:
    """One genome colour tuple -> display colour."""
    return tuple(lineage_rgb(torch.tensor([color], dtype=torch.float32))[0].tolist())


def species_rgb(sid: int) -> RGB:
    return TABLEAU10[sid % len(TABLEAU10)] if sid >= 0 else UNCLASSIFIED


def type_rgb(t: int) -> RGB:
    return DARK2[t % len(DARK2)]


def shade(col, k: float) -> RGB:
    """Scale brightness: k < 1 darker (outlines), k > 1 lighter."""
    return tuple(max(0, min(255, int(c * k))) for c in col)


def mix(a, b, t: float) -> RGB:
    """Linear blend from colour a (t=0) to b (t=1)."""
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


# ------------------------------------------------------------------------ type

_FONT_FILES = {
    "regular": ("segoeui.ttf", "arial.ttf"),
    "bold": ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"),
    "mono": ("consola.ttf",),
}
_FONTS: dict = {}


def _font_path(kind: str) -> str | None:
    # SysFont("segoeui") resolves to Segoe UI *Light* on Windows, so prefer exact files.
    fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    for name in _FONT_FILES[kind]:
        path = os.path.join(fonts_dir, name)
        if os.path.exists(path):
            return path
    fallback = {"regular": "segoeui,arial,dejavusans", "bold": "segoeui,arial,dejavusans",
                "mono": "consolas,dejavusansmono,couriernew"}[kind]
    return pygame.font.match_font(fallback, bold=kind == "bold")


def get_font(size: int, bold: bool = False, mono: bool = False) -> pygame.font.Font:
    if not pygame.font.get_init():
        pygame.font.init()
    kind = "mono" if mono else "bold" if bold else "regular"
    key = (size, kind)
    if key not in _FONTS:
        _FONTS[key] = pygame.font.Font(_font_path(kind), size)
    return _FONTS[key]


def text(surface, s: str, pos, color=None, size: int = 12, bold: bool = False, anchor: str = "topleft",
         mono: bool = False) -> pygame.Rect:
    img = get_font(size, bold, mono).render(s, True, color or theme().text)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect


def card(surface, rect: pygame.Rect, alpha: int = 255) -> None:
    """Flat panel with a 1px border (used for overlays on the world view)."""
    if rect.w <= 0 or rect.h <= 0:
        return
    th = theme()
    if alpha >= 255:
        pygame.draw.rect(surface, th.panel, rect, border_radius=4)
    else:
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*th.panel, alpha), s.get_rect(), border_radius=4)
        surface.blit(s, rect.topleft)
    pygame.draw.rect(surface, th.border, rect, 1, border_radius=4)


def section(surface, rect: pygame.Rect, title: str, note: str = "") -> int:
    """Section heading (sentence case, semibold) with an optional muted note; returns the y below it."""
    text(surface, title, (rect.x, rect.y), theme().text, 13, bold=True)
    if note:
        text(surface, note, (rect.right, rect.y + 2), theme().muted, 11, anchor="topright")
    return rect.y + 22


def rule(surface, x0: int, x1: int, y: int) -> None:
    pygame.draw.line(surface, theme().border, (x0, y), (x1, y))


def nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """Round tick values (1, 2, 2.5, 5 x 10^k) from at or below `lo` to the first at or above `hi`."""
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks = [round(start, 10)]
    while ticks[-1] < hi - step * 1e-6:
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def fmt_tick(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v / 1000:g}k"
    return f"{v:g}"


# ------------------------------------------------------------------ world view


class Renderer:
    """Draws the world view: flat particles, food, patches, bonds; optional trails/field/signal overlays."""

    def __init__(self, px: int, world_size: float):
        self.px = px
        self.scale = px / world_size
        self.ui = px / 820          # sizes were tuned at 820 px; scale them for other sizes
        self.color_mode = "lineage"
        self.trails = False
        self.show_pheromone = False
        self.show_signal = False
        self.show_patches = True
        self.show_bonds = True
        self.trail: pygame.Surface | None = None
        self._trail_theme = ""

    def cycle_colors(self) -> str:
        self.color_mode = COLOR_MODES[(COLOR_MODES.index(self.color_mode) + 1) % len(COLOR_MODES)]
        return self.color_mode

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, y * self.scale

    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return sx / self.scale, sy / self.scale

    def colors(self, world, idx: torch.Tensor) -> list:
        """Display colour per particle in `idx`, for the current colour mode."""
        if self.color_mode == "type":
            return [type_rgb(t) for t in world.genome.ptype[idx].tolist()]
        if self.color_mode == "species":
            return [species_rgb(s) for s in world.species_id[idx].tolist()]
        return [tuple(c) for c in lineage_rgb(world.genome.color[idx]).tolist()]

    def radii(self, world, idx: torch.Tensor) -> torch.Tensor:
        """Screen radius from body size, so the legend can reuse the same mapping."""
        return (1.5 + 2.6 * world.body(idx)[0]).mul(self.ui).clamp_min(1.5)

    def draw(self, surface: pygame.Surface, world, selected: int | None = None, offset=(0, 0)) -> None:
        """Draw one frame of the world view onto `surface` at `offset` (px x px)."""
        th, s, ui = theme(), self.scale, self.ui
        idx = world.alive.nonzero().squeeze(1)
        # One tensor -> list conversion per quantity per frame; the Python loops below are the cost.
        pos = (world.pos[idx] * s).round().int().tolist()
        radius = self.radii(world, idx).round().int().tolist()
        aggressive = (world.aggression[idx] > 0.3).tolist()
        cols = self.colors(world, idx)

        frame = self._base(world, pos, radius, cols)
        if self.show_patches:
            self._draw_patches(frame, world)
        fr = max(1, round(1.4 * ui))
        for x, y in (world.food[world.food_alive] * s).tolist():
            pygame.draw.circle(frame, th.food, (x, y), fr)
        if self.show_bonds:
            self._draw_bonds(frame, world, idx)
        if self.show_signal:
            self._draw_signal(frame, world, idx, pos, radius, cols)

        fill_circle, ring = pygame.gfxdraw.filled_circle, pygame.gfxdraw.aacircle
        red = th.negative
        for (x, y), r, c, agg in zip(pos, radius, cols, aggressive):
            fill_circle(frame, x, y, r, c)
            if agg:
                ring(frame, x, y, r, red)
                ring(frame, x, y, r + 1, red)
            else:
                ring(frame, x, y, r, shade(c, 0.65))

        if selected is not None and bool(world.alive[selected]):
            self._draw_selected(frame, world, selected)
        surface.blit(frame, offset)

    def _base(self, world, pos, radius, cols) -> pygame.Surface:
        """Canvas (or the fading trail layer) plus the optional pheromone heatmap."""
        th = theme()
        if self.trails:
            if self.trail is None or self._trail_theme != th.name:
                self.trail = pygame.Surface((self.px, self.px))
                self.trail.fill(th.canvas)
                self._trail_theme = th.name
            fade = pygame.Surface((self.px, self.px), pygame.SRCALPHA)
            fade.fill((*th.canvas, 30))
            self.trail.blit(fade, (0, 0))
            for (x, y), r, c in zip(pos, radius, cols):
                pygame.draw.circle(self.trail, mix(th.canvas, c, 0.35), (x, y), max(1, r - 1))
            frame = self.trail.copy()
        else:
            self.trail = None
            frame = pygame.Surface((self.px, self.px))
            frame.fill(th.canvas)
        if self.show_pheromone:
            heat, flags = self._pheromone_surface(world)
            frame.blit(heat, (0, 0), special_flags=flags)
        return frame

    def _draw_patches(self, frame, world) -> None:
        """Fertile patches as thin dashed rings."""
        pr = world.cfg.patch_radius * self.scale * 2
        col = theme().patch
        n = 36
        for cx, cy in (world.patch_pos * self.scale).tolist():
            pts = [(cx + pr * math.cos(2 * math.pi * k / n), cy + pr * math.sin(2 * math.pi * k / n))
                   for k in range(n + 1)]
            for k in range(0, n, 2):
                pygame.draw.aaline(frame, col, pts[k], pts[k + 1])

    def _draw_bonds(self, frame, world, idx) -> None:
        bonds = world.bonds[idx]
        has = (bonds >= 0).any(1)
        if not bool(has.any()):
            return
        slots = idx[has].tolist()
        rows = bonds[has].tolist()
        pos = world.pos.mul(self.scale).tolist()
        half = self.px / 2
        col = theme().bond
        for a, partners in zip(slots, rows):
            ax, ay = pos[a]
            for b in partners:
                if b <= a:          # each pair once (and skips -1)
                    continue
                bx, by = pos[b]
                if abs(bx - ax) > half or abs(by - ay) > half:
                    continue        # pair straddles the torus seam; a screen-wide line would lie
                pygame.draw.aaline(frame, col, (ax, ay), (bx, by))

    def _draw_signal(self, frame, world, idx, pos, radius, cols) -> None:
        """Signal display: a thin outer ring whose contrast follows |signal|."""
        canvas = theme().canvas
        for (x, y), r, c, sig in zip(pos, radius, cols, world.signal[idx].abs().tolist()):
            if sig > 0.15:
                pygame.gfxdraw.aacircle(frame, x, y, r + 3, mix(canvas, c, min(1.0, sig)))

    def _draw_selected(self, frame, world, i: int) -> None:
        th = theme()
        x, y = (world.pos[i] * self.scale).tolist()
        r = float(self.radii(world, torch.tensor([i]))[0])
        sense = world.cfg.sense_radius * float(world.body([i])[2][0]) * self.scale
        n = 48
        for k in range(0, n, 2):     # dashed perception radius
            a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
            pygame.draw.aaline(frame, th.muted, (x + sense * math.cos(a0), y + sense * math.sin(a0)),
                               (x + sense * math.cos(a1), y + sense * math.sin(a1)))
        for b in world.bonds[i].tolist():
            if b >= 0:
                bx, by = (world.pos[b] * self.scale).tolist()
                if abs(bx - x) < self.px / 2 and abs(by - y) < self.px / 2:
                    pygame.draw.line(frame, th.text, (x, y), (bx, by), 2)
        pygame.draw.circle(frame, th.accent, (x, y), r + 5, 2)

    def _pheromone_surface(self, world) -> tuple[pygame.Surface, int]:
        """Both scent channels summed into one subtle single-hue heatmap, smoothly upscaled.

        Light theme multiplies a tint into the canvas; dark theme adds a glow-free tint.
        """
        th = theme()
        ph = world.pheromone.detach().float().cpu().sum(0)
        # Saturating, and never louder than the particles (dark backgrounds need less to show).
        level = (1 - torch.exp(-ph * 4.0)) * (0.45 if th.name == "light" else 0.28)
        hue = torch.tensor(th.heat, dtype=torch.float32) / 255
        if th.name == "light":
            rgb = 1 - level[:, :, None] * (1 - hue)
            flags = pygame.BLEND_MULT
        else:
            rgb = level[:, :, None] * hue
            flags = pygame.BLEND_ADD
        arr = (rgb.clamp(0, 1) * 255).to(torch.uint8).numpy()   # [x, y, rgb] matches surfarray layout
        small = pygame.surfarray.make_surface(arr)
        return pygame.transform.smoothscale(small, (self.px, self.px)), flags


# ---------------------------------------------------------------------- panels


def _resolve(col) -> RGB:
    """A colour, or the name of a Theme field (so charts follow theme switches)."""
    return getattr(theme(), col) if isinstance(col, str) else col


class Graph:
    """Rolling line chart against simulation step, with round ticks and labelled axes."""

    def __init__(self, series: dict[str, tuple[str, object]], length: int = 300):
        self.series = series                          # key -> (label, colour or Theme field name)
        self.steps: deque = deque(maxlen=length)
        self.data = {k: deque(maxlen=length) for k in series}

    def push(self, stats: dict) -> None:
        self.steps.append(int(stats.get("step", len(self.steps))))
        for k in self.series:
            self.data[k].append(float(stats.get(k, 0)))

    def draw(self, surface, rect: pygame.Rect, title: str = "Population over time") -> None:
        th = theme()
        y = section(surface, rect, title)
        # Inline legend with current values, right-aligned on the heading row.
        rx = rect.right
        for k, (label, col) in reversed(list(self.series.items())):
            values = self.data[k]
            r = text(surface, f"{values[-1]:,.0f}" if values else "–", (rx, rect.y + 2), th.text, 11, anchor="topright")
            r = text(surface, label, (r.x - 4, rect.y + 2), th.muted, 11, anchor="topright")
            pygame.draw.line(surface, _resolve(col), (r.x - 16, rect.y + 10), (r.x - 4, rect.y + 10), 2)
            rx = r.x - 26
        plot = pygame.Rect(rect.x + 34, y + 14, rect.w - 40, rect.bottom - y - 14 - 18)
        if plot.h < 20:
            return
        hi = max((max(v) for v in self.data.values() if v), default=1.0)
        ticks = nice_ticks(0, max(hi, 1.0), 3)
        top = ticks[-1]
        for t in ticks:
            ty = plot.bottom - plot.h * t / top
            pygame.draw.line(surface, th.grid if t else th.border, (plot.x, ty), (plot.right, ty))
            text(surface, fmt_tick(t), (plot.x - 6, ty), th.muted, 10, anchor="midright")
        text(surface, "Count", (rect.x, y - 2), th.muted, 10)
        if len(self.steps) >= 2:
            s0, s1 = self.steps[0], self.steps[-1]
            span = max(1, s1 - s0)
            for t in nice_ticks(s0, s1, 3):
                if s0 <= t <= s1:
                    tx = plot.x + plot.w * (t - s0) / span
                    pygame.draw.line(surface, th.border, (tx, plot.bottom), (tx, plot.bottom + 3))
                    label = fmt_tick(t)
                    half = get_font(10).size(label)[0] / 2
                    tx = min(max(tx, plot.x + half), rect.right - half)    # keep end labels inside
                    text(surface, label, (tx, plot.bottom + 4), th.muted, 10, anchor="midtop")
            for k, (_, col) in self.series.items():
                pts = [(plot.x + plot.w * (s - s0) / span, plot.bottom - plot.h * v / top)
                       for s, v in zip(self.steps, self.data[k])]
                pygame.draw.aalines(surface, _resolve(col), False, pts)
        text(surface, "Step", (rect.x, plot.bottom + 4), th.muted, 10)


def species_height(world) -> int:
    """Height `draw_species` needs: heading, stacked bar, two-column list."""
    return 22 + 18 + 17 * math.ceil(max(1, world.cfg.species_k) / 3)


def draw_species(surface, rect: pygame.Rect, world, active: bool) -> None:
    """Behavioural species: one stacked bar of shares, then a three-column count list."""
    th = theme()
    counts = world.species_counts()
    y = section(surface, rect, "Behavioural species", "colouring particles" if active else "")
    if not counts:
        text(surface, "Classifying… (every %d steps)" % world.cfg.species_interval, (rect.x, y), th.muted, 12)
        return
    names = world.species.names
    total = max(1, sum(counts.values()))
    bar = pygame.Rect(rect.x, y + 2, rect.w, 8)
    x = bar.x
    items = [(sid, name, counts[name]) for sid, name in enumerate(names) if counts.get(name, 0)]
    for k, (sid, _, c) in enumerate(items):
        w = bar.w * c / total if k < len(items) - 1 else bar.right - x
        pygame.draw.rect(surface, species_rgb(sid), (round(x), bar.y, math.ceil(w), bar.h))
        x += w
    y = bar.bottom + 8
    gap = 14
    col_w = (rect.w - 2 * gap) / 3
    for k, (sid, name, c) in enumerate(items):
        cx, cy = rect.x + (k % 3) * (col_w + gap), y + (k // 3) * 17
        pygame.draw.rect(surface, species_rgb(sid), (cx, cy + 4, 9, 9), border_radius=2)
        text(surface, name.capitalize(), (cx + 14, cy), th.text, 12)
        text(surface, f"{c}", (cx + col_w, cy), th.muted, 12, anchor="topright")


def draw_legend(surface, rect: pygame.Rect, color_mode: str) -> None:
    """What the marks on the world view mean."""
    th = theme()
    y = section(surface, rect, "Legend")
    col_w = rect.w // 2
    sample = (110, 140, 175)
    for k, kind in enumerate(("size", "colour", "attack", "bond", "food", "patch")):
        x, yy = rect.x + (k % 2) * col_w + 8, y + (k // 2) * 20 + 8
        label_x = x + 22
        if kind == "size":
            for dx, r in ((-2, 3), (8, 6)):
                pygame.gfxdraw.filled_circle(surface, x + dx, yy, r, sample)
                pygame.gfxdraw.aacircle(surface, x + dx, yy, r, shade(sample, 0.65))
            label = "Size = body size"
        elif kind == "attack":
            pygame.gfxdraw.filled_circle(surface, x, yy, 5, sample)
            pygame.gfxdraw.aacircle(surface, x, yy, 5, th.negative)
            pygame.gfxdraw.aacircle(surface, x, yy, 6, th.negative)
            label = "Red outline = attacking"
        elif kind == "bond":
            pygame.draw.aaline(surface, th.bond, (x - 6, yy + 4), (x + 6, yy - 4))
            label = "Line = bond"
        elif kind == "food":
            pygame.draw.circle(surface, th.food, (x, yy), 2)
            label = "Dot = food"
        elif kind == "colour":
            palette = {"lineage": [muted(c) for c in ((1, .3, .2), (.2, .8, .4), (.3, .4, 1))],
                       "type": DARK2[:3], "species": TABLEAU10[:3]}[color_mode]
            for j, c in enumerate(palette):
                pygame.gfxdraw.filled_circle(surface, x - 3 + j * 7, yy, 3, c)
            label = "Colour = " + COLOR_MODE_LABELS[color_mode]
        else:
            for j in range(0, 12, 2):
                a0, a1 = 2 * math.pi * j / 12, 2 * math.pi * (j + 1) / 12
                pygame.draw.aaline(surface, th.muted, (x + 6 * math.cos(a0), yy + 6 * math.sin(a0)),
                                   (x + 6 * math.cos(a1), yy + 6 * math.sin(a1)))
            label = "Ring = fertile patch"
        text(surface, label, (label_x, yy), th.muted, 11, anchor="midleft")


def _node_fill(a: float) -> RGB:
    """Activation -> node fill: blue for positive, red for negative, panel colour when silent."""
    th = theme()
    return mix(th.panel, th.accent if a >= 0 else th.negative, min(1.0, abs(a)))


def draw_brain(surface, rect: pygame.Rect, world, i: int, max_edges: int = 60) -> None:
    """Layered node-link diagram of particle i's network, nodes shaded by its last activations."""
    th = theme()
    g = world.genome
    w1 = g.w1[i]                                    # (N_INPUTS, hidden)
    w2 = g.w2[i] + world.plastic[i]                 # effective, including lifetime learning
    inputs = world.last_inputs[i].tolist()
    hidden = world.last_hidden[i].tolist()
    outputs = world.last_outputs[i].tolist()
    n_h = len(hidden)

    top, bottom = rect.y + 18, rect.bottom - 4
    xs = (rect.x + 82, rect.x + rect.w * 0.5, rect.right - 104)
    for x, label in zip(xs, (f"Inputs ({N_INPUTS})", f"Hidden ({n_h})", f"Outputs ({len(outputs)})")):
        text(surface, label, (x, rect.y), th.muted, 10, anchor="midtop")

    def column(n, pad=0.0):
        span = bottom - top
        return [top + span * (pad + (1 - 2 * pad) * (k + 0.5) / n) for k in range(n)]

    y_in, y_h, y_out = column(N_INPUTS), column(n_h, 0.04), column(len(outputs), 0.16)

    # Strongest edges per layer (60/40 split) so neither layer vanishes from the picture.
    k1 = round(max_edges * 0.6)
    v1, f1 = w1.abs().flatten().topk(min(k1, w1.numel()))
    v2, f2 = w2.abs().flatten().topk(min(max_edges - k1, w2.numel()))
    edges = []
    for wm, flat, mag, (xa, ya), (xb, yb) in ((w1, f1, v1, (xs[0], y_in), (xs[1], y_h)),
                                              (w2, f2, v2, (xs[1], y_h), (xs[2], y_out))):
        top_mag = max(float(mag[0]), 1e-6)
        n_dst = wm.shape[1]
        for f, m, w in zip(flat.tolist(), mag.tolist(), wm.flatten()[flat].tolist()):
            a, b = divmod(f, n_dst)
            edges.append(((xa, ya[a]), (xb, yb[b]), w, m / top_mag))

    layer = pygame.Surface(rect.size, pygame.SRCALPHA)
    ox, oy = rect.topleft
    for (x0, y0), (x1, y1), w, rel in sorted(edges, key=lambda e: e[3]):
        col = th.accent if w >= 0 else th.negative
        pygame.draw.line(layer, (*col, int(45 + 175 * rel)), (x0 - ox, y0 - oy), (x1 - ox, y1 - oy),
                         1 + int(rel * 2.2))
    surface.blit(layer, rect.topleft)

    edge = th.muted

    def node(x, y, a, r):
        pygame.gfxdraw.filled_circle(surface, int(x), int(y), r, _node_fill(a))
        pygame.gfxdraw.aacircle(surface, int(x), int(y), r, edge)

    for k, (y, a) in enumerate(zip(y_in, inputs)):
        node(xs[0], y, a, 3)
        text(surface, SENSE_NAMES[k].capitalize(), (xs[0] - 7, y), th.text if abs(a) >= 0.05 else th.muted, 10,
             anchor="midright")
    for y, a in zip(y_h, hidden):
        node(xs[1], y, a, 4)
    for k, (y, a) in enumerate(zip(y_out, outputs)):
        node(xs[2], y, a, 5)
        text(surface, OUTPUT_NAMES[k].capitalize(), (xs[2] + 9, y), th.text, 10, anchor="midleft")
        text(surface, f"{a:+.2f}", (rect.right - 2, y), th.muted, 10, anchor="midright", mono=True)


def draw_inspector(surface, rect: pygame.Rect, world, i: int, color) -> None:
    """Selected particle: identity, body, diet, learning, energy, then its neural network."""
    th = theme()
    g, cfg = world.genome, world.cfg
    size, speed, sense = (float(t[0]) for t in world.body([i]))
    sigma = g.sigma[i].tolist()
    sid = int(world.species_id[i])
    species = world.species.names[sid] if 0 <= sid < len(world.species.names) else "unclassified"
    prey, food = float(world.prey_gained[i]), float(world.food_gained[i])
    bonds = int((world.bonds[i] >= 0).sum())
    eta = 10 ** float(g.log_eta[i]) * cfg.plasticity
    energy = float(world.energy[i])
    full = cfg.reproduce_threshold * size

    y = rect.y
    pygame.gfxdraw.filled_circle(surface, rect.x + 6, y + 9, 6, color)
    pygame.gfxdraw.aacircle(surface, rect.x + 6, y + 9, 6, shade(color, 0.65))
    text(surface, f"Particle {int(world.uid[i])}", (rect.x + 18, y), th.text, 14, bold=True)
    tag = text(surface, species.capitalize(), (rect.right, y + 2), th.text, 12, anchor="topright")
    pygame.draw.rect(surface, species_rgb(sid), (tag.x - 14, tag.y + 4, 9, 9), border_radius=2)
    y += 26

    share = prey / (prey + food) if prey + food > 0 else 0.0
    rows = [
        (("Generation", f"{int(world.generation[i])}"), ("Age", f"{int(world.age[i]):,}"),
         ("Physical type", f"{int(g.ptype[i])}")),
        (("Body size", f"{size:.2f}"), ("Speed", f"{speed:.2f}"), ("Sense", f"{sense:.2f}")),
        (("Bonds", f"{bonds} / {cfg.max_bonds}"), ("Bud prob.", f"{float(g.bud[i]):.2f}"),
         ("Learning rate", f"{eta:.4f}")),
        (("Food energy", f"{food:.1f}"), ("Prey energy", f"{prey:.1f}"), ("Predation", f"{share:.0%}")),
    ]
    gap = 14
    col_w = (rect.w - 2 * gap) / 3
    for row in rows:
        for k, (label, value) in enumerate(row):
            x0 = rect.x + k * (col_w + gap)
            text(surface, label, (x0, y), th.muted, 11)
            text(surface, value, (x0 + col_w, y), th.text, 11, anchor="topright")
        y += 17
    text(surface, "Mutation step σ (inputs / outputs / physics / body)", (rect.x, y), th.muted, 11)
    text(surface, " / ".join(f"{v:.2f}" for v in sigma), (rect.right, y), th.text, 11, anchor="topright")
    y += 19

    text(surface, "Energy", (rect.x, y), th.muted, 11)
    val = text(surface, f"{energy:.2f} of {full:.2f} to divide", (rect.right, y), th.text, 11, anchor="topright")
    bar = pygame.Rect(rect.x + 50, y + 6, val.x - rect.x - 62, 5)
    pygame.draw.rect(surface, th.grid, bar, border_radius=2)
    pygame.draw.rect(surface, th.accent, (bar.x, bar.y, max(2, bar.w * min(1, energy / full)), bar.h), border_radius=2)
    y += 24

    rule(surface, rect.x, rect.right, y - 4)
    y = section(surface, pygame.Rect(rect.x, y + 4, rect.w, 20), "Neural network",
                "60 strongest weights · blue +, red −")
    draw_brain(surface, pygame.Rect(rect.x, y, rect.w, rect.bottom - y), world, i)


class TreePanel:
    """Dendrogram of the living population's family tree; laid out at most once per `interval` s."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.last = -math.inf
        self.surface: pygame.Surface | None = None
        self.where: dict[int, tuple[float, float]] = {}   # uid -> (x, y) on screen
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.plot = pygame.Rect(0, 0, 0, 0)
        self.info = ""
        self._theme = ""

    def draw(self, surface, rect: pygame.Rect, world, selected_uid: int | None = None, force: bool = False) -> None:
        now = time.perf_counter()
        stale = self.surface is None or rect != self.rect or self._theme != theme().name
        if force or stale or now - self.last > self.interval:
            self._layout(world, rect)
            self.last = now
        surface.blit(self.surface, rect.topleft)
        if selected_uid is not None and selected_uid in self.where:
            self._highlight(surface, world, selected_uid)

    def _layout(self, world, rect: pygame.Rect) -> None:
        th = theme()
        self.rect, self._theme = rect.copy(), th.name
        surf = pygame.Surface(rect.size)
        surf.fill(th.panel)
        self.surface = surf
        self.where = {}
        alive_uids = world.uid[world.alive].tolist()
        nodes = world.lineage.living_tree(alive_uids)
        section(surf, pygame.Rect(0, 0, rect.w, 20), "Family tree", "click a branch to inspect")
        if not nodes:
            return
        plot = pygame.Rect(0, 40, rect.w - 6, rect.h - 40 - 34)
        self.plot = plot.move(rect.x, rect.y)
        now = world.step_count
        t0 = min(n["born"] for n in nodes)
        span = max(1, now - t0)

        children: dict[int, list] = {}
        by_uid = {}
        roots = []
        for n in nodes:
            by_uid[n["uid"]] = n
            (roots if n["parent"] < 0 else children.setdefault(n["parent"], [])).append(n["uid"])

        # Order leaves depth-first, big clades first, so related particles sit together.
        y_slot: dict[int, float] = {}
        order = []
        stack = sorted(roots, key=lambda u: by_uid[u]["leaves"])
        while stack:
            u = stack.pop()
            order.append(u)
            stack.extend(sorted(children.get(u, []), key=lambda c: by_uid[c]["leaves"]))
        slots = 0
        for u in order:
            if by_uid[u]["alive"] or u not in children:
                y_slot[u] = slots
                slots += 1
        for u in reversed(order):           # internal nodes sit at the middle of their children
            if u not in y_slot:
                ys = [y_slot[c] for c in children[u]]
                y_slot[u] = (min(ys) + max(ys)) / 2

        def px(t):
            return plot.x + plot.w * (t - t0) / span

        def py(v):
            return plot.y + plot.h * (v + 0.5) / max(slots, 1)

        x_now = px(now)
        for u in order:
            n = by_uid[u]
            x, y = px(n["born"]), py(y_slot[u])
            self.where[u] = (x + rect.x, y + rect.y)
            col = muted(n["color"])
            if n["parent"] >= 0:
                p = by_uid[n["parent"]]
                xp, yp = px(p["born"]), py(y_slot[p["uid"]])
                pygame.draw.line(surf, col, (xp, yp), (xp, y))
                pygame.draw.line(surf, col, (xp, y), (x, y))
            if n["alive"]:      # lifespan so far, lighter so the branching structure dominates
                pygame.draw.line(surf, mix(th.panel, col, 0.45), (x, y), (x_now, y))

        # Time axis with round ticks.
        axis_y = plot.bottom + 4
        pygame.draw.line(surf, th.border, (plot.x, axis_y), (plot.right, axis_y))
        for t in nice_ticks(t0, now, 4):
            if t0 <= t <= now:
                tx = px(t)
                pygame.draw.line(surf, th.border, (tx, axis_y), (tx, axis_y + 3))
                text(surf, fmt_tick(t), (tx, axis_y + 4), th.muted, 10, anchor="midtop")
        text(surf, "Birth step", (plot.centerx, axis_y + 17), th.muted, 10, anchor="midtop")
        self.info = f"{len(alive_uids)} living · {len(nodes)} nodes · {len(roots)} founding lines"
        text(surf, self.info, (0, 22), th.muted, 11)

    def _highlight(self, surface, world, uid: int) -> None:
        """Ancestry of the selected particle drawn over the cached tree."""
        th = theme()
        chain = world.lineage.ancestors(uid, depth=100_000)
        pts = [self.where[u] for u in chain if u in self.where]
        x, y = self.where[uid]
        pygame.draw.line(surface, th.text, (x, y), (self.plot.right, y), 2)
        for (xc, yc), (xp, yp) in zip(pts, pts[1:]):
            pygame.draw.line(surface, th.text, (xp, yp), (xp, yc), 2)
            pygame.draw.line(surface, th.text, (xp, yc), (xc, yc), 2)
        pygame.draw.circle(surface, th.accent, (self.plot.right, y), 4)
        if pts:
            pygame.draw.circle(surface, th.accent, pts[-1], 4)
