"""Drawing a World onto any pygame Surface: shared by the live viewer and headless timelapses.

`Renderer` draws the world view (particles, trails, food, patches, bonds, pheromones).
The panel helpers (`draw_brain`, `TreePanel`, `draw_legend`, `Graph`) draw the side panel
pieces into a given rect, so the viewer only decides layout.
"""

import math
import time
from collections import deque

import pygame
import torch

from .genome import N_INPUTS, OUTPUT_NAMES, SENSE_NAMES

BG = (9, 10, 16)
PANEL = (17, 19, 28)
PANEL_EDGE = (40, 44, 60)
TEXT = (214, 218, 228)
DIM = (120, 126, 145)
ACCENT = (255, 206, 110)
POS_EDGE = (255, 150, 70)    # positive weight / activation
NEG_EDGE = (80, 160, 255)    # negative weight / activation
FOOD = (170, 200, 90)
PATCH = (70, 90, 40)
AGGRO = (255, 60, 60)

COLOR_MODES = ("lineage", "type", "species")
TYPE_PALETTE = [(235, 85, 85), (80, 170, 245), (245, 205, 70), (175, 105, 235),
                (90, 225, 145), (245, 135, 45), (225, 225, 225), (125, 125, 255)]
SPECIES_PALETTE = [(255, 99, 132), (54, 200, 235), (255, 205, 86), (120, 220, 120), (190, 130, 255),
                   (255, 150, 60), (90, 230, 200), (240, 240, 240), (230, 110, 200), (160, 170, 90)]
UNCLASSIFIED = (110, 110, 120)
PHEROMONE_TINTS = ((0.15, 0.85, 0.75), (0.85, 0.25, 0.75))   # channel A teal, channel B magenta


def lineage_rgb(color: torch.Tensor) -> torch.Tensor:
    """Genome colour (n, 3) in [0, 1] -> vivid display colour (stretch away from grey so lineages pop)."""
    mean = color.mean(1, keepdim=True)
    vivid = (mean + (color - mean) * 2.2).clamp(0, 1)
    return (vivid * 200 + 55).int()


def vivid(color) -> tuple[int, int, int]:
    """One genome colour tuple -> display colour."""
    return tuple(lineage_rgb(torch.tensor([color], dtype=torch.float32))[0].tolist())


def species_rgb(sid: int) -> tuple[int, int, int]:
    return SPECIES_PALETTE[sid % len(SPECIES_PALETTE)] if sid >= 0 else UNCLASSIFIED


def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    if not pygame.font.get_init():
        pygame.font.init()
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = pygame.font.SysFont("consolas,dejavusansmono,couriernew,monospace", size, bold=bold)
    return _FONTS[key]


_FONTS: dict = {}


def text(surface, s: str, pos, color=TEXT, size: int = 13, bold: bool = False, anchor: str = "topleft") -> pygame.Rect:
    img = get_font(size, bold).render(s, True, color)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect


def panel(surface, rect: pygame.Rect, alpha: int = 225, title: str | None = None) -> None:
    """Rounded translucent card with an optional small caps title."""
    if rect.w <= 0 or rect.h <= 0:
        return
    card = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(card, (*PANEL, alpha), card.get_rect(), border_radius=8)
    pygame.draw.rect(card, (*PANEL_EDGE, 255), card.get_rect(), 1, border_radius=8)
    surface.blit(card, rect.topleft)
    if title:
        text(surface, title.upper(), (rect.x + 10, rect.y + 7), DIM, 11, bold=True)


def _shade(col, k: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * k))) for c in col)


class Renderer:
    """Draws the world view. Owns the trail layer, glow sprites and pheromone overlay."""

    def __init__(self, px: int, world_size: float):
        self.px = px
        self.scale = px / world_size
        self.ui = px / 820          # sizes were tuned at 820 px; scale them for other sizes
        self.color_mode = "lineage"
        self.trails = True
        self.show_pheromone = True
        self.show_patches = True
        self.show_bonds = True
        self.trail = pygame.Surface((px, px))
        self.trail.fill(BG)
        self.fade = pygame.Surface((px, px), pygame.SRCALPHA)
        self.fade.fill((*BG, 38))
        self._glow: dict = {}

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
            t = world.genome.ptype[idx].tolist()
            return [TYPE_PALETTE[k % len(TYPE_PALETTE)] for k in t]
        if self.color_mode == "species":
            return [species_rgb(s) for s in world.species_id[idx].tolist()]
        return [tuple(c) for c in lineage_rgb(world.genome.color[idx]).tolist()]

    # ------------------------------------------------------------------ world view

    def draw(self, surface: pygame.Surface, world, selected: int | None = None, offset=(0, 0)) -> None:
        """Draw one frame of the world view onto `surface` at `offset` (px x px)."""
        s, ui = self.scale, self.ui
        idx = world.alive.nonzero().squeeze(1)
        # One tensor -> list conversion per quantity per frame; the Python loops below are the cost.
        pos = (world.pos[idx] * s).tolist()
        radius = (1.6 + 2.8 * world.body(idx)[0]).mul(ui).clamp_min(1.0).tolist()
        aggressive = (world.aggression[idx] > 0.3).tolist()
        signal = world.signal[idx].abs().tolist()
        cols = self.colors(world, idx)

        # Trails: only faint particle cores go on the persistent layer, so halos and rings stay crisp.
        if self.trails:
            self.trail.blit(self.fade, (0, 0))
            for (x, y), r, c in zip(pos, radius, cols):
                pygame.draw.circle(self.trail, _shade(c, 0.45), (x, y), max(1.0, r * 0.6))
            frame = self.trail.copy()
        else:
            frame = pygame.Surface((self.px, self.px))
            frame.fill(BG)

        if self.show_pheromone:
            frame.blit(self._pheromone_surface(world), (0, 0), special_flags=pygame.BLEND_ADD)
        if self.show_patches:
            pr = world.cfg.patch_radius * s * 2
            for x, y in (world.patch_pos * s).tolist():
                pygame.draw.circle(frame, PATCH, (x, y), pr, max(1, int(ui)))
        fr = max(1, round(1.3 * ui))
        for x, y in (world.food[world.food_alive] * s).tolist():
            pygame.draw.circle(frame, FOOD, (x, y), fr)

        if self.show_bonds:
            self._draw_bonds(frame, world, idx, cols)

        # Signalling particles glow; additive blending lets dense chatter bloom.
        for (x, y), r, c, sig in zip(pos, radius, cols, signal):
            if sig > 0.12:
                g = self._glow_sprite(c, sig, r)
                frame.blit(g, (x - g.get_width() / 2, y - g.get_height() / 2), special_flags=pygame.BLEND_ADD)
        ring_w = max(1, int(ui))
        for (x, y), r, c, agg in zip(pos, radius, cols, aggressive):
            pygame.draw.circle(frame, c, (x, y), r)
            if agg:
                pygame.draw.circle(frame, AGGRO, (x, y), r + 2 * ui, ring_w)

        if selected is not None and bool(world.alive[selected]):
            x, y = (world.pos[selected] * s).tolist()
            pygame.draw.circle(frame, (255, 255, 255), (x, y), 12 * ui, 1)
            pygame.draw.circle(frame, (90, 95, 110), (x, y), world.cfg.sense_radius * float(world.body([selected])[2][0]) * s, 1)
            self._draw_selected_bonds(frame, world, selected)
        surface.blit(frame, offset)

    def _draw_bonds(self, frame, world, idx, cols) -> None:
        bonds = world.bonds[idx]
        has = (bonds >= 0).any(1)
        if not bool(has.any()):
            return
        slots = idx[has].tolist()
        rows = bonds[has].tolist()
        col_of = dict(zip(idx.tolist(), cols))
        pos = world.pos.mul(self.scale).tolist()
        half = self.px / 2
        w = max(1, round(1.5 * self.ui))
        for a, partners in zip(slots, rows):
            ax, ay = pos[a]
            for b in partners:
                if b <= a:          # each pair once (and skips -1)
                    continue
                bx, by = pos[b]
                if abs(bx - ax) > half or abs(by - ay) > half:
                    continue        # pair straddles the torus seam; a screen-wide line would lie
                pygame.draw.line(frame, _shade(col_of.get(a, TEXT), 0.8), (ax, ay), (bx, by), w)

    def _draw_selected_bonds(self, frame, world, i: int) -> None:
        x, y = (world.pos[i] * self.scale).tolist()
        for b in world.bonds[i].tolist():
            if b >= 0:
                bx, by = (world.pos[b] * self.scale).tolist()
                if abs(bx - x) < self.px / 2 and abs(by - y) < self.px / 2:
                    pygame.draw.line(frame, (255, 255, 255), (x, y), (bx, by), 1)

    def _glow_sprite(self, col, level: float, r: float) -> pygame.Surface:
        """Radial falloff in the particle's colour, pre-multiplied for BLEND_ADD; cached by bucket."""
        q = (col[0] >> 5, col[1] >> 5, col[2] >> 5, min(5, int(level * 6)), int(r))
        sprite = self._glow.get(q)
        if sprite is None:
            rad = int(r * 3.2) + 2
            sprite = pygame.Surface((rad * 2, rad * 2))
            sprite.fill((0, 0, 0))
            base = [(c << 5) + 16 for c in q[:3]]
            k = 0.25 + 0.75 * (q[3] + 1) / 6
            steps = 10
            for j in range(steps, 0, -1):
                f = j / steps
                a = k * (1 - f) ** 1.4 * 0.9
                pygame.draw.circle(sprite, [int(c * a) for c in base], (rad, rad), rad * f)
            self._glow[q] = sprite
        return sprite

    def _pheromone_surface(self, world) -> pygame.Surface:
        """2 scent channels -> two additive tints, smoothly upscaled to the view."""
        ph = world.pheromone.detach().float().cpu()
        level = 1 - torch.exp(-ph * 6.0)               # saturating, so strong trails don't clip flat
        rgb = sum(level[c, :, :, None] * torch.tensor(PHEROMONE_TINTS[c]) for c in range(2))
        arr = (rgb.clamp(0, 1) * 120).to(torch.uint8).numpy()   # [x, y, rgb] matches surfarray layout
        small = pygame.surfarray.make_surface(arr)
        return pygame.transform.smoothscale(small, (self.px, self.px))


# ---------------------------------------------------------------------- panels


class Graph:
    """Tiny rolling line chart of a few stats, each series normalised to its own max."""

    def __init__(self, series: dict[str, tuple[int, int, int]], length: int = 300):
        self.series = series
        self.data = {k: deque(maxlen=length) for k in series}

    def push(self, stats: dict) -> None:
        for k in self.series:
            self.data[k].append(float(stats.get(k, 0)))

    def draw(self, surface, rect: pygame.Rect) -> None:
        panel(surface, rect, title="history")
        plot = pygame.Rect(rect.x + 10, rect.y + 24, rect.w - 20, rect.h - 32)
        for gy in range(1, 4):
            y = plot.y + plot.h * gy / 4
            pygame.draw.line(surface, (30, 33, 46), (plot.x, y), (plot.right, y))
        x_label = rect.right - 10
        for k, col in reversed(list(self.series.items())):
            values = self.data[k]
            if values:
                x_label = text(surface, f"{k} {values[-1]:g}", (x_label, rect.y + 6), col, 11, anchor="topright").x - 10
            if len(values) < 2:
                continue
            # Min-max per series (with a floor on the span) so slow drifts stay visible.
            lo, hi = min(values), max(values)
            span = max(hi - lo, 0.1 * abs(hi), 1e-6)
            lo = lo - 0.1 * span
            n = values.maxlen
            pts = [(plot.x + plot.w * j / (n - 1), plot.bottom - plot.h * (v - lo) / (span * 1.2))
                   for j, v in enumerate(values)]
            pygame.draw.aalines(surface, col, False, pts)


def draw_legend(surface, rect: pygame.Rect, world, active: bool) -> None:
    """Species counts with swatches matching the species colour mode, as stacked bars."""
    counts = world.species_counts()
    panel(surface, rect, title="species" + ("  (colour mode)" if active else ""))
    if not counts:
        text(surface, "classifying...", (rect.x + 10, rect.y + 26), DIM, 12)
        return
    names = world.species.names
    total = max(1, sum(counts.values()))
    y = rect.y + 25
    bar_x, bar_w = rect.x + 120, rect.w - 170
    for sid, name in enumerate(names):
        c = counts.get(name, 0)
        if c == 0:
            continue
        col = species_rgb(sid)
        pygame.draw.circle(surface, col, (rect.x + 16, y + 7), 5)
        text(surface, name, (rect.x + 28, y), TEXT, 12)
        pygame.draw.rect(surface, (32, 35, 48), (bar_x, y + 3, bar_w, 8), border_radius=3)
        pygame.draw.rect(surface, col, (bar_x, y + 3, max(2, bar_w * c / total), 8), border_radius=3)
        text(surface, str(c), (rect.right - 10, y), DIM, 12, anchor="topright")
        y += 17
        if y > rect.bottom - 14:
            break


def _act_color(a: float) -> tuple[int, int, int]:
    """Activation -> node fill: warm for positive, cool for negative, dark when silent."""
    m = min(1.0, abs(a))
    hue = POS_EDGE if a >= 0 else NEG_EDGE
    return tuple(int(28 + (c - 28) * m) for c in hue)


def draw_brain(surface, rect: pygame.Rect, world, i: int, max_edges: int = 60) -> None:
    """Layered node-link diagram of particle i's brain, lit by its last activations."""
    g = world.genome
    w1 = g.w1[i]                                    # (N_INPUTS, hidden)
    w2 = g.w2[i] + world.plastic[i]                 # effective, including lifetime learning
    inputs = world.last_inputs[i].tolist()
    hidden = world.last_hidden[i].tolist()
    outputs = world.last_outputs[i].tolist()
    n_h = len(hidden)

    label_w = 78
    top, bottom = rect.y + 8, rect.bottom - 8
    xs = (rect.x + label_w, rect.x + rect.w * 0.5, rect.right - 98)

    def column(n, pad=0.0):
        span = bottom - top
        return [top + span * (pad + (1 - 2 * pad) * (k + 0.5) / n) for k in range(n)]

    y_in, y_h, y_out = column(N_INPUTS), column(n_h, 0.04), column(len(outputs), 0.16)

    # Strongest edges, split between layers in proportion to their size so neither vanishes.
    k1 = round(max_edges * 0.6)
    k2 = max_edges - k1
    v1, f1 = w1.abs().flatten().topk(min(k1, w1.numel()))
    v2, f2 = w2.abs().flatten().topk(min(k2, w2.numel()))
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
        col = POS_EDGE if w >= 0 else NEG_EDGE
        width = 1 + int(rel * 2.5)
        pygame.draw.line(layer, (*col, int(40 + 190 * rel)), (x0 - ox, y0 - oy), (x1 - ox, y1 - oy), width)
    surface.blit(layer, rect.topleft)

    def node(x, y, a, r):
        pygame.draw.circle(surface, _act_color(a), (x, y), r)
        pygame.draw.circle(surface, (70, 75, 95), (x, y), r, 1)

    for k, (y, a) in enumerate(zip(y_in, inputs)):
        node(xs[0], y, a, 4)
        text(surface, SENSE_NAMES[k], (xs[0] - 8, y), DIM if abs(a) < 0.05 else TEXT, 10, anchor="midright")
    for y, a in zip(y_h, hidden):
        node(xs[1], y, a, 5)
    for k, (y, a) in enumerate(zip(y_out, outputs)):
        node(xs[2], y, a, 6)
        text(surface, f"{OUTPUT_NAMES[k]} {a:+.2f}", (xs[2] + 10, y), TEXT, 10, anchor="midleft")


def draw_inspector(surface, rect: pygame.Rect, world, i: int, color) -> None:
    """Selected particle: identity, body, diet, learning, then its brain."""
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

    panel(surface, rect)
    pygame.draw.circle(surface, color, (rect.x + 18, rect.y + 17), 7)
    text(surface, f"#{int(world.uid[i])}", (rect.x + 32, rect.y + 8), TEXT, 15, bold=True)
    text(surface, species, (rect.right - 10, rect.y + 9), species_rgb(sid), 13, bold=True, anchor="topright")

    x0, y = rect.x + 12, rect.y + 32
    prey_share = prey / (prey + food) if prey + food > 0 else 0.0
    rows = [
        [("gen", f"{int(world.generation[i])}"), ("age", f"{int(world.age[i])}"),
         ("type", f"{int(g.ptype[i])}"), ("bonds", f"{bonds}/{cfg.max_bonds}")],
        [("size", f"{size:.2f}"), ("speed", f"{speed:.2f}"), ("sense", f"{sense:.2f}"),
         ("bud", f"{float(g.bud[i]):.2f}")],
        [("prey", f"{prey:.1f}"), ("food", f"{food:.1f}"), ("diet", f"{prey_share:.0%} meat"), ("", "")],
        [("σ in", f"{sigma[0]:.2f}"), ("out", f"{sigma[1]:.2f}"), ("phys", f"{sigma[2]:.2f}"),
         ("body", f"{sigma[3]:.2f}")],
    ]
    col_w = (rect.w - 24) / 4
    for row in rows:
        for k, (label, value) in enumerate(row):
            if label:
                r = text(surface, label, (x0 + k * col_w, y + 1), DIM, 11)
                text(surface, value, (r.right + 4, y), TEXT, 12)
        y += 16
    text(surface, "learn", (x0, y + 1), DIM, 11)
    text(surface, f"{eta:.4f}", (x0 + 40, y), TEXT, 12)
    text(surface, "σ = evolved mutation step", (x0 + col_w * 2, y + 1), DIM, 10)
    y += 18

    # Energy bar: how close to splitting.
    text(surface, f"energy {energy:.2f}", (x0, y), TEXT, 12)
    bar = pygame.Rect(x0 + 100, y + 4, rect.w - 124, 8)
    pygame.draw.rect(surface, (32, 35, 48), bar, border_radius=3)
    pygame.draw.rect(surface, ACCENT, (bar.x, bar.y, max(2, bar.w * min(1, energy / full)), bar.h), border_radius=3)
    y += 20
    pygame.draw.line(surface, PANEL_EDGE, (rect.x + 8, y), (rect.right - 8, y))
    text(surface, "BRAIN", (x0, y + 4), DIM, 11, bold=True)
    text(surface, "warm +  cool -   top 60 weights", (rect.right - 10, y + 5), DIM, 10, anchor="topright")
    draw_brain(surface, pygame.Rect(rect.x + 4, y + 18, rect.w - 8, rect.bottom - y - 22), world, i)


class TreePanel:
    """Dendrogram of the living population's family tree; laid out at most once per `interval` s."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.last = -math.inf
        self.surface: pygame.Surface | None = None
        self.where: dict[int, tuple[float, float]] = {}   # uid -> (x, y) inside the panel
        self.parent: dict[int, int] = {}
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.info = ""

    def draw(self, surface, rect: pygame.Rect, world, selected_uid: int | None = None, force: bool = False) -> None:
        now = time.perf_counter()
        if force or self.surface is None or rect != self.rect or now - self.last > self.interval:
            self._layout(world, rect)
            self.last = now
        surface.blit(self.surface, rect.topleft)
        if selected_uid is not None and selected_uid in self.where:
            self._highlight(surface, rect, world, selected_uid)

    def _layout(self, world, rect: pygame.Rect) -> None:
        self.rect = rect.copy()
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel(surf, surf.get_rect(), alpha=235, title="family tree")
        alive_uids = world.uid[world.alive].tolist()
        nodes = world.lineage.living_tree(alive_uids)
        self.where, self.parent = {}, {}
        self.surface = surf
        if not nodes:
            return
        plot = pygame.Rect(10, 26, rect.w - 20, rect.h - 46)
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
        thin = slots > plot.h
        for u in order:
            n = by_uid[u]
            x, y = px(n["born"]), py(y_slot[u])
            self.where[u] = (x + rect.x, y + rect.y)
            self.parent[u] = n["parent"]
            col = vivid(n["color"])
            if n["parent"] >= 0:
                p = by_uid[n["parent"]]
                xp, yp = px(p["born"]), py(y_slot[p["uid"]])
                pygame.draw.line(surf, _shade(col, 0.75), (xp, yp), (xp, y))
                pygame.draw.line(surf, _shade(col, 0.75), (xp, y), (x, y))
            if n["alive"]:
                pygame.draw.line(surf, _shade(col, 0.35 if thin else 0.55), (x, y), (x_now, y))
                surf.fill(col, (int(x_now) - 1, int(y), 3, 1 if thin else 2))
        # Time axis.
        axis_y = rect.h - 16
        pygame.draw.line(surf, PANEL_EDGE, (plot.x, axis_y), (plot.right, axis_y))
        text(surf, f"step {t0}", (plot.x, axis_y + 2), DIM, 10)
        text(surf, f"{now}", (plot.right, axis_y + 2), DIM, 10, anchor="topright")
        roots_alive = len(roots)
        self.info = f"{len(alive_uids)} living  {len(nodes)} nodes  {roots_alive} roots"
        text(surf, self.info, (rect.w - 10, 8), DIM, 10, anchor="topright")

    def _highlight(self, surface, rect, world, uid: int) -> None:
        """Ancestry of the selected particle drawn over the cached tree."""
        chain = world.lineage.ancestors(uid, depth=100_000)
        pts = [self.where[u] for u in chain if u in self.where]
        x_now = rect.x + 10 + (rect.w - 20)
        x, y = self.where[uid]
        pygame.draw.line(surface, (255, 255, 255), (x, y), (x_now, y), 2)
        for (xc, yc), (xp, yp) in zip(pts, pts[1:]):
            pygame.draw.line(surface, (255, 255, 255), (xp, yp), (xp, yc), 2)
            pygame.draw.line(surface, (255, 255, 255), (xp, yc), (xc, yc), 2)
        pygame.draw.circle(surface, (255, 255, 255), (x_now, y), 4)
        if pts:
            pygame.draw.circle(surface, ACCENT, pts[-1], 4)
