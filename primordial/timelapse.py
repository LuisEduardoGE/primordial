"""Headless timelapse: step a World and render every N steps to a video, no window needed.

Also home to `VideoWriter`, which the live viewer reuses for its record key.
"""

import os
import time
from pathlib import Path

import pygame

from .render import Renderer, card, get_font, text, theme


class VideoWriter:
    """Frames -> mp4 through imageio-ffmpeg, falling back to an animated GIF via Pillow."""

    def __init__(self, path: str | Path, fps: int = 30):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frames = 0
        self._gif: list | None = None
        try:
            import imageio.v2 as imageio

            if self.path.suffix.lower() == ".gif":
                raise RuntimeError("gif requested")
            # macro_block_size=1: we crop to even sizes ourselves instead of letting imageio rescale.
            self._writer = imageio.get_writer(str(self.path), fps=fps, codec="libx264", quality=7,
                                              macro_block_size=1, pixelformat="yuv420p")
        except Exception:
            self.path = self.path.with_suffix(".gif")
            self._writer = None
            self._gif = []

    def add(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        frame = pygame.surfarray.array3d(surface)[: w - w % 2, : h - h % 2].swapaxes(0, 1)
        if self._writer is not None:
            self._writer.append_data(frame)
        else:
            from PIL import Image

            self._gif.append(Image.fromarray(frame.copy()))
        self.frames += 1

    def close(self) -> Path:
        if self._writer is not None:
            self._writer.close()
        elif self._gif:
            self._gif[0].save(self.path, save_all=True, append_images=self._gif[1:],
                              duration=int(1000 / self.fps), loop=0)
        return self.path


def run_timelapse(world, steps: int, every: int, out_path: str, size: int = 720, fps: int = 30) -> Path:
    """Advance `world` by `steps`, writing one frame every `every` steps. Returns the file written."""
    if not pygame.display.get_init():
        # Offscreen only. Set here, not at import, so importing this module never hides the viewer window.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    renderer = Renderer(size, world.cfg.world_size)
    frame = pygame.Surface((size, size))
    writer = VideoWriter(out_path, fps)
    t0 = time.perf_counter()
    try:
        for i in range(1, steps + 1):
            world.step()
            if i % every == 0 or i == steps:
                renderer.draw(frame, world)
                _caption(frame, world)
                writer.add(frame)
            if i % max(every * 50, 500) == 0:
                rate = i / (time.perf_counter() - t0)
                print(f"step {world.step_count}  pop {world.population}  frames {writer.frames}  "
                      f"({rate:.0f} steps/s)", flush=True)
    finally:
        path = writer.close()
    return path


def _caption(frame: pygame.Surface, world) -> None:
    """Small bordered label: step and population, so a timelapse frame stands on its own."""
    size = max(10, frame.get_width() // 64)
    label = f"Step {world.step_count:,}  ·  Population {world.population:,}"
    w, h = get_font(size).size(label)
    box = pygame.Rect(8, 8, w + 2 * size, h + size)
    card(frame, box)
    text(frame, label, box.center, theme().text, size, anchor="center")


def cmd_timelapse(args) -> None:
    from .config import Config
    from .world import World

    if args.load:
        world = World.load(args.load)
        world.cfg.override(args.set)
    else:
        world = World(Config(seed=args.seed).override(args.set))
    out = args.out or f"recordings/timelapse_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    path = run_timelapse(world, args.steps, args.every, out, args.size, args.fps)
    print(f"wrote {path}")


def add_subparser(subparsers) -> None:
    """Register `primordial timelapse ...` on an argparse subparsers object."""
    t = subparsers.add_parser("timelapse", help="render a headless timelapse video")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--load", help="start from a checkpoint")
    t.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    t.add_argument("--steps", type=int, default=20_000)
    t.add_argument("--every", type=int, default=20, help="simulation steps per video frame")
    t.add_argument("--out", help="output file (.mp4, or .gif); default recordings/timelapse_<time>.mp4")
    t.add_argument("--size", type=int, default=720, help="frame width and height in pixels")
    t.add_argument("--fps", type=int, default=30)
    t.set_defaults(func=cmd_timelapse)
