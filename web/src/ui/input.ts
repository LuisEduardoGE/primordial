/** Pointer input on the canvas: drag to pan, wheel / pinch to zoom, click to use a tool. */
import type { View } from '../render/view';
import type { UiState } from './controls';

export interface InputHandlers {
  /** World coords wrapped into [0, W), plus the unwrapped coords near the camera (for effects). */
  click(wx: number, wy: number, uwx: number, uwy: number): void;
  toggleHud(): void;
}

export function attachInput(canvas: HTMLCanvasElement, view: View, _state: UiState, h: InputHandlers): void {
  const pointers = new Map<number, { x: number; y: number }>();
  let downAt: { x: number; y: number; t: number } | null = null;
  let moved = false;
  let pinchDist = 0;

  const rel = (e: PointerEvent | WheelEvent) => {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  canvas.addEventListener('pointerdown', (e) => {
    try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic or already-released pointer */ }
    const p = rel(e);
    pointers.set(e.pointerId, p);
    if (pointers.size === 1) { downAt = { ...p, t: performance.now() }; moved = false; }
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
      moved = true;
    }
  });

  canvas.addEventListener('pointermove', (e) => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    const p = rel(e);
    pointers.set(e.pointerId, p);
    if (pointers.size === 1) {
      if (downAt && Math.hypot(p.x - downAt.x, p.y - downAt.y) > 5) moved = true;
      if (moved) {
        canvas.classList.add('grab');
        view.cam.pan(p.x - prev.x, p.y - prev.y);
      }
    } else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchDist > 0) view.cam.zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, d / pinchDist);
      pinchDist = d;
    }
  });

  const up = (e: PointerEvent) => {
    if (!pointers.has(e.pointerId)) return;
    pointers.delete(e.pointerId);
    canvas.classList.remove('grab');
    if (pointers.size === 0 && downAt && !moved) {
      const p = rel(e);
      const [uwx, uwy] = view.cam.toWorld(p.x, p.y);
      const W = view.W;
      h.click(((uwx % W) + W) % W, ((uwy % W) + W) % W, uwx, uwy);
    }
    if (pointers.size === 0) downAt = null;
  };
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', (e) => { pointers.delete(e.pointerId); downAt = null; });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const p = rel(e);
    const k = e.deltaMode === 1 ? 0.05 : 0.0015;
    view.cam.zoomAt(p.x, p.y, Math.exp(-e.deltaY * k));
  }, { passive: false });
}
