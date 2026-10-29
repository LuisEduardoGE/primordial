/** Orthographic 2D camera over a torus: pan/zoom state, screen <-> world conversion. */
import * as THREE from 'three';

export class Camera2D {
  camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
  W: number;
  cx: number;
  cy: number;
  /** Screen pixels (CSS) per world unit. */
  pxPerUnit = 100;
  w = 1;
  h = 1;
  private fitted = false;
  version = 0;   // bumps on every view change (trails reset)

  constructor(W: number) {
    this.W = W;
    this.cx = W / 2;
    this.cy = W / 2;
  }

  resize(w: number, h: number): void {
    this.w = Math.max(1, w);
    this.h = Math.max(1, h);
    if (!this.fitted) { this.fit(); this.pxPerUnit *= 1.9; this.fitted = true; }
    this.clampZoom();
    this.apply();
  }

  /** Fill the screen with the world (the larger screen side spans the whole torus). */
  fit(): void {
    this.pxPerUnit = Math.max(this.w, this.h) / this.W;
    this.cx = this.W / 2;
    this.cy = this.W / 2;
    this.apply();
  }

  minPxPerUnit(): number {
    return Math.max(this.w, this.h) / (2 * this.W) * 1.0001;
  }

  clampZoom(): void {
    this.pxPerUnit = Math.min(Math.max(this.pxPerUnit, this.minPxPerUnit()), 40000);
  }

  apply(): void {
    const W = this.W;
    this.cx = ((this.cx % W) + W) % W;
    this.cy = ((this.cy % W) + W) % W;
    const hw = this.w / 2 / this.pxPerUnit, hh = this.h / 2 / this.pxPerUnit;
    const c = this.camera;
    c.left = -hw; c.right = hw; c.top = hh; c.bottom = -hh;
    c.position.set(this.cx, this.cy, 5);
    c.updateProjectionMatrix();
    c.updateMatrixWorld();
    this.version++;
    c.userData.version = this.version;
  }

  bounds(): [number, number, number, number] {
    const hw = this.w / 2 / this.pxPerUnit, hh = this.h / 2 / this.pxPerUnit;
    return [this.cx - hw, this.cx + hw, this.cy - hh, this.cy + hh];
  }

  /** Screen (CSS px, origin top-left) to world coordinates, unwrapped (near the camera). */
  toWorld(sx: number, sy: number): [number, number] {
    return [this.cx + (sx - this.w / 2) / this.pxPerUnit, this.cy - (sy - this.h / 2) / this.pxPerUnit];
  }

  pan(dxPx: number, dyPx: number): void {
    this.cx -= dxPx / this.pxPerUnit;
    this.cy += dyPx / this.pxPerUnit;
    this.apply();
  }

  /** Zoom by `factor` keeping the world point under screen (sx, sy) fixed. */
  zoomAt(sx: number, sy: number, factor: number): void {
    const [wx, wy] = this.toWorld(sx, sy);
    this.pxPerUnit *= factor;
    this.clampZoom();
    this.cx = wx - (sx - this.w / 2) / this.pxPerUnit;
    this.cy = wy + (sy - this.h / 2) / this.pxPerUnit;
    this.apply();
  }
}
