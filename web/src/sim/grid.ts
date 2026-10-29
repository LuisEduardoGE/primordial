/**
 * Uniform-grid spatial hash on a torus. Points are counting-sorted into cells each
 * rebuild (O(n)); queries scan the block of cells covering the query radius.
 *
 * Mirrors primordial/spatial.py, but produces neighbour lists in CSR form
 * (start offsets + flat arrays) instead of padded tensors.
 */

export function wrapDelta(d: number, world: number): number {
  return d - Math.round(d / world) * world;
}

export class Grid {
  g = 1;
  cell = 1;
  world = 1;
  n = 0;
  counts = new Int32Array(1);
  starts = new Int32Array(2);
  items = new Int32Array(0);
  private cellOf = new Int32Array(0);
  xs: ArrayLike<number> = [];
  ys: ArrayLike<number> = [];

  /** Rebuild from `n` points with cells no smaller than `minCell`. */
  build(xs: ArrayLike<number>, ys: ArrayLike<number>, n: number, world: number, minCell: number): void {
    this.xs = xs;
    this.ys = ys;
    this.n = n;
    this.world = world;
    const g = Math.max(1, Math.min(512, Math.floor(world / minCell)));
    this.g = g;
    this.cell = world / g;
    const nc = g * g;
    if (this.counts.length < nc) {
      this.counts = new Int32Array(nc);
      this.starts = new Int32Array(nc + 1);
    } else {
      this.counts.fill(0, 0, nc);
    }
    if (this.items.length < n) {
      this.items = new Int32Array(Math.max(n, 16) * 2);
      this.cellOf = new Int32Array(this.items.length);
    }
    const inv = g / world;
    const counts = this.counts;
    const cellOf = this.cellOf;
    for (let i = 0; i < n; i++) {
      let cx = Math.floor(xs[i] * inv);
      let cy = Math.floor(ys[i] * inv);
      if (cx < 0) cx = 0; else if (cx >= g) cx = g - 1;
      if (cy < 0) cy = 0; else if (cy >= g) cy = g - 1;
      const c = cx * g + cy;
      cellOf[i] = c;
      counts[c]++;
    }
    const starts = this.starts;
    starts[0] = 0;
    for (let c = 0; c < nc; c++) starts[c + 1] = starts[c] + counts[c];
    // reuse counts as write cursors
    for (let c = 0; c < nc; c++) counts[c] = starts[c];
    const items = this.items;
    for (let i = 0; i < n; i++) items[counts[cellOf[i]]++] = i;
    for (let c = 0; c < nc; c++) counts[c] -= starts[c];
  }

  /**
   * Visit every point within `radius` of (qx, qy). The callback gets the point index,
   * the shortest torus displacement (dx, dy) from the query to it, and the squared distance.
   * Points are visited in cell order; within a cell in index order.
   */
  query(qx: number, qy: number, radius: number, fn: (j: number, dx: number, dy: number, d2: number) => void): void {
    const { g, world, xs, ys, items, starts } = this;
    const r2 = radius * radius;
    const inv = g / world;
    let R = Math.ceil(radius * inv);
    const full = 2 * R + 1 >= g;
    let cx0: number, cy0: number;
    if (full) {
      R = 0;
      cx0 = 0;
      cy0 = 0;
    } else {
      cx0 = Math.min(g - 1, Math.max(0, Math.floor(qx * inv))) - R;
      cy0 = Math.min(g - 1, Math.max(0, Math.floor(qy * inv))) - R;
    }
    const span = full ? g : 2 * R + 1;
    const half = world * 0.5;
    for (let ix = 0; ix < span; ix++) {
      let cx = cx0 + ix;
      if (cx < 0) cx += g; else if (cx >= g) cx -= g;
      for (let iy = 0; iy < span; iy++) {
        let cy = cy0 + iy;
        if (cy < 0) cy += g; else if (cy >= g) cy -= g;
        const c = cx * g + cy;
        const end = starts[c + 1];
        for (let k = starts[c]; k < end; k++) {
          const j = items[k];
          let dx = xs[j] - qx;
          let dy = ys[j] - qy;
          if (dx > half) dx -= world; else if (dx < -half) dx += world;
          if (dy > half) dy -= world; else if (dy < -half) dy += world;
          const d2 = dx * dx + dy * dy;
          if (d2 <= r2) fn(j, dx, dy, d2);
        }
      }
    }
  }
}

/** Result of `nearest` (a shared object, so queries don't allocate). */
export const nearestHit = { j: -1, dx: 0, dy: 0, d: Infinity };

// module-level search state for the ring scan (avoids closures in the hot path)
let sBest = Infinity, sJ = -1, sDx = 0, sDy = 0;

function scanCell(grid: Grid, c: number, qx: number, qy: number, r2: number): void {
  const { world, xs, ys, items, starts } = grid;
  const half = world * 0.5;
  const end = starts[c + 1];
  for (let k = starts[c]; k < end; k++) {
    const j = items[k];
    let dx = xs[j] - qx;
    let dy = ys[j] - qy;
    if (dx > half) dx -= world; else if (dx < -half) dx += world;
    if (dy > half) dy -= world; else if (dy < -half) dy += world;
    const d2 = dx * dx + dy * dy;
    if (d2 <= r2 && d2 < sBest) { sBest = d2; sJ = j; sDx = dx; sDy = dy; }
  }
}

/**
 * Nearest grid point to (qx, qy) within `radius`. Scans rings of cells outward from the
 * query's cell and stops as soon as no farther ring can hold anything closer.
 */
export function nearest(grid: Grid, qx: number, qy: number, radius: number): boolean {
  const { g, world } = grid;
  const inv = g / world;
  const R = Math.ceil(radius * inv);
  if (2 * R + 1 >= g) return nearestFull(grid, qx, qy, radius);
  const cell = world / g, r2 = radius * radius;
  const ccx = Math.min(g - 1, Math.max(0, Math.floor(qx * inv)));
  const ccy = Math.min(g - 1, Math.max(0, Math.floor(qy * inv)));
  sBest = Infinity; sJ = -1; sDx = 0; sDy = 0;
  for (let r = 0; r <= R; r++) {
    for (let ix = -r; ix <= r; ix++) {
      let cx = ccx + ix;
      if (cx < 0) cx += g; else if (cx >= g) cx -= g;
      const edge = ix === -r || ix === r;
      const step = edge || r === 0 ? 1 : 2 * r;
      for (let iy = -r; iy <= r; iy += step) {
        let cy = ccy + iy;
        if (cy < 0) cy += g; else if (cy >= g) cy -= g;
        scanCell(grid, cx * g + cy, qx, qy, r2);
      }
    }
    const reach = r * cell;
    if (sJ >= 0 && sBest <= reach * reach) break;
  }
  nearestHit.j = sJ; nearestHit.dx = sDx; nearestHit.dy = sDy; nearestHit.d = Math.sqrt(sBest);
  return sJ >= 0;
}

/** Brute scan of the whole block covering `radius` (used when it wraps the whole grid). */
function nearestFull(grid: Grid, qx: number, qy: number, radius: number): boolean {
  const { g, world, xs, ys, items, starts } = grid;
  const inv = g / world;
  const R = Math.ceil(radius * inv);
  const full = 2 * R + 1 >= g;
  const span = full ? g : 2 * R + 1;
  const cx0 = full ? 0 : Math.min(g - 1, Math.max(0, Math.floor(qx * inv))) - R;
  const cy0 = full ? 0 : Math.min(g - 1, Math.max(0, Math.floor(qy * inv))) - R;
  const half = world * 0.5;
  const r2 = radius * radius;
  let best = Infinity, bj = -1, bdx = 0, bdy = 0;
  for (let ix = 0; ix < span; ix++) {
    let cx = cx0 + ix;
    if (cx < 0) cx += g; else if (cx >= g) cx -= g;
    for (let iy = 0; iy < span; iy++) {
      let cy = cy0 + iy;
      if (cy < 0) cy += g; else if (cy >= g) cy -= g;
      const c = cx * g + cy;
      const end = starts[c + 1];
      for (let k = starts[c]; k < end; k++) {
        const j = items[k];
        let dx = xs[j] - qx;
        let dy = ys[j] - qy;
        if (dx > half) dx -= world; else if (dx < -half) dx += world;
        if (dy > half) dy -= world; else if (dy < -half) dy += world;
        const d2 = dx * dx + dy * dy;
        if (d2 <= r2 && d2 < best) { best = d2; bj = j; bdx = dx; bdy = dy; }
      }
    }
  }
  nearestHit.j = bj; nearestHit.dx = bdx; nearestHit.dy = bdy; nearestHit.d = Math.sqrt(best);
  return bj >= 0;
}

/** CSR neighbour list: for query i, entries [start[i], start[i+1]). */
export class NeighbourList {
  start = new Int32Array(1);
  j = new Int32Array(1024);
  dx = new Float32Array(1024);
  dy = new Float32Array(1024);
  dist = new Float32Array(1024);
  len = 0;

  reset(nq: number): void {
    if (this.start.length < nq + 1) this.start = new Int32Array((nq + 1) * 2);
    this.len = 0;
  }

  push(j: number, dx: number, dy: number, d: number): void {
    if (this.len >= this.j.length) this.grow();
    const k = this.len++;
    this.j[k] = j;
    this.dx[k] = dx;
    this.dy[k] = dy;
    this.dist[k] = d;
  }

  private grow(): void {
    const n = this.j.length * 2;
    const j = new Int32Array(n); j.set(this.j); this.j = j;
    const dx = new Float32Array(n); dx.set(this.dx); this.dx = dx;
    const dy = new Float32Array(n); dy.set(this.dy); this.dy = dy;
    const d = new Float32Array(n); d.set(this.dist); this.dist = d;
  }
}

/**
 * All points of `grid` within `radius` of each of the grid's own points (self excluded),
 * written to `out` as a CSR list. This is the hot path, so it is written without closures.
 */
export function selfNeighbours(grid: Grid, radius: number, out: NeighbourList): void {
  const n = grid.n;
  out.reset(n);
  const { g, world, xs, ys, items, starts } = grid;
  const r2 = radius * radius;
  const inv = g / world;
  const R = Math.ceil(radius * inv);
  const full = 2 * R + 1 >= g;
  const span = full ? g : 2 * R + 1;
  const half = world * 0.5;
  for (let i = 0; i < n; i++) {
    out.start[i] = out.len;
    const qx = xs[i];
    const qy = ys[i];
    const cx0 = full ? 0 : Math.min(g - 1, Math.max(0, Math.floor(qx * inv))) - R;
    const cy0 = full ? 0 : Math.min(g - 1, Math.max(0, Math.floor(qy * inv))) - R;
    for (let ix = 0; ix < span; ix++) {
      let cx = cx0 + ix;
      if (cx < 0) cx += g; else if (cx >= g) cx -= g;
      for (let iy = 0; iy < span; iy++) {
        let cy = cy0 + iy;
        if (cy < 0) cy += g; else if (cy >= g) cy -= g;
        const c = cx * g + cy;
        const end = starts[c + 1];
        for (let k = starts[c]; k < end; k++) {
          const j = items[k];
          if (j === i) continue;
          let dx = xs[j] - qx;
          let dy = ys[j] - qy;
          if (dx > half) dx -= world; else if (dx < -half) dx += world;
          if (dy > half) dy -= world; else if (dy < -half) dy += world;
          const d2 = dx * dx + dy * dy;
          if (d2 <= r2) out.push(j, dx, dy, Math.sqrt(d2));
        }
      }
    }
  }
  out.start[n] = out.len;
}
