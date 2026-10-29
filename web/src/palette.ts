/** Colour helpers shared by the worker (per-particle colours) and the UI (legends). */

/**
 * Lineage colour: the hue of the inherited genome colour at a muted, mid saturation and
 * mid lightness, so families stay distinguishable and every mark reads on a white background.
 */
export function lineageColor(r: number, g: number, b: number, out: [number, number, number]): void {
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
  let h = 0;
  if (d > 1e-6) {
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h /= 6;
    if (h < 0) h += 1;
  }
  const s0 = max > 0 ? d / max : 0;
  hsl(h, 0.3 + 0.16 * Math.min(1, s0 * 1.4), 0.56 + 0.06 * (max - 0.5), out);
}

export function hsl(h: number, s: number, l: number, out: [number, number, number]): void {
  const k = (n: number) => (n + h * 12) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  out[0] = f(0); out[1] = f(8); out[2] = f(4);
}

export function hsv(h: number, s: number, v: number, out: [number, number, number]): void {
  const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (((i % 6) + 6) % 6) {
    case 0: out[0] = v; out[1] = t; out[2] = p; break;
    case 1: out[0] = q; out[1] = v; out[2] = p; break;
    case 2: out[0] = p; out[1] = v; out[2] = t; break;
    case 3: out[0] = p; out[1] = q; out[2] = v; break;
    case 4: out[0] = t; out[1] = p; out[2] = v; break;
    default: out[0] = v; out[1] = p; out[2] = q;
  }
}

const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);

/** Species colours (by cluster index): Tableau 10. */
export const SPECIES_PALETTE: number[][] = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#b07aa1', '#76b7b2', '#edc948', '#9c755f', '#ff9da7', '#bab0ac'].map(hex);

/** Physical type colours (particle-life types): ColorBrewer Dark2, slightly softened. */
export const TYPE_PALETTE: number[][] = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02'].map(hex);

export const css = (c: number[]) => `rgb(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)})`;
