/** Colour helpers shared by the worker (per-particle colours) and the UI (legends). */

/** Lineage colour: the genome colour with its hue kept but saturation and value lifted, so lineages read vividly. */
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
  const s = 0.42 + 0.4 * Math.min(1, s0 * 1.5);
  const v = 0.78 + 0.2 * max;
  hsv(h, s, v, out);
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

/** Species colours (by cluster index). */
export const SPECIES_PALETTE: number[][] = [
  [0.36, 0.78, 1.0], [1.0, 0.62, 0.26], [0.55, 0.95, 0.5], [0.98, 0.42, 0.72], [0.82, 0.7, 1.0],
  [1.0, 0.9, 0.4], [0.4, 1.0, 0.85], [1.0, 0.5, 0.45],
];

/** Physical type colours (particle-life types). */
export const TYPE_PALETTE: number[][] = [
  [1.0, 0.45, 0.4], [0.4, 0.85, 1.0], [0.75, 1.0, 0.45], [0.95, 0.6, 1.0], [1.0, 0.85, 0.35], [0.5, 1.0, 0.8],
];

export const css = (c: number[]) => `rgb(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)})`;
