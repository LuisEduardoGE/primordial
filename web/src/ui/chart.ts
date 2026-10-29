/** Population over time: total plus one line per behavioural species, with labelled axes. */
import type { Stats } from '../sim/stats';
import { SPECIES_PALETTE, css } from '../palette';

interface Sample { step: number; pop: number; species: number[] }

export class Chart {
  private canvas: HTMLCanvasElement;
  private legend: HTMLElement;
  private samples: Sample[] = [];
  private names: string[] = [];
  private max = 600;
  private lastStep = -1;
  private legendKey = '';

  constructor(canvas: HTMLCanvasElement, legend: HTMLElement) {
    this.canvas = canvas;
    this.legend = legend;
  }

  reset(): void { this.samples = []; this.lastStep = -1; this.names = []; }

  push(s: Stats): void {
    if (s.step - this.lastStep < 20) return;   // one sample per >= 20 sim steps
    this.lastStep = s.step;
    this.names = s.speciesNames;
    this.samples.push({ step: s.step, pop: s.population, species: s.speciesNames.map((n) => s.species[n] ?? 0) });
    if (this.samples.length > this.max) this.samples.shift();
  }

  draw(): void {
    const c = this.canvas, dpr = Math.min(devicePixelRatio, 2);
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext('2d')!;
    const cs = getComputedStyle(document.documentElement);
    const ink = cs.getPropertyValue('--chart-total').trim(), muted = cs.getPropertyValue('--muted').trim(), grid = cs.getPropertyValue('--grid').trim();
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    this.drawLegend(ink);
    const S = this.samples;
    const L = 48, R = 8, T = 6, B = 22;          // plot margins
    const pw = w - L - R, ph = h - T - B;
    let top = 100;
    for (const s of S) top = Math.max(top, s.pop);
    top = niceCeil(top);
    const s0 = S.length ? S[0].step : 0, s1 = S.length ? Math.max(S[S.length - 1].step, s0 + 1) : 1;
    const x = (step: number) => L + ((step - s0) / (s1 - s0)) * pw;
    const y = (v: number) => T + ph - (v / top) * ph;

    g.font = '11px ' + cs.getPropertyValue('--font');
    g.fillStyle = muted;
    g.strokeStyle = grid;
    g.lineWidth = 1;
    g.textBaseline = 'middle';
    g.textAlign = 'right';
    const div = tickDivisions(top);
    for (let k = 0; k <= div; k++) {
      const v = (top * k) / div, yy = Math.round(y(v)) + 0.5;
      g.beginPath(); g.moveTo(L, yy); g.lineTo(L + pw, yy); g.stroke();
      g.fillText(compact(v), L - 6, yy);
    }
    g.textBaseline = 'top';
    g.textAlign = 'left';
    g.fillText(compact(s0), L, T + ph + 6);
    g.textAlign = 'right';
    g.fillText(`${compact(s1)} steps`, L + pw, T + ph + 6);
    g.save();
    g.translate(10, T + ph / 2);
    g.rotate(-Math.PI / 2);
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText('particles', 0, 0);
    g.restore();
    if (S.length < 2) return;

    const k = Math.max(...S.map((s) => s.species.length));
    for (let j = 0; j < k; j++) {
      g.beginPath();
      let started = false;
      for (const s of S) {
        const v = s.species[j];
        if (v === undefined) { started = false; continue; }
        if (!started) { g.moveTo(x(s.step), y(v)); started = true; } else g.lineTo(x(s.step), y(v));
      }
      g.strokeStyle = css(SPECIES_PALETTE[j % SPECIES_PALETTE.length]);
      g.lineWidth = 1.25;
      g.stroke();
    }
    g.beginPath();
    S.forEach((s, i) => (i ? g.lineTo(x(s.step), y(s.pop)) : g.moveTo(x(s.step), y(s.pop))));
    g.strokeStyle = ink;
    g.lineWidth = 1.75;
    g.stroke();
  }

  private drawLegend(ink: string): void {
    const key = this.names.join('|') + ink;
    if (key === this.legendKey) return;
    this.legendKey = key;
    const item = (color: string, label: string) => `<span><i style="background:${color}"></i>${label}</span>`;
    this.legend.innerHTML = item(ink, 'Total') + this.names.map((n, j) => item(css(SPECIES_PALETTE[j % SPECIES_PALETTE.length]), n)).join('');
  }
}

/** Divisions giving round tick values for a niceCeil() top (e.g. 5000 -> 5 ticks of 1000). */
function tickDivisions(top: number): number {
  const lead = top / Math.pow(10, Math.floor(Math.log10(top)));
  return lead === 2 ? 4 : 5;
}

function niceCeil(v: number): number {
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

function compact(v: number): string {
  return v >= 10000 ? `${(v / 1000).toFixed(0)}k` : v >= 1000 ? `${(v / 1000).toFixed(1).replace(/\.0$/, '')}k` : `${Math.round(v)}`;
}
