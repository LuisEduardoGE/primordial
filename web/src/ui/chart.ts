/** Rolling population chart: total as a filled line, species as stacked thin lines. */
import type { Stats } from '../sim/stats';
import { SPECIES_PALETTE, css } from '../palette';

interface Sample { pop: number; species: number[] }

export class Chart {
  private canvas: HTMLCanvasElement;

  private samples: Sample[] = [];
  private max = 600;
  private lastStep = -1;

  constructor(canvas: HTMLCanvasElement, legend: HTMLElement) {
    this.canvas = canvas;

    legend.innerHTML = `<span><i style="background:#dfe8f3"></i>total</span><span><i style="background:${css(SPECIES_PALETTE[0])}"></i>species</span>`;
  }

  reset(): void { this.samples = []; this.lastStep = -1; }

  push(s: Stats): void {
    if (s.step - this.lastStep < 20) return;   // one sample per >= 20 sim steps
    this.lastStep = s.step;
    this.samples.push({ pop: s.population, species: s.speciesNames.map((n) => s.species[n] ?? 0) });
    if (this.samples.length > this.max) this.samples.shift();
  }

  draw(): void {
    const c = this.canvas, dpr = Math.min(devicePixelRatio, 2);
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const S = this.samples;
    if (S.length < 2) return;
    let top = 1;
    for (const s of S) top = Math.max(top, s.pop);
    top *= 1.12;
    const span = Math.max(S.length, 150) - 1;
    const x = (i: number) => (i / span) * w;
    const y = (v: number) => h - 4 - (v / top) * (h - 10);
    g.strokeStyle = 'rgba(140,180,220,0.08)';
    g.lineWidth = 1;
    for (let k = 1; k < 4; k++) { g.beginPath(); g.moveTo(0, (h * k) / 4); g.lineTo(w, (h * k) / 4); g.stroke(); }
    // total
    const grad = g.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(56,214,232,0.35)');
    grad.addColorStop(1, 'rgba(56,214,232,0)');
    g.beginPath();
    g.moveTo(x(0), h);
    S.forEach((s, i) => g.lineTo(x(i), y(s.pop)));
    g.lineTo(x(S.length - 1), h);
    g.fillStyle = grad;
    g.fill();
    g.beginPath();
    S.forEach((s, i) => (i ? g.lineTo(x(i), y(s.pop)) : g.moveTo(x(i), y(s.pop))));
    g.strokeStyle = '#dfe8f3';
    g.lineWidth = 1.5;
    g.stroke();
    // species
    const k = Math.max(...S.map((s) => s.species.length));
    for (let j = 0; j < k; j++) {
      g.beginPath();
      let started = false;
      S.forEach((s, i) => {
        const v = s.species[j];
        if (v === undefined) { started = false; return; }
        if (!started) { g.moveTo(x(i), y(v)); started = true; } else g.lineTo(x(i), y(v));
      });
      g.strokeStyle = css(SPECIES_PALETTE[j % SPECIES_PALETTE.length]);
      g.globalAlpha = 0.85;
      g.lineWidth = 1.1;
      g.stroke();
      g.globalAlpha = 1;
    }
    g.fillStyle = 'rgba(223,232,243,0.6)';
    g.font = '10px ui-monospace, monospace';
    g.fillText(Math.round(top / 1.12).toLocaleString(), 4, 11);
  }
}
