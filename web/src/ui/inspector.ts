/** Selected-particle card: body, energy, lineage info and a live brain diagram. */
import type { Inspect } from '../protocol';
import { SENSE_NAMES, OUTPUT_NAMES } from '../sim/genome';
import { lineageColor, css, TYPE_PALETTE } from '../palette';

const meter = (label: string, value: string, frac: number, color = 'var(--accent)') =>
  `<div class="meter"><span>${label}<b>${value}</b></span><div class="bar"><div style="width:${Math.max(0, Math.min(1, frac)) * 100}%;background:${color}"></div></div></div>`;

export class Inspector {
  private root: HTMLElement;
  private canvas: HTMLCanvasElement;
  onClose: () => void = () => {};

  constructor() {
    this.root = document.getElementById('inspector')!;
    this.canvas = document.getElementById('brain') as HTMLCanvasElement;
    document.getElementById('insClose')!.addEventListener('click', () => this.onClose());
  }

  hide(): void { this.root.hidden = true; }

  update(p: Inspect): void {
    this.root.hidden = false;
    const rgb: [number, number, number] = [0, 0, 0];
    lineageColor(p.color[0], p.color[1], p.color[2], rgb);
    const sw = document.getElementById('insSwatch')!;
    sw.style.background = css(rgb);
    sw.style.color = css(rgb);
    document.getElementById('insTitle')!.textContent = `#${p.uid} · ${p.species}`;
    const diet = p.preyGained / Math.max(1e-6, p.preyGained + p.foodGained);
    document.getElementById('insSub')!.innerHTML =
      `generation ${p.generation} · type <span style="color:${css(TYPE_PALETTE[p.ptype])}">${'ABCDEF'[p.ptype]}</span> · ${p.bonds} bond${p.bonds === 1 ? '' : 's'} · diet ${(diet * 100).toFixed(0)}% prey`;
    document.getElementById('insBody')!.innerHTML = [
      meter('Energy', `${p.energy.toFixed(2)} / ${p.threshold.toFixed(2)}`, p.energy / p.threshold, '#7ee787'),
      meter('Age', `${p.age} / ${p.maxAge}`, p.age / p.maxAge, '#8796ab'),
      meter('Size', p.size.toFixed(2), (p.size - 0.6) / 1.2, '#38d6e8'),
      meter('Speed', p.speed.toFixed(2), (p.speed - 0.5) / 1.5, '#a78bfa'),
      meter('Sense', p.sense.toFixed(2), (p.sense - 0.5) / 1.1, '#ffcf5e'),
      meter('Learning rate', `10^${p.logEta.toFixed(2)}`, (p.logEta + 4) / 3.5, '#ff9ecf'),
      meter('Aggression', p.aggression.toFixed(2), p.aggression, '#ff6b5e'),
      meter('Signal', p.signal.toFixed(2), Math.abs(p.signal), '#ffffff'),
    ].join('');
    this.drawBrain(p);
  }

  private drawBrain(p: Inspect): void {
    const c = this.canvas, dpr = Math.min(devicePixelRatio, 2);
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const NI = p.inputs.length, H = p.hidden.length, NO = p.outputs.length;
    const xIn = 64, xHid = w / 2 + 4, xOut = w - 62;
    const yAt = (i: number, n: number) => 8 + ((h - 16) * (i + 0.5)) / n;
    // edges: strongest only, so the picture stays readable
    const edges = (from: number[], fromN: number, toN: number, wts: number[], x0: number, x1: number, rowMajorTo: number) => {
      const list: [number, number, number][] = [];
      for (let i = 0; i < fromN; i++) for (let j = 0; j < toN; j++) {
        const wv = wts[i * rowMajorTo + j];
        const act = Math.abs(from[i]) * Math.abs(wv);
        list.push([i, j, wv * (0.35 + act)]);
      }
      list.sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]));
      for (const [i, j, v] of list.slice(0, 110)) {
        g.beginPath();
        g.moveTo(x0, yAt(i, fromN));
        g.bezierCurveTo((x0 + x1) / 2, yAt(i, fromN), (x0 + x1) / 2, yAt(j, toN), x1, yAt(j, toN));
        g.strokeStyle = v > 0 ? `rgba(56,214,232,${Math.min(0.75, Math.abs(v) * 0.6)})` : `rgba(255,107,94,${Math.min(0.75, Math.abs(v) * 0.6)})`;
        g.lineWidth = Math.min(2.2, 0.3 + Math.abs(v));
        g.stroke();
      }
    };
    edges(p.inputs, NI, H, p.w1, xIn, xHid, H);
    edges(p.hidden, H, NO, p.w2, xHid, xOut, NO);
    const node = (x: number, y: number, v: number, r: number) => {
      g.beginPath();
      g.arc(x, y, r, 0, Math.PI * 2);
      const a = Math.min(1, Math.abs(v));
      g.fillStyle = v >= 0 ? `rgba(56,214,232,${0.15 + 0.85 * a})` : `rgba(255,107,94,${0.15 + 0.85 * a})`;
      g.fill();
      g.strokeStyle = 'rgba(255,255,255,0.25)';
      g.lineWidth = 0.6;
      g.stroke();
    };
    g.font = '9px ui-monospace, monospace';
    g.textBaseline = 'middle';
    g.fillStyle = '#8796ab';
    for (let i = 0; i < NI; i++) {
      node(xIn, yAt(i, NI), p.inputs[i], 2.6);
      g.fillStyle = '#6a7a90';
      g.textAlign = 'right';
      g.fillText(SENSE_NAMES[i], xIn - 6, yAt(i, NI));
    }
    for (let i = 0; i < H; i++) node(xHid, yAt(i, H), p.hidden[i], 3.6);
    for (let i = 0; i < NO; i++) {
      node(xOut, yAt(i, NO), p.outputs[i], 4.4);
      g.fillStyle = '#b8c4d4';
      g.textAlign = 'left';
      g.fillText(OUTPUT_NAMES[i], xOut + 8, yAt(i, NO));
    }
  }
}
