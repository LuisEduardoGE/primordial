/** Top-left statistics card. */
import type { Stats } from '../sim/stats';
import { SPECIES_PALETTE, css } from '../palette';

const fmt = (v: number, d = 0) => v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export class Hud {
  private el: HTMLElement;

  constructor(el: HTMLElement) {
    this.el = el;
    el.addEventListener('click', () => el.classList.toggle('expanded'));
  }

  update(s: Stats, fps: number, stepsPerSec: number, paused: boolean): void {
    const floor = s.population <= s.minPopulation * 1.05;
    const total = Math.max(1, Object.values(s.species).reduce((a, b) => a + b, 0));
    const species = s.speciesNames
      .map((name, i) => ({ name, i, n: s.species[name] ?? 0 }))
      .filter((r) => r.n > 0)
      .sort((a, b) => b.n - a.n)
      .map((r) => {
        const c = css(SPECIES_PALETTE[r.i % SPECIES_PALETTE.length]);
        return `<div class="sp-row"><i style="background:${c}"></i><span>${r.name}</span><b>${fmt(r.n)}</b>
          <div class="sp-bar"><div style="width:${(100 * r.n) / total}%;background:${c}"></div></div></div>`;
      })
      .join('');
    const knob = ((s.season + 1) / 2) * 100;
    const seasonName = s.season > 0.33 ? 'bloom' : s.season < -0.33 ? 'famine' : 'temperate';
    this.el.innerHTML = `
      <div class="big"><b>${fmt(s.population)}</b><span>particles${floor ? ' · <em style="color:#ffb35e;font-style:normal">abiogenesis floor</em>' : ''}${paused ? ' · paused' : ''}</span></div>
      <div class="kv">
        <span>Generation avg/max</span><b>${fmt(s.meanGeneration, 1)} / ${fmt(s.maxGeneration)}</b>
        <span>Step</span><b>${fmt(s.step)}</b>
        <span>Food on map</span><b>${fmt(s.food)}</b>
        <span>Predation share</span><b>${pct(s.predationShare)}</b>
        <span>Bonded</span><b>${pct(s.bondedFraction)}</b>
        <span>Attacking now</span><b>${fmt(s.attackers)}</b>
        <span>Lineages</span><b>${s.lineages}</b>
        <span>Size·speed·sense</span><b>${s.meanSize.toFixed(2)} · ${s.meanSpeed.toFixed(2)} · ${s.meanSense.toFixed(2)}</b>
        <span>σ perc·behav·phys</span><b>${s.sigma[0].toFixed(2)} · ${s.sigma[1].toFixed(2)} · ${s.sigma[2].toFixed(2)}</b>
        <span>Births (sexual)</span><b>${fmt(s.births)} (${pct(s.matedBirths / Math.max(1, s.births))})</b>
        <span>FPS · steps/s</span><b>${fmt(fps)} · ${fmt(stepsPerSec)}</b>
      </div>
      <div class="season"><span>${seasonName}</span><div class="track"><div class="knob" style="left:${knob}%"></div></div></div>
      <div class="species">${species || '<span style="color:var(--faint)">species classified every 250 steps</span>'}</div>`;
  }
}
