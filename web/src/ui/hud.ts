/** Statistics panel. Every row carries a plain-language tooltip. */
import type { Stats } from '../sim/stats';
import { SPECIES_PALETTE, css } from '../palette';

const fmt = (v: number, d = 0) => v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

const TIPS = {
  population: 'Number of particles currently alive.',
  generation: 'Generation counts how many ancestors a particle has since its line was spawned at random. Shown as the population average and the current maximum.',
  step: 'Simulation time steps since the start of this run.',
  food: 'Food pellets currently lying on the map, waiting to be eaten.',
  predation: 'Fraction of all energy the living particles have gained that came from eating other particles rather than food.',
  bonded: 'Fraction of particles physically linked to at least one relative by a bond, forming multi-particle organisms that share energy.',
  attacking: 'Particles whose attack output is currently strong (above 0.3). Attacks drain energy from touching, unrelated, smaller-or-similar particles.',
  lineages: 'Effective number of families: the exponential of the Shannon entropy of colour families. 1 means one family dominates; higher means more families of similar size coexist.',
  body: 'Average evolved body traits. Size: energy capacity and attack strength. Speed: thrust. Sense: how far a particle can perceive. All cost energy to maintain.',
  sigma: 'Average mutation step size for the perception layer, behaviour layer and attraction forces. These step sizes evolve themselves: small values mean offspring differ little from parents.',
  births: 'Total births in this run, and the share produced by sexual reproduction (genes mixed from two parents) rather than cloning.',
  climate: 'Seasonal cycle. Food grows faster or slower than normal by the factor shown; the cycle repeats every 8,000 steps.',
  speed: 'Display frames per second, and simulation steps computed per second on this device.',
  species: 'Behavioural species: particles grouped (k-means clustering) by body traits and recent behaviour, recomputed every 250 steps. Names describe the dominant behaviour of each group.',
  floor: 'The population has dropped to the minimum, so new random particles are being spawned to keep the world populated.',
};

export class Hud {
  private el: HTMLElement;

  constructor(el: HTMLElement) {
    this.el = el;
    el.addEventListener('click', () => el.classList.toggle('expanded'));
  }

  update(s: Stats, fps: number, stepsPerSec: number, paused: boolean, climateAmplitude: number): void {
    const floor = s.population <= s.minPopulation * 1.05;
    const species = s.speciesNames
      .map((name, i) => ({ name, i, n: s.species[name] ?? 0 }))
      .filter((r) => r.n > 0)
      .sort((a, b) => b.n - a.n)
      .map((r) => `<div class="sp-row"><i style="background:${css(SPECIES_PALETTE[r.i % SPECIES_PALETTE.length])}"></i><span>${r.name}</span><b>${fmt(r.n)}</b></div>`)
      .join('');
    const food = 1 + climateAmplitude * s.season;
    const row = (label: string, value: string, tip: string) => `<tr title="${tip}"><td>${label}</td><td>${value}</td></tr>`;
    this.el.innerHTML = `
      <div class="big" title="${TIPS.population}"><b>${fmt(s.population)}</b><span>particles${paused ? ' (paused)' : ''}</span><span class="expand">Details</span></div>
      ${floor ? `<div class="note" title="${TIPS.floor}">At minimum population: random particles are being added.</div>` : ''}
      <table>
        ${row('Generation (mean / max)', `${fmt(s.meanGeneration, 1)} / ${fmt(s.maxGeneration)}`, TIPS.generation)}
        ${row('Time step', fmt(s.step), TIPS.step)}
        ${row('Food items', fmt(s.food), TIPS.food)}
        ${row('Predation share', pct(s.predationShare), TIPS.predation)}
        ${row('Bonded', pct(s.bondedFraction), TIPS.bonded)}
        ${row('Attacking', fmt(s.attackers), TIPS.attacking)}
        ${row('Effective lineages', fmt(s.lineages, 1), TIPS.lineages)}
        ${row('Size / speed / sense', `${s.meanSize.toFixed(2)} / ${s.meanSpeed.toFixed(2)} / ${s.meanSense.toFixed(2)}`, TIPS.body)}
        ${row('Mutation step sizes', `${s.sigma[0].toFixed(2)} / ${s.sigma[1].toFixed(2)} / ${s.sigma[2].toFixed(2)}`, TIPS.sigma)}
        ${row('Births (sexual)', `${fmt(s.births)} (${pct(s.matedBirths / Math.max(1, s.births))})`, TIPS.births)}
        ${row('Season: food growth', `×${food.toFixed(2)}`, TIPS.climate)}
        ${row('Frame rate / sim speed', `${fmt(fps)} fps / ${fmt(stepsPerSec)} steps/s`, TIPS.speed)}
      </table>
      <div class="species" title="${TIPS.species}"><h3>Behavioural species</h3>${species || '<span style="color:var(--faint)">First classification at step 250.</span>'}</div>`;
  }
}
