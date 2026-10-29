/** Panel wiring: buttons, segmented controls, layer chips and live-parameter sliders. */
import type { ColorMode } from '../protocol';
import type { LiveParam } from '../sim/config';
import { PY_DEFAULTS } from '../sim/config';
import type { Layers } from '../render/view';

export type Tool = 'inspect' | 'food' | 'meteor';

export interface UiState {
  paused: boolean;
  speed: number;
  colorMode: ColorMode;
  tool: Tool;
  layers: Layers;
}

export interface UiHandlers {
  onReset(seed: number, scale: number): void;
  onStep(): void;
  onFit(): void;
  onParam(key: LiveParam, value: number): void;
  onLayers(): void;
}

const PARAMS: { key: LiveParam; label: string; min: number; max: number; step: number; fmt: (v: number) => string }[] = [
  { key: 'foodRegrowProb', label: 'Food regrowth', min: 0, max: 0.012, step: 0.0002, fmt: (v) => v.toFixed(4) },
  { key: 'attackRate', label: 'Attack rate', min: 0, max: 0.12, step: 0.002, fmt: (v) => v.toFixed(3) },
  { key: 'mutationRate', label: 'Mutation rate', min: 0, max: 0.4, step: 0.01, fmt: (v) => v.toFixed(2) },
  { key: 'climateAmplitude', label: 'Climate swing', min: 0, max: 1, step: 0.05, fmt: (v) => v.toFixed(2) },
];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export function setupControls(state: UiState, h: UiHandlers): { syncPlay(): void; setScale(scale: number): void; setSeed(seed: number): void; resetParams(): void } {
  const play = $<HTMLButtonElement>('play');
  const syncPlay = () => { play.textContent = state.paused ? 'Play' : 'Pause'; };
  play.addEventListener('click', () => { state.paused = !state.paused; syncPlay(); });
  $('stepOnce').addEventListener('click', () => h.onStep());
  $('fit').addEventListener('click', () => h.onFit());

  const speed = $<HTMLInputElement>('speed'), speedVal = $('speedVal');
  speed.value = String(state.speed);
  speedVal.textContent = String(state.speed);
  speed.addEventListener('input', () => { state.speed = Number(speed.value); speedVal.textContent = speed.value; });

  const seed = $<HTMLInputElement>('seed'), scale = $<HTMLSelectElement>('scale');
  $('reset').addEventListener('click', () => h.onReset(Math.max(0, Math.floor(Number(seed.value) || 0)), Number(scale.value)));
  scale.addEventListener('change', () => h.onReset(Math.max(0, Math.floor(Number(seed.value) || 0)), Number(scale.value)));

  segmented('colorMode', (v) => { state.colorMode = v as ColorMode; });
  segmented('tool', (v) => { state.tool = v as Tool; });

  const chips = $('layers');
  chips.querySelectorAll<HTMLButtonElement>('button').forEach((b) => {
    const key = b.dataset.l as keyof Layers;
    b.classList.toggle('on', state.layers[key]);
    b.addEventListener('click', () => {
      state.layers[key] = !state.layers[key];
      b.classList.toggle('on', state.layers[key]);
      h.onLayers();
    });
  });

  const params = $('params');
  const inputs: Partial<Record<LiveParam, HTMLInputElement>> = {};
  for (const p of PARAMS) {
    const label = document.createElement('label');
    label.className = 'slider';
    const def = PY_DEFAULTS[p.key];
    label.innerHTML = `<span>${p.label} <em>${p.fmt(def)}</em></span><input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${def}" />`;
    const input = label.querySelector('input')!, em = label.querySelector('em')!;
    input.addEventListener('input', () => { em.textContent = p.fmt(Number(input.value)); h.onParam(p.key, Number(input.value)); });
    inputs[p.key] = input;
    params.appendChild(label);
  }
  const resetParams = () => {
    for (const p of PARAMS) {
      const input = inputs[p.key]!;
      input.value = String(PY_DEFAULTS[p.key]);
      input.dispatchEvent(new Event('input'));
    }
  };

  // Panel collapse (starts collapsed on phones).
  const panel = $('panel'), toggle = $('panelToggle');
  const setOpen = (open: boolean) => { panel.classList.toggle('closed', !open); toggle.setAttribute('aria-expanded', String(open)); };
  setOpen(window.innerWidth > 760);
  toggle.addEventListener('click', () => setOpen(panel.classList.contains('closed')));

  return {
    syncPlay,
    setScale: (s) => { scale.value = String(s); },
    setSeed: (s) => { seed.value = String(s); },
    resetParams,
  };
}

/** Wire a `.seg` radio group; exposes programmatic selection via the element's `select` event. */
export function segmented(id: string, on: (v: string) => void): void {
  const root = $(id);
  const buttons = root.querySelectorAll<HTMLButtonElement>('button');
  const pick = (v: string) => {
    buttons.forEach((x) => x.classList.toggle('on', x.dataset.v === v));
    on(v);
  };
  buttons.forEach((b) => b.addEventListener('click', () => pick(b.dataset.v!)));
  root.addEventListener('select-value', (e) => pick((e as CustomEvent<string>).detail));
}

export function selectSegment(id: string, v: string): void {
  $(id).dispatchEvent(new CustomEvent('select-value', { detail: v }));
}
