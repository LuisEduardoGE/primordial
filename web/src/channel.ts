/**
 * Transport between the UI and the simulation host. Normally a Web Worker; falls back to
 * running the host on the main thread when worker threads are badly starved by the OS
 * (measured at startup), or when forced with ?thread=main.
 */
import SimWorker from './worker?worker&inline';
import { handle, setPost } from './host';
import { bench } from './bench';
import type { ToMain, ToWorker } from './protocol';

export interface SimChannel {
  mode: 'worker' | 'main';
  workerSpeed: number;   // relative to the main thread (1 = same), NaN if not measured
  send(m: ToWorker, transfer?: Transferable[]): void;
  onmessage: (m: ToMain) => void;
}

export async function createChannel(pref: string | null): Promise<SimChannel> {
  if (pref === 'main') return mainChannel(NaN);
  const worker = new SimWorker();
  let ratio = NaN;
  if (pref !== 'worker') {
    const workerMmacs = await new Promise<number>((resolve) => {
      worker.onmessage = (ev: MessageEvent<ToMain>) => { if (ev.data.type === 'bench') resolve(ev.data.mmacs); };
      worker.postMessage({ type: 'bench' } satisfies ToWorker);
      setTimeout(() => resolve(-1), 4000);
    });
    const mainMmacs = bench();
    ratio = workerMmacs > 0 ? workerMmacs / mainMmacs : 0;
    if (ratio < 0.45) {
      worker.terminate();
      return mainChannel(ratio);
    }
  }
  const ch: SimChannel = {
    mode: 'worker',
    workerSpeed: ratio,
    send: (m, transfer = []) => worker.postMessage(m, transfer),
    onmessage: () => {},
  };
  worker.onmessage = (ev: MessageEvent<ToMain>) => ch.onmessage(ev.data);
  return ch;
}

function mainChannel(ratio: number): SimChannel {
  const ch: SimChannel = {
    mode: 'main',
    workerSpeed: ratio,
    send: (m) => { setTimeout(() => handle(m), 0); },
    onmessage: () => {},
  };
  setPost((msg) => ch.onmessage(msg));
  return ch;
}
