/// <reference lib="webworker" />
/** Web Worker entry: forwards messages to the simulation host. */
import { handle, setPost } from './host';
import type { ToWorker } from './protocol';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
setPost((msg, transfer) => ctx.postMessage(msg, transfer));
ctx.onmessage = (ev: MessageEvent<ToWorker>) => handle(ev.data);
