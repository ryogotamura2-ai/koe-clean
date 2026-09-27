/* dfn3-worker.js — DeepFilterNet3 noise suppression in a Web Worker (module worker).
 *
 *   const w = new Worker(new URL('./dfn3-worker.js', import.meta.url), { type: 'module' });
 *   w.postMessage({ id, samples, attenLimDb }, [samples.buffer]);   // Float32Array, 48 kHz mono
 *   w.onmessage = ({ data }) => {
 *     data.type === 'ready'    -> { type, id, loadMs }                     (answer to {type:'init'})
 *     data.type === 'progress' -> { type, id, value }                      value in 0..1
 *     data.type === 'result'   -> { type, id, samples: Float32Array, stats } same length, time-aligned
 *     data.type === 'error'    -> { type, id, message }
 *   };
 *
 * Input message: { id?: any, samples: Float32Array (48 kHz mono), attenLimDb?: number|null,
 *                  chunkFrames?: number (default 250 = 2.5 s per model call; tests only) }
 *   attenLimDb: like DeepFilterNet's atten_lim_db — null/0 removes as much noise as the model
 *   wants; 12 keeps the noise at most 12 dB down by mixing back the noisy spectrum.
 * Optional { type: 'init', model?: ArrayBuffer|Uint8Array, wasmBinary?: ArrayBuffer } loads the
 *   runtime + model ahead of the first job. Pass the bytes when you run several workers: the page
 *   downloads the 14 MB runtime + 8.6 MB model once (with its own progress bar) and every worker
 *   reuses them, instead of each worker fetching the URLs itself (the default).
 * Jobs sent to one worker run one after another; run several workers for parallelism.
 *
 * Runs single-threaded (no SharedArrayBuffer / cross-origin isolation needed) with WASM SIMD.
 * Every runtime file is loaded relative to this script: vendor/ort/*, models/dfn3_stateful.onnx.
 */
import * as ort from './vendor/ort/ort.wasm.min.mjs';
import { enhance, SR } from './dfn3-core.js';

const MODEL_URL = new URL('./models/dfn3_stateful.onnx', import.meta.url).href;
ort.env.wasm.numThreads = 1; // GitHub Pages cannot send COOP/COEP -> no wasm threads
ort.env.wasm.proxy = false; // we already are in a worker
ort.env.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;

let sessionPromise = null;
let loadMs = 0;
function getSession(model) {
  if (!sessionPromise) {
    const t0 = performance.now();
    const src = model ? (model instanceof Uint8Array ? model : new Uint8Array(model)) : MODEL_URL;
    sessionPromise = ort.InferenceSession.create(src, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }).then((s) => { loadMs = performance.now() - t0; return s; });
    sessionPromise.catch(() => { sessionPromise = null; }); // allow a retry after e.g. a network error
  }
  return sessionPromise;
}

async function runJob(msg) {
  const { id = null, samples, attenLimDb = null, chunkFrames = 250 } = msg;
  if (!(samples instanceof Float32Array)) throw new Error('samples must be a Float32Array (48 kHz mono)');
  if (attenLimDb != null && !Number.isFinite(attenLimDb)) throw new Error('attenLimDb must be a number or null');
  const session = await getSession();
  const t0 = performance.now();
  let lastPost = 0;
  const out = await enhance({
    ort, session, samples, attenLimDb, chunkFrames,
    onProgress(v) {
      const now = performance.now();
      if (v >= 1 || now - lastPost > 100) { lastPost = now; postMessage({ type: 'progress', id, value: v }); }
    },
  });
  const procMs = performance.now() - t0;
  const stats = {
    procMs, loadMs, seconds: samples.length / SR, rtf: procMs / 1000 / (samples.length / SR || 1),
    crossOriginIsolated: self.crossOriginIsolated === true, ortVersion: ort.env.versions?.web,
    jsHeapBytes: performance.memory ? performance.memory.usedJSHeapSize : null,
  };
  postMessage({ type: 'result', id, samples: out, stats }, [out.buffer]);
}

let queue = Promise.resolve();
self.onmessage = (ev) => {
  const msg = ev.data || {};
  if (msg.type === 'init') {
    // wasmBinary must be set before the runtime initialises (first InferenceSession.create)
    if (msg.wasmBinary && !sessionPromise) ort.env.wasm.wasmBinary = msg.wasmBinary;
    queue = queue.then(() => getSession(msg.model))
      .then(() => postMessage({ type: 'ready', id: msg.id ?? null, loadMs }))
      .catch((e) => postMessage({ type: 'error', id: msg.id ?? null, message: String(e?.message || e) }));
    return;
  }
  queue = queue.then(() => runJob(msg))
    .catch((e) => postMessage({ type: 'error', id: msg.id ?? null, message: String(e?.message || e) }));
};
