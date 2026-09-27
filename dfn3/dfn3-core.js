/* dfn3-core.js — DeepFilterNet3 offline enhancement (48 kHz mono), pure ES module.
 *
 * Port of what the Python reference `df.enhance.enhance(model, df_state, audio, pad=True,
 * atten_lim_db)` does, around the model file models/dfn3_stateful.onnx (the official DFN3 ONNX
 * export merged into one graph whose GRU states / causal-conv contexts are explicit inputs and
 * outputs, see tools/make_stateful_onnx.py). Because the state is carried from chunk to chunk, the
 * chunked result is identical to one pass over the whole file, while memory stays bounded.
 *
 * DSP ported from libDF (Rikorose/DeepFilterNet, libDF/src/lib.rs + transforms.rs, MIT/Apache-2.0):
 *   STFT: Vorbis window, fft 960, hop 480, forward scaled by 1/960, inverse unscaled (realfft)
 *   ERB feature: band power mean over 32 ERB bands -> 10*log10(x+1e-10) -> exp. mean norm / 40
 *   complex feature: lowest 96 bins, exponential unit norm; alpha = 0.99 (norm_tau 1 s)
 *   model sees features shifted by conv_lookahead = 2 frames (pad_feat)
 *   ERB gains applied per band (bins >= 96), deep filter order 5 / lookahead 2 on bins < 96
 *   atten_lim: enhanced = noisy * lim + enhanced * (1 - lim), lim = 10^(-|dB|/20)
 *   delay compensation: pad input with 960 zeros, drop the first 480 output samples.
 * No dependencies; `ort` (onnxruntime-web) and the session are passed in.
 */

export const SR = 48000;
const N_FFT = 960;
const HOP = 480;
const N_FREQ = N_FFT / 2 + 1; // 481
const NB_ERB = 32;
const NB_DF = 96;
const DF_ORDER = 5;
const DF_LOOKAHEAD = 2;
const CONV_LOOKAHEAD = 2;
const ALPHA = Math.fround(0.99); // get_norm_alpha(): exp(-hop/sr/tau) rounded -> 0.99
const WNORM = Math.fround(1 / ((N_FFT * N_FFT) / (2 * HOP))); // 1/960
// libDF erb_fb(48000, 960, nb_bands=32, min_nb_erb_freqs=2) (checked against libdf.DF.erb_widths()).
const ERB_WIDTHS = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67];

// State tensors of models/dfn3_stateful.onnx: [name, dims]. Zeros at the start of every file.
const STATES = [
  ['erb_ctx', [1, 1, 2, NB_ERB]],
  ['spec_ctx', [1, 2, 2, NB_DF]],
  ['c0_ctx', [1, 64, 4, NB_DF]],
  ['h_enc', [1, 1, 256]],
  ['h_erb0', [1, 1, 256]],
  ['h_erb1', [1, 1, 256]],
  ['h_df0', [1, 1, 256]],
  ['h_df1', [1, 1, 256]],
];

/* Vorbis window: sin(pi/2 * sin^2(pi*(n+0.5)/N)), f64 then rounded to f32 like libDF. */
const WINDOW = new Float32Array(N_FFT);
for (let i = 0; i < N_FFT; i++) {
  const s = Math.sin((0.5 * Math.PI * (i + 0.5)) / (N_FFT / 2));
  WINDOW[i] = Math.sin(0.5 * Math.PI * s * s);
}
const ERB_BIN_BAND = new Uint8Array(N_FREQ); // band index of each frequency bin
{
  let k = 0;
  ERB_WIDTHS.forEach((w, b) => { for (let j = 0; j < w; j++) ERB_BIN_BAND[k++] = b; });
  if (k !== N_FREQ) throw new Error('ERB widths do not cover 481 bins');
}

/* ---------- complex FFT (mixed radix Stockham autosort, DIF) ---------- */
class ComplexFFT {
  constructor(n, radices) {
    this.n = n;
    this.stages = [];
    let len = n;
    let stride = 1;
    for (const r of radices) {
      const m = len / r;
      if (!Number.isInteger(m)) throw new Error('bad radices');
      const twr = new Float64Array(m * r);
      const twi = new Float64Array(m * r);
      for (let p = 0; p < m; p++) {
        for (let k = 0; k < r; k++) {
          const a = (-2 * Math.PI * p * k) / len;
          twr[p * r + k] = Math.cos(a);
          twi[p * r + k] = Math.sin(a);
        }
      }
      const wr = new Float64Array(r * r);
      const wi = new Float64Array(r * r);
      for (let j = 0; j < r; j++) {
        for (let k = 0; k < r; k++) {
          const a = (-2 * Math.PI * ((j * k) % r)) / r;
          wr[j * r + k] = Math.cos(a);
          wi[j * r + k] = Math.sin(a);
        }
      }
      this.stages.push({ s: stride, r, m, twr, twi, wr, wi });
      len = m;
      stride *= r;
    }
    if (len !== 1) throw new Error('radices do not multiply to n');
    this.tr = new Float64Array(n);
    this.ti = new Float64Array(n);
    this.gr = new Float64Array(8);
    this.gi = new Float64Array(8);
  }

  /** In-place forward DFT (sign -1, unscaled) of (re, im). */
  forward(re, im) {
    let ar = re;
    let ai = im;
    let br = this.tr;
    let bi = this.ti;
    for (const st of this.stages) {
      if (st.r === 4) stage4(st, ar, ai, br, bi);
      else if (st.r === 2) stage2(st, ar, ai, br, bi);
      else if (st.r === 3) stage3(st, ar, ai, br, bi);
      else if (st.r === 5) stage5(st, ar, ai, br, bi);
      else this.stageGeneric(st, ar, ai, br, bi);
      const xr = ar; ar = br; br = xr;
      const xi = ai; ai = bi; bi = xi;
    }
    if (ar !== re) {
      re.set(ar);
      im.set(ai);
    }
  }

  stageGeneric({ s, r, m, twr, twi, wr, wi }, ar, ai, br, bi) {
    const gr = this.gr;
    const gi = this.gi;
    const sm = s * m;
    for (let p = 0; p < m; p++) {
      for (let q = 0; q < s; q++) {
        const base = q + s * p;
        for (let j = 0; j < r; j++) {
          gr[j] = ar[base + j * sm];
          gi[j] = ai[base + j * sm];
        }
        const ob = q + s * r * p;
        for (let k = 0; k < r; k++) {
          let sr = 0;
          let si = 0;
          for (let j = 0; j < r; j++) {
            const c = wr[j * r + k];
            const d = wi[j * r + k];
            sr += gr[j] * c - gi[j] * d;
            si += gr[j] * d + gi[j] * c;
          }
          const c = twr[p * r + k];
          const d = twi[p * r + k];
          br[ob + s * k] = sr * c - si * d;
          bi[ob + s * k] = sr * d + si * c;
        }
      }
    }
  }
}

/* Specialised Stockham DIF butterflies: inputs x[q + s(p + j m)], outputs
 * y[q + s(r p + k)] = (sum_j x_j W_r^{jk}) * W_len^{pk}. */
function stage2({ s, m, twr, twi }, ar, ai, br, bi) {
  const sm = s * m;
  for (let p = 0; p < m; p++) {
    const c1 = twr[p * 2 + 1];
    const d1 = twi[p * 2 + 1];
    for (let q = 0; q < s; q++) {
      const i0 = q + s * p;
      const i1 = i0 + sm;
      const o = q + 2 * s * p;
      const x0r = ar[i0]; const x0i = ai[i0]; const x1r = ar[i1]; const x1i = ai[i1];
      br[o] = x0r + x1r; bi[o] = x0i + x1i;
      const yr = x0r - x1r; const yi = x0i - x1i;
      br[o + s] = yr * c1 - yi * d1; bi[o + s] = yr * d1 + yi * c1;
    }
  }
}
function stage3({ s, m, twr, twi }, ar, ai, br, bi) {
  const sm = s * m;
  const h = Math.sqrt(3) / 2;
  for (let p = 0; p < m; p++) {
    const c1 = twr[p * 3 + 1]; const d1 = twi[p * 3 + 1];
    const c2 = twr[p * 3 + 2]; const d2 = twi[p * 3 + 2];
    for (let q = 0; q < s; q++) {
      const i0 = q + s * p;
      const i1 = i0 + sm;
      const i2 = i1 + sm;
      const o = q + 3 * s * p;
      const a0r = ar[i0]; const a0i = ai[i0];
      const t1r = ar[i1] + ar[i2]; const t1i = ai[i1] + ai[i2];
      const t2r = ar[i1] - ar[i2]; const t2i = ai[i1] - ai[i2];
      br[o] = a0r + t1r; bi[o] = a0i + t1i;
      const m1r = a0r - 0.5 * t1r; const m1i = a0i - 0.5 * t1i;
      const m2r = h * t2i; const m2i = -h * t2r; // -i*h*t2
      const y1r = m1r + m2r; const y1i = m1i + m2i;
      const y2r = m1r - m2r; const y2i = m1i - m2i;
      br[o + s] = y1r * c1 - y1i * d1; bi[o + s] = y1r * d1 + y1i * c1;
      br[o + 2 * s] = y2r * c2 - y2i * d2; bi[o + 2 * s] = y2r * d2 + y2i * c2;
    }
  }
}
function stage4({ s, m, twr, twi }, ar, ai, br, bi) {
  const sm = s * m;
  for (let p = 0; p < m; p++) {
    const c1 = twr[p * 4 + 1]; const d1 = twi[p * 4 + 1];
    const c2 = twr[p * 4 + 2]; const d2 = twi[p * 4 + 2];
    const c3 = twr[p * 4 + 3]; const d3 = twi[p * 4 + 3];
    for (let q = 0; q < s; q++) {
      const i0 = q + s * p;
      const i1 = i0 + sm;
      const i2 = i1 + sm;
      const i3 = i2 + sm;
      const o = q + 4 * s * p;
      const t0r = ar[i0] + ar[i2]; const t0i = ai[i0] + ai[i2];
      const t1r = ar[i0] - ar[i2]; const t1i = ai[i0] - ai[i2];
      const t2r = ar[i1] + ar[i3]; const t2i = ai[i1] + ai[i3];
      const t3r = ai[i1] - ai[i3]; const t3i = ar[i3] - ar[i1]; // -i*(x1-x3)
      br[o] = t0r + t2r; bi[o] = t0i + t2i;
      const y1r = t1r + t3r; const y1i = t1i + t3i;
      const y2r = t0r - t2r; const y2i = t0i - t2i;
      const y3r = t1r - t3r; const y3i = t1i - t3i;
      br[o + s] = y1r * c1 - y1i * d1; bi[o + s] = y1r * d1 + y1i * c1;
      br[o + 2 * s] = y2r * c2 - y2i * d2; bi[o + 2 * s] = y2r * d2 + y2i * c2;
      br[o + 3 * s] = y3r * c3 - y3i * d3; bi[o + 3 * s] = y3r * d3 + y3i * c3;
    }
  }
}
const C51 = Math.cos((2 * Math.PI) / 5);
const C52 = Math.cos((4 * Math.PI) / 5);
const S51 = Math.sin((2 * Math.PI) / 5);
const S52 = Math.sin((4 * Math.PI) / 5);
function stage5({ s, m, twr, twi }, ar, ai, br, bi) {
  const sm = s * m;
  for (let p = 0; p < m; p++) {
    const tb = p * 5;
    for (let q = 0; q < s; q++) {
      const i0 = q + s * p;
      const i1 = i0 + sm;
      const i2 = i1 + sm;
      const i3 = i2 + sm;
      const i4 = i3 + sm;
      const o = q + 5 * s * p;
      const a0r = ar[i0]; const a0i = ai[i0];
      const b1r = ar[i1] + ar[i4]; const b1i = ai[i1] + ai[i4];
      const b2r = ar[i2] + ar[i3]; const b2i = ai[i2] + ai[i3];
      const d1r = ar[i1] - ar[i4]; const d1i = ai[i1] - ai[i4];
      const d2r = ar[i2] - ar[i3]; const d2i = ai[i2] - ai[i3];
      br[o] = a0r + b1r + b2r; bi[o] = a0i + b1i + b2i;
      const e1r = a0r + C51 * b1r + C52 * b2r; const e1i = a0i + C51 * b1i + C52 * b2i;
      const e2r = a0r + C52 * b1r + C51 * b2r; const e2i = a0i + C52 * b1i + C51 * b2i;
      const f1r = S51 * d1r + S52 * d2r; const f1i = S51 * d1i + S52 * d2i; // X1 = e1 - i f1, X4 = e1 + i f1
      const f2r = S52 * d1r - S51 * d2r; const f2i = S52 * d1i - S51 * d2i; // X2 = e2 - i f2, X3 = e2 + i f2
      let yr = e1r + f1i; let yi = e1i - f1r; let c = twr[tb + 1]; let d = twi[tb + 1];
      br[o + s] = yr * c - yi * d; bi[o + s] = yr * d + yi * c;
      yr = e2r + f2i; yi = e2i - f2r; c = twr[tb + 2]; d = twi[tb + 2];
      br[o + 2 * s] = yr * c - yi * d; bi[o + 2 * s] = yr * d + yi * c;
      yr = e2r - f2i; yi = e2i + f2r; c = twr[tb + 3]; d = twi[tb + 3];
      br[o + 3 * s] = yr * c - yi * d; bi[o + 3 * s] = yr * d + yi * c;
      yr = e1r - f1i; yi = e1i + f1r; c = twr[tb + 4]; d = twi[tb + 4];
      br[o + 4 * s] = yr * c - yi * d; bi[o + 4 * s] = yr * d + yi * c;
    }
  }
}

/* ---------- real FFT of size 960 via a complex FFT of size 480 ---------- */
const HALF = N_FFT / 2; // 480
const FFT480 = new ComplexFFT(HALF, [4, 4, 2, 3, 5]);
const RC = new Float64Array(HALF + 1); // cos(2*pi*k/960)
const RS = new Float64Array(HALF + 1); // sin(2*pi*k/960)
for (let k = 0; k <= HALF; k++) {
  RC[k] = Math.cos((2 * Math.PI * k) / N_FFT);
  RS[k] = Math.sin((2 * Math.PI * k) / N_FFT);
}
const zr = new Float64Array(HALF);
const zi = new Float64Array(HALF);

/** Windowed analysis of one frame: frame t covers input samples [480(t-1), 480(t+1)).
 *  Writes 481 complex bins (scaled by 1/960) to outRe/outIm at offset `o`. */
function analysisFrame(x, t, outRe, outIm, o) {
  const base = HOP * (t - 1);
  const L = x.length;
  for (let n = 0; n < HALF; n++) {
    const i0 = base + 2 * n;
    const i1 = i0 + 1;
    zr[n] = (i0 >= 0 && i0 < L ? x[i0] : 0) * WINDOW[2 * n];
    zi[n] = (i1 >= 0 && i1 < L ? x[i1] : 0) * WINDOW[2 * n + 1];
  }
  FFT480.forward(zr, zi);
  for (let k = 0; k <= HALF; k++) {
    const a = k % HALF;
    const b = (HALF - k) % HALF;
    const fer = 0.5 * (zr[a] + zr[b]);
    const fei = 0.5 * (zi[a] - zi[b]);
    const for_ = 0.5 * (zi[a] + zi[b]);
    const foi = -0.5 * (zr[a] - zr[b]);
    const c = RC[k];
    const s = -RS[k];
    outRe[o + k] = (fer + (c * for_ - s * foi)) * WNORM;
    outIm[o + k] = (fei + (c * foi + s * for_)) * WNORM;
  }
}

const synth = new Float64Array(N_FFT);
/** Unscaled inverse real FFT (like realfft; imaginary parts of DC and Nyquist are ignored),
 *  then windowed. Result in `synth` (960 samples). */
function synthesisFrame(re, im, o) {
  const x0 = re[o];
  const xn = re[o + HALF];
  for (let k = 0; k < HALF; k++) {
    const ar = k === 0 ? x0 : re[o + k];
    const ai = k === 0 ? 0 : im[o + k];
    const kk = HALF - k; // B = conj(X[480-k])
    const br = kk === HALF ? xn : re[o + kk];
    const bi = kk === HALF ? 0 : -im[o + kk];
    const fer = ar + br;
    const fei = ai + bi;
    const dr = ar - br;
    const di = ai - bi;
    const c = RC[k];
    const s = RS[k]; // W^-k = c + i s
    const for_ = dr * c - di * s;
    const foi = dr * s + di * c;
    // inverse via forward FFT of the conjugate: store conj(Z)
    zr[k] = fer - foi;
    zi[k] = -(fei + for_);
  }
  FFT480.forward(zr, zi);
  for (let n = 0; n < HALF; n++) {
    synth[2 * n] = zr[n] * WINDOW[2 * n];
    synth[2 * n + 1] = -zi[n] * WINDOW[2 * n + 1];
  }
}

function linspace(a, b, n) {
  const out = new Float64Array(n);
  const step = Math.fround((b - a) / (n - 1));
  for (let i = 0; i < n; i++) out[i] = Math.fround(a + i * step);
  return out;
}

/**
 * Enhance a 48 kHz mono signal. Returns a Float32Array of the same length, time-aligned.
 * @param {object} p
 * @param {object} p.ort              onnxruntime-web module (for ort.Tensor)
 * @param {object} p.session          InferenceSession of models/dfn3_stateful.onnx
 * @param {Float32Array} p.samples    input, 48 kHz mono
 * @param {number|null} [p.attenLimDb] null/0 = unlimited (DeepFilterNet atten_lim_db)
 * @param {number} [p.chunkFrames]    model frames per session.run (10 ms each)
 * @param {(v:number)=>void} [p.onProgress]
 */
export async function enhance({ ort, session, samples, attenLimDb = null, chunkFrames = 250, onProgress }) {
  const x = samples;
  const L = x.length;
  const out = new Float32Array(L);
  if (L === 0) return out;
  const T = Math.floor((L + N_FFT) / HOP); // frames of the zero-padded input (pad=True)
  const C = Math.max(1, Math.min(chunkFrames | 0, T));
  const lim = attenLimDb != null && Math.abs(attenLimDb) > 0 ? Math.pow(10, -Math.abs(attenLimDb) / 20) : 0;

  // spec buffer for frames [s-2, e+2)
  const SPAN = C + DF_ORDER - 1;
  const specRe = new Float32Array(SPAN * N_FREQ);
  const specIm = new Float32Array(SPAN * N_FREQ);
  const enhRe = new Float32Array(N_FREQ);
  const enhIm = new Float32Array(N_FREQ);
  const erbState = linspace(-60, -90, NB_ERB); // MEAN_NORM_INIT
  const unitState = linspace(0.001, 0.0001, NB_DF); // UNIT_NORM_INIT
  const bandPow = new Float64Array(NB_ERB);
  const invW = ERB_WIDTHS.map((w) => Math.fround(1 / w));
  const olaMem = new Float64Array(HOP);
  let featNext = 0; // next feature frame whose normalisation state has to be advanced

  let state = {};
  for (const [name, dims] of STATES) {
    state[name] = new ort.Tensor('float32', new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
  }

  for (let s = 0; s < T; s += C) {
    const e = Math.min(T, s + C);
    const n = e - s;
    const s0 = s - DF_LOOKAHEAD; // first frame held in spec buffer
    const sEnd = Math.min(T, e + DF_LOOKAHEAD); // exclusive
    // 1) STFT of frames [s-2, e+2) (frames outside [0, T) are zero)
    for (let t = s0; t < s0 + SPAN; t++) {
      const o = (t - s0) * N_FREQ;
      if (t < 0 || t >= sEnd) {
        specRe.fill(0, o, o + N_FREQ);
        specIm.fill(0, o, o + N_FREQ);
      } else {
        analysisFrame(x, t, specRe, specIm, o);
      }
    }
    // 2) features; model frame t sees feature frame t + conv_lookahead
    const featErb = new Float32Array(n * NB_ERB);
    const featSpec = new Float32Array(2 * n * NB_DF);
    for (let u = featNext; u < sEnd; u++) {
      const o = (u - s0) * N_FREQ;
      bandPow.fill(0);
      for (let f = 0; f < N_FREQ; f++) {
        const re = specRe[o + f];
        const im = specIm[o + f];
        bandPow[ERB_BIN_BAND[f]] += (re * re + im * im) * invW[ERB_BIN_BAND[f]];
      }
      const t = u - CONV_LOOKAHEAD - s; // row in this chunk's model input (may be < 0 for u = 0, 1)
      for (let b = 0; b < NB_ERB; b++) {
        const v = Math.log10(bandPow[b] + 1e-10) * 10;
        erbState[b] = v * (1 - ALPHA) + erbState[b] * ALPHA;
        if (t >= 0) featErb[t * NB_ERB + b] = (v - erbState[b]) / 40;
      }
      for (let f = 0; f < NB_DF; f++) {
        const re = specRe[o + f];
        const im = specIm[o + f];
        unitState[f] = Math.hypot(re, im) * (1 - ALPHA) + unitState[f] * ALPHA;
        if (t >= 0) {
          const k = 1 / Math.sqrt(unitState[f]);
          featSpec[t * NB_DF + f] = re * k;
          featSpec[(n + t) * NB_DF + f] = im * k;
        }
      }
    }
    featNext = sEnd;

    // 3) model
    const feeds = {
      feat_erb: new ort.Tensor('float32', featErb, [1, 1, n, NB_ERB]),
      feat_spec: new ort.Tensor('float32', featSpec, [1, 2, n, NB_DF]),
      ...state,
    };
    const res = await session.run(feeds);
    const m = res.m.data; // [1,1,n,32]
    const coefs = res.coefs.data; // [1,n,96,5*2]
    const next = {};
    for (const [name] of STATES) next[name] = res[name + '_out'];
    for (const k of Object.keys(feeds)) if (!(k in next) && feeds[k].dispose) feeds[k].dispose();
    state = next;

    // 4) ERB gains (bins >= 96), deep filter (bins < 96), atten limit, ISTFT + overlap-add
    for (let t = s; t < e; t++) {
      const r = t - s;
      const oc = (t - s0) * N_FREQ; // centre frame in spec buffer
      for (let f = NB_DF; f < N_FREQ; f++) {
        const g = m[r * NB_ERB + ERB_BIN_BAND[f]];
        enhRe[f] = specRe[oc + f] * g;
        enhIm[f] = specIm[oc + f] * g;
      }
      const cb = r * NB_DF * DF_ORDER * 2;
      for (let f = 0; f < NB_DF; f++) {
        let yr = 0;
        let yi = 0;
        for (let k = 0; k < DF_ORDER; k++) {
          const os = (t - DF_LOOKAHEAD + k - s0) * N_FREQ + f; // spec frame t-2+k
          const sr = specRe[os];
          const si = specIm[os];
          const ci = cb + (f * DF_ORDER + k) * 2;
          const cr = coefs[ci];
          const cim = coefs[ci + 1];
          yr += sr * cr - si * cim;
          yi += sr * cim + si * cr;
        }
        enhRe[f] = yr;
        enhIm[f] = yi;
      }
      if (lim > 0) {
        for (let f = 0; f < N_FREQ; f++) {
          enhRe[f] = specRe[oc + f] * lim + enhRe[f] * (1 - lim);
          enhIm[f] = specIm[oc + f] * lim + enhIm[f] * (1 - lim);
        }
      }
      synthesisFrame(enhRe, enhIm, 0);
      // synthesis output of frame t is samples [480t, 480t+480) of the padded signal;
      // delay compensation (n_fft - hop = 480) maps it to out[480(t-1) ...]
      const ob = HOP * (t - 1);
      for (let i = 0; i < HOP; i++) {
        const j = ob + i;
        if (j >= 0 && j < L) out[j] = synth[i] + olaMem[i];
        olaMem[i] = synth[HOP + i];
      }
    }
    for (const k of ['m', 'coefs']) if (res[k].dispose) res[k].dispose();
    if (onProgress) onProgress(e / T);
  }
  for (const k of Object.keys(state)) if (state[k].dispose) state[k].dispose();
  return out;
}

// exported for unit tests
export const _test = { analysisFrame, synthesisFrame, synth, ComplexFFT, WINDOW, ERB_WIDTHS };
