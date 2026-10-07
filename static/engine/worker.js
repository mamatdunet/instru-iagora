// Stable Audio 3.0 Small Music, run entirely in the browser with ONNX Runtime Web.
// Pipeline: tokenizer → text encoder (T5Gemma) → duration embedder → diffusion transformer (8 "ping-pong" steps)
// → audio decoder. Ported from stable-audio-tools (generate_diffusion_cond_inpaint + sample_flow_pingpong).
import { loadTokenizer } from './tokenizer.js';

// Two builds of ONNX Runtime: only the processor build has the 4-bit embedding operator (GatherBlockQuantized)
// for the processor; the graphics-card build has it for the graphics card.
const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
const ORT_BUILDS = { webgpu: 'ort.webgpu.bundle.min.mjs', wasm: 'ort.wasm.bundle.min.mjs' };
let ort = null;

const MODEL_BASE = 'https://huggingface.co/lsb/stable-audio-3-small-music-onnx/resolve/d523962dc41a9c632a4928ff9e02538bb11f1806';
const STORE_NAME = 'instru-iagora-modele-v1';
const OLD_CACHE_NAME = 'instru-iagora-modele-v1'; // Cache Storage used by the first versions (Safari dropped its files)

// Sizes are listed so the progress bar knows the total before the first byte arrives.
// Biggest first: its weights are read while nothing else is in memory yet.
const GRAPHS = {
  dit: { file: 'onnx/dit_q4.onnx', size: 5929366, chunks: [['dit_q4_chunk_0.data', 96468992], ['dit_q4_chunk_1.data', 99614720], ['dit_q4_chunk_2.data', 99614720], ['dit_q4_chunk_3.data', 84451328]] },
  textEncoder: { file: 'onnx/text_encoder_q4.onnx', size: 2232988, chunks: [['text_encoder_q4_chunk_0.data', 98304000], ['text_encoder_q4_chunk_1.data', 99418112], ['text_encoder_q4_chunk_2.data', 14811136]] },
  decoder: { file: 'onnx/decoder_q4.onnx', size: 1653261, chunks: [['decoder_q4_chunk_0.data', 44894208]] },
  seconds: { file: 'onnx/number_conditioner.onnx', size: 798844, chunks: [] },
};

const SAMPLE_RATE = 44100;
const LATENT_CHANNELS = 256;
const TEXT_TOKENS = 256;
const COND_DIM = 768;
const STEPS = 8; // what the model was distilled for; 4 is twice as fast, a little rougher
const PADDING_SECONDS = 6; // generation always runs 6 s longer than asked, as in stable-audio-tools
const MAX_SECONDS = 120;

let tokenizer = null;
let sessions = null;
let backend = null;
let cancelRequested = false;

const post = (message, transfer) => self.postMessage(message, transfer ?? []);

function allFiles() {
  const files = [];
  for (const graph of Object.values(GRAPHS)) {
    files.push([graph.file, graph.size]);
    for (const [name, size] of graph.chunks) files.push([`onnx/${name}`, size]);
  }
  return files;
}

// The model is kept in the browser's private file system (OPFS): it streams to disk while downloading, and a worker
// can read any byte range of a file synchronously, which is what keeps the weights out of memory (see DiskBytes).
// Cache Storage was used first, but Safari on iPhone silently kept none of these large files.
async function openStore() {
  try {
    const root = await navigator.storage.getDirectory();
    const directory = await root.getDirectoryHandle(STORE_NAME, { create: true });
    // Sync access handles are the part that matters; old browsers have the directory without them.
    if (typeof FileSystemFileHandle?.prototype.createSyncAccessHandle !== 'function') return null;
    return directory;
  } catch {
    return null; // Private window or old browser: the model is kept in memory instead.
  }
}

const fileName = path => path.replaceAll('/', '__');

async function storedSize(store, path) {
  try {
    return (await (await store.getFileHandle(fileName(path))).getFile()).size;
  } catch {
    return -1;
  }
}

async function isStored() {
  const store = await openStore();
  if (!store) return false;
  const sizes = await Promise.all(allFiles().map(([path]) => storedSize(store, path)));
  return allFiles().every(([, size], index) => sizes[index] === size);
}

async function fetchChecked(path) {
  const response = await fetch(`${MODEL_BASE}/${path}`);
  if (!response.ok) throw new Error(`téléchargement de ${path} impossible (${response.status})`);
  return response;
}

// Without storage, the file has to stay in memory: read it straight into a buffer of the expected size.
async function downloadToMemory(path, size, onBytes) {
  const reader = (await fetchChecked(path)).body.getReader();
  let data = new Uint8Array(size);
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.byteLength > data.byteLength) {
      const bigger = new Uint8Array(Math.max(data.byteLength * 2, received + value.byteLength));
      bigger.set(data);
      data = bigger;
    }
    data.set(value, received);
    received += value.byteLength;
    onBytes(value.byteLength);
  }
  if (received !== size) throw new Error(`téléchargement de ${path} interrompu`);
  return data;
}

// With storage, each piece goes to disk as it arrives. An interrupted file has the wrong size and is downloaded again.
async function downloadToStore(store, path, size, onBytes) {
  const response = await fetchChecked(path);
  const handle = await (await store.getFileHandle(fileName(path), { create: true })).createSyncAccessHandle();
  let written = 0;
  try {
    handle.truncate(0);
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      written += handle.write(value, { at: written });
      onBytes(value.byteLength);
    }
    handle.flush();
  } catch (error) {
    if (error?.name === 'QuotaExceededError') throw new Error('pas assez de place libre sur cet appareil pour garder le modèle (660 Mo)');
    throw error;
  } finally {
    handle.close();
  }
  if (written !== size) throw new Error(`téléchargement de ${path} interrompu`);
}

// Two downloads at a time: enough to fill the connection, little memory in flight.
async function downloadAll(store, onProgress) {
  const files = allFiles();
  const total = files.reduce((sum, [, size]) => sum + size, 0);
  let loaded = 0;
  const report = bytes => {
    loaded += bytes;
    onProgress(Math.min(loaded, total), total);
  };
  const memory = new Map();
  const queue = [...files];
  await Promise.all(Array.from({ length: 2 }, async () => {
    while (queue.length) {
      const [path, size] = queue.shift();
      if (store && await storedSize(store, path) === size) report(size);
      else if (store) await downloadToStore(store, path, size, report);
      else memory.set(path, await downloadToMemory(path, size, report));
    }
  }));

  // Read one file back, only when it is needed. Weights (lazy = true) are not even read whole: see DiskBytes.
  const openHandles = [];
  const read = async (path, { lazy = false } = {}) => {
    if (!store) {
      const data = memory.get(path);
      memory.delete(path);
      return data;
    }
    const handle = await (await store.getFileHandle(fileName(path))).createSyncAccessHandle();
    if (lazy) {
      openHandles.push(handle);
      return new DiskBytes(handle);
    }
    try {
      const data = new Uint8Array(handle.getSize());
      handle.read(data, { at: 0 });
      return data;
    } finally {
      handle.close();
    }
  };
  // Files read lazily stay open until their session is built.
  read.closeAll = () => {
    for (const handle of openHandles.splice(0)) handle.close();
  };
  return read;
}

// ONNX Runtime only reads a weights file through byteLength and subarray(start, end), one weight at a time, and
// copies each piece straight to the graphics card (or its own memory). This stand-in reads each piece from disk
// when asked, so a 380 Mo weights file never sits in the page's memory: what lets phones load the model.
class DiskBytes extends Uint8Array {
  constructor(handle) {
    super(0);
    this.handle = handle;
    this.size = handle.getSize();
  }

  get byteLength() {
    return this.size;
  }

  subarray(start = 0, end = this.size) {
    const piece = new Uint8Array(end - start);
    this.handle.read(piece, { at: start });
    return piece;
  }
}

async function pickBackend(requested) {
  if (requested === 'wasm' || requested === 'webgpu') return requested;
  try {
    const adapter = self.navigator.gpu && await self.navigator.gpu.requestAdapter();
    // A software "graphics card" (SwiftShader, on machines without a usable GPU) is slower than the processor.
    const software = adapter && (adapter.info?.isFallbackAdapter || /swiftshader/i.test(`${adapter.info?.architecture} ${adapter.info?.description}`));
    if (adapter && !software) return 'webgpu';
  } catch {
    // Fall through to the processor.
  }
  return 'wasm';
}

// One graph at a time: read its files, build the session, then let the bytes go before the next graph.
async function createSession(read, graph, executionProvider) {
  const model = await read(graph.file);
  const externalData = [];
  for (const [name] of graph.chunks) externalData.push({ path: name, data: await read(`onnx/${name}`, { lazy: true }) });
  return ort.InferenceSession.create(model, {
    executionProviders: [executionProvider],
    externalData,
    graphOptimizationLevel: 'all',
    enableCpuMemArena: false, // the arena keeps its peak size forever; phones cannot afford it
    extra: { session: { disable_prepacking: '1' } }, // no second, repacked copy of the weights on the processor
  });
}

async function importRuntime(name) {
  ort = await import(ORT_DIST + ORT_BUILDS[name]);
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, self.navigator.hardwareConcurrency || 1) : 1;
}

async function load(requestedBackend) {
  try {
    await caches.delete(OLD_CACHE_NAME); // free the space taken by the first versions
  } catch {
    // No Cache Storage here: nothing to free.
  }
  const store = await openStore();
  const read = await downloadAll(store, (loaded, total) => post({ type: 'loading', loaded, total }));
  tokenizer = await loadTokenizer(new URL('./tokenizer', import.meta.url).href);

  backend = await pickBackend(requestedBackend);
  post({ type: 'starting', backend });
  const create = async executionProvider => {
    const created = {};
    for (const [name, graph] of Object.entries(GRAPHS)) {
      created[name] = await createSession(read, graph, executionProvider);
      read.closeAll();
      post({ type: 'stage', stage: name, backend: executionProvider });
      globalThis.gc?.(); // only exists when a test browser exposes it, to measure memory without pending garbage
    }
    return created;
  };
  try {
    await importRuntime(backend);
    sessions = await create(backend);
  } catch (error) {
    read.closeAll();
    // Without storage the files were handed over once and are gone: no second attempt on the processor.
    if (backend !== 'webgpu' || !store) throw error;
    backend = 'wasm';
    post({ type: 'starting', backend });
    await importRuntime(backend);
    sessions = await create(backend);
  }
  post({ type: 'ready', backend });
}

// Deterministic Gaussian noise so that a seed always gives the same instru (mulberry32 + Box-Muller).
function gaussianNoise(seed) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    const u = 1 - uniform();
    const v = uniform();
    const radius = Math.sqrt(-2 * Math.log(u));
    spare = radius * Math.sin(2 * Math.PI * v);
    return radius * Math.cos(2 * Math.PI * v);
  };
}

// LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0): steps evenly spaced in log-SNR, from pure noise (1) to clean (0).
function schedule(steps) {
  const t = [];
  for (let i = 0; i <= steps; i++) {
    const linear = 1 - i / steps;
    const logSnr = 2.0 - linear * (2.0 + 6.2);
    t.push(1 / (1 + Math.exp(logSnr)));
  }
  t[0] = 1;
  t[steps] = 0;
  return t;
}

async function generate({ id, prompt, seconds, seed, steps = STEPS }) {
  cancelRequested = false;
  const started = performance.now();
  seconds = Math.max(1, Math.min(MAX_SECONDS, seconds));

  // 1. Text: 256 tokens, padded on the right (the model learned its own padding embedding).
  const ids = tokenizer.encode(prompt).slice(0, TEXT_TOKENS);
  const inputIds = new BigInt64Array(TEXT_TOKENS);
  const attentionMask = new BigInt64Array(TEXT_TOKENS);
  ids.forEach((token, index) => {
    inputIds[index] = BigInt(token);
    attentionMask[index] = 1n;
  });
  const text = await sessions.textEncoder.run({
    input_ids: new ort.Tensor('int64', inputIds, [1, TEXT_TOKENS]),
    attention_mask: new ort.Tensor('int64', attentionMask, [1, TEXT_TOKENS]),
  });
  const duration = await sessions.seconds.run({ seconds: new ort.Tensor('float32', Float32Array.of(seconds), [1]) });
  const textData = await text.last_hidden_state.getData();
  const durationData = await duration.embedding.getData();

  // Cross-attention sees the 256 text tokens followed by the duration; adaLN sees the duration alone.
  const crossData = new Float32Array((TEXT_TOKENS + 1) * COND_DIM);
  crossData.set(textData, 0);
  crossData.set(durationData, TEXT_TOKENS * COND_DIM);
  const crossAttnCond = new ort.Tensor('float32', crossData, [1, TEXT_TOKENS + 1, COND_DIM]);
  const globalEmbed = new ort.Tensor('float32', Float32Array.from(durationData), [1, COND_DIM]);

  // 2. Latent length: (seconds + 6) s, aligned to 8192 samples, one latent frame per 4096 samples.
  const frames = Math.ceil((seconds + PADDING_SECONDS) * SAMPLE_RATE / 8192) * 2;
  const validFrames = Math.min(frames, Math.ceil(Math.floor(seconds * SAMPLE_RATE) / 4096) + Math.floor(PADDING_SECONDS * SAMPLE_RATE / 4096));
  const paddingMask = new Uint8Array(frames);
  paddingMask.fill(1, 0, validFrames);
  const paddingMaskTensor = new ort.Tensor('bool', paddingMask, [1, frames]);
  // No inpainting: the mask channel and the masked audio are all zeros.
  const localAddCond = new ort.Tensor('float32', new Float32Array((LATENT_CHANNELS + 1) * frames), [1, LATENT_CHANNELS + 1, frames]);

  // 3. Ping-pong sampler: predict the clean latent, then re-noise it to the next (lower) noise level.
  const noise = gaussianNoise(seed);
  const size = LATENT_CHANNELS * frames;
  let x = new Float32Array(size);
  for (let i = 0; i < size; i++) x[i] = noise();
  const t = schedule(steps);
  for (let step = 0; step < steps; step++) {
    if (cancelRequested) throw new Error('annulé');
    const output = await sessions.dit.run({
      x: new ort.Tensor('float32', x, [1, LATENT_CHANNELS, frames]),
      t: new ort.Tensor('float32', Float32Array.of(t[step]), [1]),
      cross_attn_cond: crossAttnCond,
      global_embed: globalEmbed,
      local_add_cond: localAddCond,
      padding_mask: paddingMaskTensor,
    });
    const velocity = await output.out.getData();
    output.out.dispose();
    const next = new Float32Array(size);
    const tNow = t[step];
    const tNext = t[step + 1];
    for (let i = 0; i < size; i++) {
      const denoised = x[i] - tNow * velocity[i];
      next[i] = tNext === 0 ? denoised : (1 - tNext) * denoised + tNext * noise();
    }
    x = next;
    post({ type: 'progress', id, step: step + 1, steps: steps + 1 });
  }

  // 4. Decode to 44.1 kHz stereo and keep only the requested duration.
  const decoded = await sessions.decoder.run({ latents: new ort.Tensor('float32', x, [1, LATENT_CHANNELS, frames]) });
  const audio = await decoded.audio.getData();
  const totalSamples = decoded.audio.dims[2];
  const length = Math.min(totalSamples, Math.round(seconds * SAMPLE_RATE));
  const fade = Math.min(length, Math.round(0.03 * SAMPLE_RATE)); // avoids a click on the last sample
  const channels = [0, 1].map(channel => {
    const data = new Float32Array(length);
    const offset = channel * totalSamples;
    for (let i = 0; i < length; i++) {
      const gain = i >= length - fade ? (length - i) / fade : 1;
      data[i] = Math.max(-1, Math.min(1, audio[offset + i])) * gain;
    }
    return data;
  });
  decoded.audio.dispose();
  for (const tensor of [text.last_hidden_state, duration.embedding]) tensor.dispose();
  post({ type: 'progress', id, step: steps + 1, steps: steps + 1 });
  post({
    type: 'done', id, left: channels[0], right: channels[1], sampleRate: SAMPLE_RATE,
    seconds: (performance.now() - started) / 1000,
  }, [channels[0].buffer, channels[1].buffer]);
}

self.addEventListener('message', async ({ data }) => {
  try {
    if (data.type === 'check-cache') post({ type: 'cache', cached: await isStored(), gpu: Boolean(self.navigator.gpu) });
    else if (data.type === 'load') await load(data.backend);
    else if (data.type === 'cancel') cancelRequested = true;
    else if (data.type === 'generate') await generate(data);
  } catch (error) {
    post({ type: 'error', id: data.id ?? null, message: error?.message ?? String(error) });
  }
});
