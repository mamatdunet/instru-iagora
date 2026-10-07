// Stable Audio 3.0 Small Music, run entirely in the browser with ONNX Runtime Web.
// Pipeline: tokenizer → text encoder (T5Gemma) → duration embedder → diffusion transformer (8 "ping-pong" steps)
// → audio decoder. Ported from stable-audio-tools (generate_diffusion_cond_inpaint + sample_flow_pingpong).
import { loadTokenizer } from './tokenizer.js';
import { MODEL_BASE, GRAPHS, allFiles, openStore, storedSize, fileName, storeReader, importRuntime, createSession } from './model.js';

const OLD_CACHE_NAME = 'instru-iagora-modele-v1'; // Cache Storage used by the first versions (Safari dropped its files)

const SAMPLE_RATE = 44100;
const LATENT_CHANNELS = 256;
const TEXT_TOKENS = 256;
const COND_DIM = 768;
const STEPS = 8; // what the model was distilled for; 4 is twice as fast, a little rougher
const PADDING_SECONDS = 6; // generation always runs 6 s longer than asked, as in stable-audio-tools
const MAX_SECONDS = 120;

let ort = null;
let tokenizer = null;
let sessions = null;
let lowMemory = false;
let backend = null;
let cancelRequested = false;

const post = (message, transfer) => self.postMessage(message, transfer ?? []);

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

  if (store) return storeReader(store);
  // Without storage, each file can be handed over once, then it is forgotten.
  const read = async path => {
    const data = memory.get(path);
    memory.delete(path);
    return data;
  };
  read.closeAll = () => {};
  return read;
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

// Normal mode keeps the four graphs on the graphics card (660 Mo). Low-memory mode, for phones, keeps none: each
// generation opens the graphs it needs in a short-lived worker (see graph.js) and closes it before the next ones,
// so the peak is the biggest graph alone (380 Mo) instead of all of them. Each instru takes a few seconds longer.
async function load(requestedBackend, requestedLowMemory) {
  try {
    await caches.delete(OLD_CACHE_NAME); // free the space taken by the first versions
  } catch {
    // No Cache Storage here: nothing to free.
  }
  const store = await openStore();
  const read = await downloadAll(store, (loaded, total) => post({ type: 'loading', loaded, total }));
  tokenizer = await loadTokenizer(new URL('./tokenizer', import.meta.url).href);
  // The short-lived workers read the files from disk; without storage only the normal mode is possible.
  lowMemory = Boolean(requestedLowMemory && store);

  backend = await pickBackend(requestedBackend);
  post({ type: 'starting', backend, lowMemory });
  if (lowMemory) {
    ort = await importRuntime(backend); // for its Tensor type only: no graph lives here
    sessions = {};
    post({ type: 'ready', backend, lowMemory });
    return;
  }
  const create = async executionProvider => {
    const created = {};
    for (const [name, graph] of Object.entries(GRAPHS)) {
      created[name] = await createSession(ort, read, graph, executionProvider);
      read.closeAll();
      post({ type: 'stage', stage: name, backend: executionProvider });
      globalThis.gc?.(); // only exists when a test browser exposes it, to measure memory without pending garbage
    }
    return created;
  };
  try {
    ort = await importRuntime(backend);
    sessions = await create(backend);
  } catch (error) {
    read.closeAll();
    // Without storage the files were handed over once and are gone: no second attempt on the processor.
    if (backend !== 'webgpu' || !store) throw error;
    backend = 'wasm';
    post({ type: 'starting', backend, lowMemory });
    ort = await importRuntime(backend);
    sessions = await create(backend);
  }
  post({ type: 'ready', backend, lowMemory });
}

// Graphs opened in a short-lived worker, with the same run() as a local session.
async function openGraphs(names) {
  const worker = new Worker(new URL('./graph.js', import.meta.url), { type: 'module' });
  const pending = new Map();
  let next = 0;
  const failAll = message => {
    for (const { reject } of pending.values()) reject(new Error(message));
    pending.clear();
  };
  worker.addEventListener('message', ({ data }) => {
    const call = pending.get(data.id);
    pending.delete(data.id);
    if (data.type === 'error') call?.reject(new Error(data.message));
    else call?.resolve(data);
  });
  worker.addEventListener('error', event => failAll(event.message || 'une partie du moteur s\'est arrêtée'));
  const call = message => new Promise((resolve, reject) => {
    const id = ++next;
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...message, id });
  });
  try {
    await call({ type: 'open', names, backend });
  } catch (error) {
    worker.terminate();
    throw error;
  }
  const graphs = Object.fromEntries(names.map(name => [name, {
    async run(feeds) {
      const plain = Object.fromEntries(Object.entries(feeds).map(([key, tensor]) => [key, { type: tensor.type, data: tensor.data, dims: tensor.dims }]));
      const { outputs } = await call({ type: 'run', name, feeds: plain });
      return Object.fromEntries(Object.entries(outputs).map(([key, output]) => [key, {
        dims: output.dims,
        getData: async () => output.data,
        dispose() {},
      }]));
    },
  }]));
  return { graphs, close: () => worker.terminate() };
}

const PHASES = { textEncoder: 'Chargement de la lecture du texte…', dit: 'Chargement du compositeur…', decoder: 'Chargement du décodeur audio…' };

// Run a task with some graphs: the resident ones, or (low-memory mode) ones opened for the task and closed after.
async function withGraphs(id, names, task) {
  if (!lowMemory) return task(sessions);
  post({ type: 'phase', id, label: PHASES[names[0]] });
  const { graphs, close } = await openGraphs(names);
  try {
    return await task(graphs);
  } finally {
    close();
  }
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
  const [textData, durationData] = await withGraphs(id, ['textEncoder', 'seconds'], async graphs => {
    const { last_hidden_state: hidden } = await graphs.textEncoder.run({
      input_ids: new ort.Tensor('int64', inputIds, [1, TEXT_TOKENS]),
      attention_mask: new ort.Tensor('int64', attentionMask, [1, TEXT_TOKENS]),
    });
    const { embedding } = await graphs.seconds.run({ seconds: new ort.Tensor('float32', Float32Array.of(seconds), [1]) });
    const data = [await hidden.getData(), await embedding.getData()];
    hidden.dispose();
    embedding.dispose();
    return data;
  });

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
  x = await withGraphs(id, ['dit'], async ({ dit }) => {
    for (let step = 0; step < steps; step++) {
      if (cancelRequested) throw new Error('annulé');
      const output = await dit.run({
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
    return x;
  });

  // 4. Decode to 44.1 kHz stereo and keep only the requested duration.
  const [audio, totalSamples] = await withGraphs(id, ['decoder'], async ({ decoder }) => {
    const { audio: decoded } = await decoder.run({ latents: new ort.Tensor('float32', x, [1, LATENT_CHANNELS, frames]) });
    const result = [await decoded.getData(), decoded.dims[2]];
    decoded.dispose();
    return result;
  });
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
  post({ type: 'progress', id, step: steps + 1, steps: steps + 1 });
  post({
    type: 'done', id, left: channels[0], right: channels[1], sampleRate: SAMPLE_RATE,
    seconds: (performance.now() - started) / 1000,
  }, [channels[0].buffer, channels[1].buffer]);
}

// Everything that matters to know why the model does not start on a given device, for the "copy diagnostic" button.
async function diagnose() {
  const report = { gpuInWorker: Boolean(self.navigator.gpu), crossOriginIsolated: self.crossOriginIsolated, backend, lowMemory, ready: Boolean(sessions) };
  try {
    const adapter = await self.navigator.gpu?.requestAdapter();
    if (adapter) {
      const { vendor, architecture, description, isFallbackAdapter } = adapter.info ?? {};
      report.adapter = { vendor, architecture, description, isFallbackAdapter };
      report.limits = { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize };
      report.features = [...adapter.features].join(' ');
    } else report.adapter = null;
  } catch (error) {
    report.adapter = `erreur : ${error.message}`;
  }
  try {
    report.storage = await self.navigator.storage?.estimate?.();
    const store = await openStore();
    report.opfs = Boolean(store);
    if (store) {
      const sizes = await Promise.all(allFiles().map(([path]) => storedSize(store, path)));
      report.storedFiles = `${allFiles().filter(([, size], index) => sizes[index] === size).length}/${allFiles().length}`;
    }
  } catch (error) {
    report.opfs = `erreur : ${error.message}`;
  }
  return report;
}

self.addEventListener('message', async ({ data }) => {
  try {
    if (data.type === 'check-cache') post({ type: 'cache', cached: await isStored(), gpu: Boolean(self.navigator.gpu) });
    else if (data.type === 'load') await load(data.backend, data.lowMemory);
    else if (data.type === 'cancel') cancelRequested = true;
    else if (data.type === 'diagnose') post({ type: 'diagnostic', report: await diagnose() });
    else if (data.type === 'generate') await generate(data);
  } catch (error) {
    post({ type: 'error', id: data.id ?? null, message: error?.message ?? String(error) });
  }
});
