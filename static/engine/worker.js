// Stable Audio 3.0 Small Music, run entirely in the browser with ONNX Runtime Web.
// Pipeline: tokenizer → text encoder (T5Gemma) → duration embedder → diffusion transformer (8 "ping-pong" steps)
// → audio decoder. Ported from stable-audio-tools (generate_diffusion_cond_inpaint + sample_flow_pingpong).
import { Tokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/+esm';

// Two builds of ONNX Runtime: only the processor build has the 4-bit embedding operator (GatherBlockQuantized)
// for the processor; the graphics-card build has it for the graphics card.
const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
const ORT_BUILDS = { webgpu: 'ort.webgpu.bundle.min.mjs', wasm: 'ort.wasm.bundle.min.mjs' };
let ort = null;

const MODEL_BASE = 'https://huggingface.co/lsb/stable-audio-3-small-music-onnx/resolve/d523962dc41a9c632a4928ff9e02538bb11f1806';
const CACHE_NAME = 'instru-iagora-modele-v1';

// Sizes are listed so the progress bar knows the total before the first byte arrives.
const GRAPHS = {
  textEncoder: { file: 'onnx/text_encoder_q4.onnx', size: 2232988, chunks: [['text_encoder_q4_chunk_0.data', 98304000], ['text_encoder_q4_chunk_1.data', 99418112], ['text_encoder_q4_chunk_2.data', 14811136]] },
  seconds: { file: 'onnx/number_conditioner.onnx', size: 798844, chunks: [] },
  dit: { file: 'onnx/dit_q4.onnx', size: 5929366, chunks: [['dit_q4_chunk_0.data', 96468992], ['dit_q4_chunk_1.data', 99614720], ['dit_q4_chunk_2.data', 99614720], ['dit_q4_chunk_3.data', 84451328]] },
  decoder: { file: 'onnx/decoder_q4.onnx', size: 1653261, chunks: [['decoder_q4_chunk_0.data', 44894208]] },
};
const TOKENIZER_FILES = [['tokenizer/tokenizer.json', 34362428], ['tokenizer/tokenizer_config.json', 469]];

const SAMPLE_RATE = 44100;
const LATENT_CHANNELS = 256;
const TEXT_TOKENS = 256;
const COND_DIM = 768;
const STEPS = 8;
const PADDING_SECONDS = 6; // generation always runs 6 s longer than asked, as in stable-audio-tools
const MAX_SECONDS = 120;

let tokenizer = null;
let sessions = null;
let backend = null;
let cancelRequested = false;

const post = (message, transfer) => self.postMessage(message, transfer ?? []);

function allFiles() {
  const files = [...TOKENIZER_FILES];
  for (const graph of Object.values(GRAPHS)) {
    files.push([graph.file, graph.size]);
    for (const [name, size] of graph.chunks) files.push([`onnx/${name}`, size]);
  }
  return files;
}

async function openCache() {
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null; // No Cache Storage (private window, insecure context): download every visit.
  }
}

async function isCached() {
  const cache = await openCache();
  if (!cache) return false;
  const found = await Promise.all(allFiles().map(([path]) => cache.match(`${MODEL_BASE}/${path}`)));
  return found.every(Boolean);
}

// Download one file (or read it from the cache), reporting bytes as they arrive.
async function fetchFile(cache, path, onBytes) {
  const url = `${MODEL_BASE}/${path}`;
  const cached = cache && await cache.match(url);
  if (cached) {
    const data = new Uint8Array(await cached.arrayBuffer());
    onBytes(data.byteLength);
    return data;
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`téléchargement de ${path} impossible (${response.status})`);
  const expected = Number(response.headers.get('content-length')) || 0;
  const reader = response.body.getReader();
  const parts = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    received += value.byteLength;
    onBytes(value.byteLength);
  }
  if (expected && received !== expected) throw new Error(`téléchargement de ${path} interrompu`);
  const data = new Uint8Array(received);
  let offset = 0;
  for (const part of parts) {
    data.set(part, offset);
    offset += part.byteLength;
  }
  if (cache) {
    try {
      await cache.put(url, new Response(data));
    } catch {
      // Storage full: the model still runs, it will simply be downloaded again next time.
    }
  }
  return data;
}

// At most three downloads at a time keeps the browser responsive and the progress bar smooth.
async function downloadAll(onProgress) {
  const cache = await openCache();
  const files = allFiles();
  const total = files.reduce((sum, [, size]) => sum + size, 0);
  let loaded = 0;
  const report = bytes => {
    loaded += bytes;
    onProgress(Math.min(loaded, total), total);
  };
  const results = new Map();
  const queue = [...files];
  await Promise.all(Array.from({ length: 3 }, async () => {
    while (queue.length) {
      const [path] = queue.shift();
      results.set(path, await fetchFile(cache, path, report));
    }
  }));
  return results;
}

async function pickBackend(requested) {
  if (requested === 'wasm') return 'wasm';
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

async function createSession(files, graph, executionProvider) {
  const externalData = graph.chunks.map(([name]) => ({ path: name, data: files.get(`onnx/${name}`) }));
  return ort.InferenceSession.create(files.get(graph.file), {
    executionProviders: [executionProvider],
    externalData,
    graphOptimizationLevel: 'all',
  });
}

async function importRuntime(name) {
  ort = await import(ORT_DIST + ORT_BUILDS[name]);
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, self.navigator.hardwareConcurrency || 1) : 1;
}

async function load(requestedBackend) {
  const files = await downloadAll((loaded, total) => post({ type: 'loading', loaded, total }));
  const decodeJson = path => JSON.parse(new TextDecoder().decode(files.get(path)));
  tokenizer = new Tokenizer(decodeJson('tokenizer/tokenizer.json'), decodeJson('tokenizer/tokenizer_config.json'));

  backend = await pickBackend(requestedBackend);
  post({ type: 'starting', backend });
  // GPU resources must be created one graph at a time; doing it in parallel can exhaust GPU memory.
  const create = async executionProvider => {
    const created = {};
    for (const [name, graph] of Object.entries(GRAPHS)) created[name] = await createSession(files, graph, executionProvider);
    return created;
  };
  try {
    await importRuntime(backend);
    sessions = await create(backend);
  } catch (error) {
    if (backend !== 'webgpu') throw error;
    backend = 'wasm';
    post({ type: 'starting', backend });
    await importRuntime(backend);
    sessions = await create(backend);
  }
  files.clear();
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

async function generate({ id, prompt, seconds, seed }) {
  cancelRequested = false;
  const started = performance.now();
  seconds = Math.max(1, Math.min(MAX_SECONDS, seconds));

  // 1. Text: 256 tokens, padded on the right (the model learned its own padding embedding).
  const ids = tokenizer.encode(prompt, { add_special_tokens: false }).ids.slice(0, TEXT_TOKENS);
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
  const t = schedule(STEPS);
  for (let step = 0; step < STEPS; step++) {
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
    post({ type: 'progress', id, step: step + 1, steps: STEPS + 1 });
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
  post({ type: 'progress', id, step: STEPS + 1, steps: STEPS + 1 });
  post({
    type: 'done', id, left: channels[0], right: channels[1], sampleRate: SAMPLE_RATE,
    seconds: (performance.now() - started) / 1000,
  }, [channels[0].buffer, channels[1].buffer]);
}

self.addEventListener('message', async ({ data }) => {
  try {
    if (data.type === 'check-cache') post({ type: 'cache', cached: await isCached() });
    else if (data.type === 'load') await load(data.backend);
    else if (data.type === 'cancel') cancelRequested = true;
    else if (data.type === 'generate') await generate(data);
  } catch (error) {
    post({ type: 'error', id: data.id ?? null, message: error?.message ?? String(error) });
  }
});
