// What both engine workers share: where the model comes from, where it is kept on this device, and how one of its
// graphs is built with ONNX Runtime Web.

// Two builds of ONNX Runtime: only the processor build has the 4-bit embedding operator (GatherBlockQuantized)
// for the processor; the graphics-card build has it for the graphics card.
const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
const ORT_BUILDS = { webgpu: 'ort.webgpu.bundle.min.mjs', wasm: 'ort.wasm.bundle.min.mjs' };

export const MODEL_BASE = 'https://huggingface.co/lsb/stable-audio-3-small-music-onnx/resolve/d523962dc41a9c632a4928ff9e02538bb11f1806';
const STORE_NAME = 'instru-iagora-modele-v1';

// Sizes are listed so the progress bar knows the total before the first byte arrives.
// Biggest first: its weights are read while nothing else is in memory yet.
export const GRAPHS = {
  dit: { file: 'onnx/dit_q4.onnx', size: 5929366, chunks: [['dit_q4_chunk_0.data', 96468992], ['dit_q4_chunk_1.data', 99614720], ['dit_q4_chunk_2.data', 99614720], ['dit_q4_chunk_3.data', 84451328]] },
  textEncoder: { file: 'onnx/text_encoder_q4.onnx', size: 2232988, chunks: [['text_encoder_q4_chunk_0.data', 98304000], ['text_encoder_q4_chunk_1.data', 99418112], ['text_encoder_q4_chunk_2.data', 14811136]] },
  decoder: { file: 'onnx/decoder_q4.onnx', size: 1653261, chunks: [['decoder_q4_chunk_0.data', 44894208]] },
  seconds: { file: 'onnx/number_conditioner.onnx', size: 798844, chunks: [] },
};

export function allFiles() {
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
export async function openStore() {
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

export const fileName = path => path.replaceAll('/', '__');

export async function storedSize(store, path) {
  try {
    return (await (await store.getFileHandle(fileName(path))).getFile()).size;
  } catch {
    return -1;
  }
}

// Read a stored file back: whole (the graph description) or lazily (weights, see DiskBytes).
// Lazily read files stay open until closeAll(), once their session is built.
export function storeReader(store) {
  const openHandles = [];
  const read = async (path, { lazy = false } = {}) => {
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

export async function importRuntime(backend) {
  const ort = await import(ORT_DIST + ORT_BUILDS[backend]);
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, self.navigator.hardwareConcurrency || 1) : 1;
  return ort;
}

// Read a graph's files and build its session; the weights are only read while ONNX Runtime copies them.
export async function createSession(ort, read, graph, executionProvider) {
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
