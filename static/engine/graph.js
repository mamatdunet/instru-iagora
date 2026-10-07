// A short-lived worker holding one or two graphs of the model, for the low-memory mode. The main engine worker
// closes it as soon as its graphs are no longer needed: closing a worker is the only sure way to make the browser
// give back its graphics memory (ONNX Runtime keeps released buffers for reuse).
import { GRAPHS, openStore, storeReader, importRuntime, createSession } from './model.js';

let ort = null;
const sessions = {};

self.addEventListener('message', async ({ data }) => {
  try {
    if (data.type === 'open') {
      ort = await importRuntime(data.backend);
      const read = storeReader(await openStore());
      for (const name of data.names) {
        try {
          sessions[name] = await createSession(ort, read, GRAPHS[name], data.backend);
        } finally {
          read.closeAll();
        }
      }
      self.postMessage({ type: 'opened', id: data.id });
    } else if (data.type === 'run') {
      const feeds = Object.fromEntries(Object.entries(data.feeds).map(([name, tensor]) => [name, new ort.Tensor(tensor.type, tensor.data, tensor.dims)]));
      const results = await sessions[data.name].run(feeds);
      const outputs = {};
      const transfer = [];
      for (const [name, tensor] of Object.entries(results)) {
        const values = await tensor.getData();
        outputs[name] = { data: values, dims: tensor.dims };
        transfer.push(values.buffer);
        tensor.dispose();
      }
      self.postMessage({ type: 'ran', id: data.id, outputs }, transfer);
    }
  } catch (error) {
    self.postMessage({ type: 'error', id: data.id, message: error?.message ?? String(error) });
  }
});
