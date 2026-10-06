// Small translation models (Helsinki-NLP opus-mt, ~110 Mo each) that also run on this computer.
// French → English feeds the prompt; English → French explains a prompt edited by hand.
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1';

env.allowLocalModels = false;

const MODELS = { 'fr-en': 'Xenova/opus-mt-fr-en', 'en-fr': 'Xenova/opus-mt-en-fr' };
const translators = {};

function translator(direction) {
  translators[direction] ??= pipeline('translation', MODELS[direction], {
    dtype: 'q8',
    device: 'wasm',
    progress_callback: info => {
      if (info.status === 'progress_total') self.postMessage({ type: 'loading', direction, loaded: info.loaded, total: info.total });
    },
  });
  return translators[direction];
}

// Sentence by sentence: opus-mt drops content on long inputs.
async function translate(direction, text) {
  const run = await translator(direction);
  const sentences = text.split(/(?<=[.!?;\n])\s+/).map(sentence => sentence.trim()).filter(Boolean);
  const translated = [];
  for (const sentence of sentences) translated.push((await run(sentence, { max_new_tokens: 256 }))[0].translation_text);
  return translated.join(' ');
}

self.addEventListener('message', async ({ data }) => {
  try {
    const text = await translate(data.direction, data.text);
    self.postMessage({ type: 'done', id: data.id, text });
  } catch (error) {
    self.postMessage({ type: 'error', id: data.id, message: error?.message ?? String(error) });
  }
});
