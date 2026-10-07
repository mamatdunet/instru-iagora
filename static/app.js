import { STYLES, MOODS, INSTRUMENTS, KEYS, FORMS, bpmFeel } from '/static/vocabulaire.js';

const $ = selector => document.querySelector(selector);
const MAX_MOODS = 3;
const MAX_INSTRUMENTS = 5;

// --- Composer state -------------------------------------------------------------------------------

const state = {
  style: STYLES[0],
  moods: [MOODS.find(mood => mood.en === 'nostalgic')],
  instruments: [INSTRUMENTS.find(item => item.en === 'Rhodes electric piano'), INSTRUMENTS.find(item => item.en === 'vinyl crackle')],
  bpm: STYLES[0].bpm,
  key: KEYS[0],
  form: FORMS[0],
  seconds: 30,
  steps: 8,
  stepsChosen: false, // once the visitor picks a speed, the engine no longer picks it for them
  extraFr: '',
  extraEn: '',
  manual: false,
};

function chip(label, pressed, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'ia-tag';
  button.textContent = label;
  button.setAttribute('aria-pressed', String(pressed));
  button.addEventListener('click', onClick);
  return button;
}

function renderChips() {
  $('#styles').replaceChildren(...STYLES.map(style => chip(style.fr, state.style === style, () => {
    state.style = state.style === style ? null : style;
    if (state.style) setBpm(state.style.bpm);
    changed();
  })));
  // Multiple choice with a ceiling: picking one more drops the oldest choice.
  const toggle = (list, item, max) => {
    const index = list.indexOf(item);
    if (index >= 0) list.splice(index, 1);
    else {
      list.push(item);
      if (list.length > max) list.shift();
    }
    changed();
  };
  $('#moods').replaceChildren(...MOODS.map(mood => chip(mood.fr, state.moods.includes(mood), () => toggle(state.moods, mood, MAX_MOODS))));
  $('#instruments').replaceChildren(...INSTRUMENTS.map(item => chip(item.fr, state.instruments.includes(item), () => toggle(state.instruments, item, MAX_INSTRUMENTS))));
}

function fillSelect(select, options, current) {
  select.replaceChildren(...options.map((option, index) => {
    const element = document.createElement('option');
    element.value = String(index);
    element.textContent = option.fr;
    element.selected = option === current;
    return element;
  }));
}

function setBpm(bpm) {
  state.bpm = bpm;
  $('#bpm').value = String(bpm);
}

// --- Prompt (English, for the model) and its meaning (French, for the visitor) ----------------------

const joinWith = (items, word) => items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} ${word} ${items.at(-1)}`;
const capitalize = text => text.charAt(0).toUpperCase() + text.slice(1);

function buildPrompt() {
  const parts = [state.style ? `${capitalize(state.style.en)} instrumental` : 'Instrumental beat'];
  if (state.moods.length) parts.push(joinWith(state.moods.map(mood => mood.en), 'and'));
  if (state.instruments.length) parts.push(`with ${joinWith(state.instruments.map(item => item.en), 'and')}`);
  if (state.form.en) parts.push(state.form.en);
  if (state.extraEn) parts.push(state.extraEn.replace(/[.\s]+$/, '').replace(/^[A-Z](?![A-Z])/, letter => letter.toLowerCase()));
  parts.push(`${state.bpm} BPM`);
  if (state.key.en) parts.push(state.key.en);
  return parts.join(', ');
}

function styleInFrench(style) {
  if (style.phrase) return style.phrase;
  return style.fr === style.fr.toUpperCase() ? style.fr : style.fr.toLowerCase();
}

function buildMeaning() {
  const parts = [state.style ? `Une instru ${styleInFrench(state.style)}` : 'Une instru'];
  if (state.moods.length) parts.push(joinWith(state.moods.map(mood => mood.fr.toLowerCase()), 'et'));
  if (state.instruments.length) parts.push(`avec ${joinWith(state.instruments.map(item => item.phrase), 'et')}`);
  if (state.form.phrase) parts.push(state.form.phrase);
  parts.push(`à ${state.bpm} battements par minute (tempo ${bpmFeel(state.bpm)})`);
  if (state.key.phrase) parts.push(state.key.phrase);
  let meaning = `${parts.join(', ')}.`;
  if (state.extraFr.trim()) meaning += ` Et vos précisions : « ${state.extraFr.trim().replace(/[.\s]+$/, '')} ».`;
  return meaning;
}

function changed() {
  renderChips();
  $('#bpmValue').textContent = String(state.bpm);
  $('#bpmFeel').textContent = `(${bpmFeel(state.bpm)})`;
  $('#secondsValue').textContent = formatDuration(state.seconds);
  if (!state.manual) {
    $('#prompt').value = buildPrompt();
    $('#meaning').textContent = buildMeaning();
  }
  $('#manualBar').hidden = !state.manual;
  $('#translateButton').hidden = !state.manual;
  updateGenerateButton();
}

function formatDuration(seconds) {
  if (seconds < 60) return `${seconds} s`;
  const rest = seconds % 60;
  return rest ? `${Math.floor(seconds / 60)} min ${rest} s` : `${seconds / 60} min`;
}

$('#bpm').addEventListener('input', event => { state.bpm = Number(event.target.value); changed(); });
$('#seconds').addEventListener('input', event => { state.seconds = Number(event.target.value); changed(); });
$('#speed').addEventListener('change', event => {
  state.steps = Number(event.target.value);
  state.stepsChosen = true;
});
$('#key').addEventListener('change', event => { state.key = KEYS[Number(event.target.value)]; changed(); });
$('#form').addEventListener('change', event => { state.form = FORMS[Number(event.target.value)]; changed(); });

$('#prompt').addEventListener('input', () => {
  state.manual = $('#prompt').value.trim() !== buildPrompt();
  $('#meaning').textContent = state.manual
    ? 'Vous avez écrit ce prompt vous-même. Cliquez sur « Traduire en français » pour vérifier ce qu\'il dit.'
    : buildMeaning();
  changed();
});

$('#rebuildButton').addEventListener('click', () => {
  state.manual = false;
  changed();
});

$('#copyButton').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($('#prompt').value);
    flash($('#copyButton'), 'Copié !');
  } catch {
    $('#prompt').select();
  }
});

function flash(button, text) {
  const original = button.textContent;
  button.textContent = text;
  setTimeout(() => { button.textContent = original; }, 1400);
}

const pick = list => list[Math.floor(Math.random() * list.length)];
function pickSome(list, count) {
  const pool = [...list];
  return Array.from({ length: count }, () => pool.splice(Math.floor(Math.random() * pool.length), 1)[0]);
}

$('#surpriseButton').addEventListener('click', () => {
  state.style = pick(STYLES);
  state.moods = pickSome(MOODS, 1 + Math.floor(Math.random() * 2));
  state.instruments = pickSome(INSTRUMENTS, 2 + Math.floor(Math.random() * 2));
  setBpm(state.style.bpm + Math.round((Math.random() - 0.5) * 10));
  state.manual = false;
  changed();
});

// --- Translation, on this computer ------------------------------------------------------------------

const translator = (() => {
  let worker = null;
  let next = 1;
  const pending = new Map();
  let onLoading = () => {};
  function start() {
    worker = new Worker('/static/engine/translate.js', { type: 'module' });
    worker.addEventListener('message', ({ data }) => {
      if (data.type === 'loading') return onLoading(data.loaded, data.total);
      const handlers = pending.get(data.id);
      pending.delete(data.id);
      if (data.type === 'done') handlers?.resolve(data.text);
      else handlers?.reject(new Error(data.message));
    });
  }
  return {
    translate(direction, text, loading = () => {}) {
      if (!worker) start();
      onLoading = loading;
      const id = next++;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker.postMessage({ id, direction, text });
      });
    },
  };
})();

const loadingNote = (loaded, total) => `Premier usage : téléchargement du traducteur, ${formatMegabytes(loaded)} sur ${formatMegabytes(total)}…`;

let extraTimer = null;
let extraTranslation = Promise.resolve();
$('#extra').addEventListener('input', () => {
  clearTimeout(extraTimer);
  const status = $('#extraStatus');
  state.extraFr = $('#extra').value;
  if (!state.extraFr.trim()) {
    state.extraEn = '';
    status.hidden = true;
    changed();
    return;
  }
  status.hidden = false;
  status.textContent = 'Traduction en anglais sur votre ordinateur dès que vous arrêtez d\'écrire…';
  extraTimer = setTimeout(() => {
    const text = state.extraFr.trim();
    status.textContent = 'Traduction en anglais sur votre ordinateur…';
    extraTranslation = translator.translate('fr-en', text, (loaded, total) => { status.textContent = loadingNote(loaded, total); })
      .then(english => {
        if (state.extraFr.trim() !== text) return; // the visitor kept typing
        state.extraEn = english;
        status.textContent = `Traduit pour le modèle : « ${english} »`;
        changed();
      })
      .catch(error => {
        status.textContent = `La traduction n'a pas marché (${error.message}). Vous pouvez écrire vos précisions en anglais directement dans le prompt.`;
      });
  }, 900);
  changed();
});

$('#translateButton').addEventListener('click', async () => {
  const button = $('#translateButton');
  const meaning = $('#meaning');
  button.disabled = true;
  meaning.textContent = 'Traduction sur votre ordinateur…';
  try {
    const french = await translator.translate('en-fr', $('#prompt').value, (loaded, total) => { meaning.textContent = loadingNote(loaded, total); });
    meaning.textContent = `Traduction automatique, approximative (le jargon musical est parfois traduit mot à mot) : « ${french} »`;
  } catch (error) {
    meaning.textContent = `La traduction n'a pas marché : ${error.message}`;
  } finally {
    button.disabled = false;
  }
});

// --- Music engine ---------------------------------------------------------------------------------

const engine = new Worker('/static/engine/worker.js', { type: 'module' });
let engineReady = false;
let backend = null;
let current = null; // generation in progress
let nextGeneration = 1;
let secondsPerStepAndSecond = null; // measured after the first instru, for time estimates

const engineBar = $('#engineBar');
const engineLabel = $('#engineLabel');
const formatMegabytes = bytes => `${Math.round(bytes / 1e6)} Mo`;

// ?moteur=processeur or ?moteur=carte-graphique forces one or the other, to compare them on a machine.
const requestedBackend = { processeur: 'wasm', 'carte-graphique': 'webgpu' }[new URLSearchParams(location.search).get('moteur')] ?? 'auto';

engine.addEventListener('message', ({ data }) => {
  if (data.type === 'cache') {
    if (data.cached) loadEngine();
    else {
      engineLabel.textContent = 'Le modèle n\'est pas encore sur cet ordinateur. Il pèse 660 Mo et ne se télécharge '
        + 'qu\'une fois, puis le navigateur le garde. De préférence en wifi.';
      $('#loadButton').hidden = false;
    }
  } else if (data.type === 'loading') {
    setProgress(engineBar, data.loaded / data.total);
    engineLabel.textContent = `Téléchargement du modèle d'IA : ${formatMegabytes(data.loaded)} sur ${formatMegabytes(data.total)} (une seule fois sur cet ordinateur)`;
  } else if (data.type === 'starting') {
    setProgress(engineBar, 1);
    engineLabel.textContent = data.backend === 'webgpu'
      ? 'Démarrage du modèle sur la carte graphique…'
      : 'Démarrage du modèle sur le processeur…';
  } else if (data.type === 'ready') {
    engineReady = true;
    backend = data.backend;
    engineBar.hidden = true;
    $('#engineCard').classList.add('ready');
    if (backend !== 'webgpu' && !state.stepsChosen) {
      state.steps = 4;
      $('#speed').value = '4';
    }
    engineLabel.textContent = backend === 'webgpu'
      ? 'Prêt, sur la carte graphique de cet ordinateur.'
      : 'Prêt, sur le processeur de cet ordinateur (pas de carte graphique utilisable ici) : c\'est plus lent. '
        + 'La vitesse « Rapide » est choisie pour vous, et des instrus courtes vont plus vite.';
    updateGenerateButton();
  } else if (data.type === 'progress' && current?.id === data.id) {
    setProgress($('#generation .progress'), data.step / data.steps);
    const label = data.step < data.steps - 1 ? `Composition : étape ${data.step + 1} sur ${data.steps - 1}`
      : data.step === data.steps - 1 ? 'Mixage du son…' : 'Presque fini…';
    $('#generationLabel').textContent = current.estimate ? `${label} · environ ${current.estimate}` : label;
  } else if (data.type === 'done' && current?.id === data.id) {
    finishGeneration(data);
  } else if (data.type === 'error') {
    if (data.id == null) {
      engineLabel.textContent = `Le modèle d'IA n'a pas pu démarrer : ${data.message}. Essayez un navigateur récent `
        + '(Chrome, Edge, Firefox ou Safari) sur un ordinateur avec au moins 8 Go de mémoire.';
      $('#engineCard').classList.add('failed');
      $('#loadButton').hidden = false;
      $('#loadButton').textContent = 'Réessayer';
    } else if (current?.id === data.id) {
      stopGeneration(data.message === 'annulé' ? null : `La génération a échoué : ${data.message}`);
    }
  }
});

function setProgress(bar, ratio) {
  bar.hidden = false;
  bar.style.setProperty('--progress', String(Math.max(0, Math.min(1, ratio))));
}

function loadEngine() {
  $('#loadButton').hidden = true;
  $('#engineCard').classList.remove('failed');
  engineLabel.textContent = 'Préparation du modèle…';
  setProgress(engineBar, 0);
  navigator.storage?.persist?.().catch(() => {});
  engine.postMessage({ type: 'load', backend: requestedBackend });
}

$('#loadButton').addEventListener('click', loadEngine);
engine.postMessage({ type: 'check-cache' });

function updateGenerateButton() {
  const button = $('#generateButton');
  button.disabled = !engineReady || Boolean(current) || !$('#prompt').value.trim();
  button.textContent = engineReady ? 'Générer l\'instru' : 'Générer l\'instru (modèle à charger)';
}

async function generate({ prompt, meaning, seconds, steps, seed = crypto.getRandomValues(new Uint32Array(1))[0], styleName, bpm }) {
  if (!engineReady || current) return;
  const id = nextGeneration++;
  const estimate = secondsPerStepAndSecond ? formatWait(secondsPerStepAndSecond * seconds * steps) : null;
  current = { id, prompt, meaning, seconds, steps, seed, styleName, bpm, estimate };
  $('#generation').hidden = false;
  setProgress($('#generation .progress'), 0);
  $('#generationLabel').textContent = 'Lecture du prompt…';
  updateGenerateButton();
  engine.postMessage({ type: 'generate', id, prompt, seconds, seed, steps });
}

function formatWait(seconds) {
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`;
  return `${Math.round(seconds / 60)} min`;
}

$('#generateButton').addEventListener('click', async () => {
  if (!state.manual && state.extraFr.trim() && !state.extraEn) {
    clearTimeout(extraTimer);
    $('#extra').dispatchEvent(new Event('input'));
  }
  await extraTranslation;
  generate({
    prompt: $('#prompt').value.trim(),
    meaning: $('#meaning').textContent,
    seconds: state.seconds,
    steps: state.steps,
    styleName: state.style?.fr ?? 'instru',
    bpm: state.bpm,
  });
});

$('#cancelButton').addEventListener('click', () => engine.postMessage({ type: 'cancel' }));

function stopGeneration(message) {
  current = null;
  $('#generation').hidden = !message;
  if (message) {
    setProgress($('#generation .progress'), 0);
    $('#generationLabel').textContent = message;
  }
  updateGenerateButton();
}

// --- Results --------------------------------------------------------------------------------------

let resultCount = 0;

function finishGeneration({ left, right, sampleRate, seconds: computeSeconds }) {
  const job = current;
  secondsPerStepAndSecond = computeSeconds / (job.seconds * job.steps);
  stopGeneration(null);
  resultCount += 1;
  addResult({ ...job, number: resultCount, left, right, sampleRate, computeSeconds });
}

function wavBlob(left, right, sampleRate) {
  const frames = left.length;
  const buffer = new ArrayBuffer(44 + frames * 4);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, 36 + frames * 4, true); text(8, 'WAVE');
  text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 2, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 4, true); view.setUint16(32, 4, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, frames * 4, true);
  const toInt = sample => (sample < 0 ? Math.max(-32768, Math.round(sample * 32768)) : Math.min(32767, Math.round(sample * 32767)));
  for (let i = 0, offset = 44; i < frames; i++, offset += 4) {
    view.setInt16(offset, toInt(left[i]), true);
    view.setInt16(offset + 2, toInt(right[i]), true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

function peaks(left, right, count) {
  const size = Math.max(1, Math.floor(left.length / count));
  return Array.from({ length: count }, (_, bar) => {
    let peak = 0;
    for (let i = bar * size, end = Math.min(left.length, i + size); i < end; i++) {
      peak = Math.max(peak, Math.abs(left[i]), Math.abs(right[i]));
    }
    return peak;
  });
}

function drawWave(canvas, values, played) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  const styles = getComputedStyle(document.documentElement);
  const bar = width / values.length;
  values.forEach((value, index) => {
    const barHeight = Math.max(2, value * height);
    context.fillStyle = index / values.length <= played ? styles.getPropertyValue('--pink-600') : styles.getPropertyValue('--blue-300');
    context.fillRect(index * bar + bar * 0.15, (height - barHeight) / 2, bar * 0.7, barHeight);
  });
}

function addResult(result) {
  const item = $('#resultTemplate').content.firstElementChild.cloneNode(true);
  const blob = wavBlob(result.left, result.right, result.sampleRate);
  const url = URL.createObjectURL(blob);
  const audio = item.querySelector('audio');
  audio.src = url;
  item.querySelector('.result-title').textContent = `Instru n° ${result.number}`;
  item.querySelector('.result-meta').textContent = `${formatDuration(result.seconds)} · graine ${result.seed}${result.steps < 8 ? ' · rapide' : ''} · calculée en ${formatWait(result.computeSeconds)}`;
  item.querySelector('.result-prompt').textContent = result.prompt;
  item.querySelector('.result-meaning').textContent = result.meaning;
  const download = item.querySelector('[data-download]');
  download.href = url;
  const slug = result.styleName.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  download.download = `instru-iagora-${result.number}-${slug}-${result.bpm}bpm.wav`;

  const canvas = item.querySelector('.wave');
  let values = null;
  const redraw = () => {
    values ??= peaks(result.left, result.right, 160);
    drawWave(canvas, values, audio.duration ? audio.currentTime / audio.duration : 0);
  };
  audio.addEventListener('timeupdate', redraw);
  audio.addEventListener('play', () => document.querySelectorAll('.result audio').forEach(other => other !== audio && other.pause()));
  canvas.addEventListener('click', event => {
    if (!audio.duration) return;
    audio.currentTime = (event.offsetX / canvas.clientWidth) * audio.duration;
    audio.play();
  });
  new ResizeObserver(redraw).observe(canvas);

  item.querySelector('[data-variant]').addEventListener('click', () => generate({ ...result, seed: undefined }));
  item.querySelector('[data-reuse]').addEventListener('click', () => {
    $('#prompt').value = result.prompt;
    $('#prompt').dispatchEvent(new Event('input'));
    if (state.manual) $('#meaning').textContent = result.meaning;
    $('#prompt').scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
  item.querySelector('[data-remove]').addEventListener('click', () => {
    URL.revokeObjectURL(url);
    item.remove();
    updateResults();
  });

  $('#results').prepend(item);
  updateResults();
  audio.play().catch(() => {}); // autoplay may be refused; the player is right there
}

function updateResults() {
  const count = $('#results').children.length;
  $('#emptyResults').hidden = count > 0;
  $('#resultsCount').textContent = count ? `${count} instru${count > 1 ? 's' : ''}` : '';
}

// --- About ----------------------------------------------------------------------------------------

const about = $('#aboutDialog');
const openAbout = () => { about.hidden = false; about.querySelector('[data-close-about]').focus(); };
$('#aboutButton').addEventListener('click', openAbout);
document.querySelectorAll('[data-open-about]').forEach(button => button.addEventListener('click', openAbout));
about.addEventListener('click', event => {
  if (event.target === about || event.target.closest('[data-close-about]')) about.hidden = true;
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') about.hidden = true;
});

// --- Start ----------------------------------------------------------------------------------------

fillSelect($('#key'), KEYS, state.key);
fillSelect($('#form'), FORMS, state.form);
changed();
