# Atelier instru IAgora

Un petit site pour créer son instru avec une IA de musique qui tourne **directement dans le navigateur**,
sur l'ordinateur de la personne : rien n'est envoyé à un service d'IA en ligne.

- Modèle : [Stable Audio 3.0 Small Music](https://huggingface.co/stabilityai/stable-audio-3-small-music)
  (Stability AI), le même que celui de la [démo en ligne](https://huggingface.co/spaces/stabilityai/stable-audio-3),
  dans sa [conversion ONNX 4 bits pour le navigateur](https://huggingface.co/lsb/stable-audio-3-small-music-onnx) (≈ 680 Mo,
  téléchargés une seule fois puis gardés dans le cache du navigateur).
- Moteur : ONNX Runtime Web, sur la carte graphique (WebGPU) quand c'est possible, sinon sur le processeur.
- Interface en français, aux couleurs du design system IAgora. On compose avec des mots français (style, ambiance,
  instruments, tempo, tonalité, forme) ; l'atelier écrit le prompt anglais que le modèle comprend et explique en
  français ce qu'il dit. Les précisions libres en français sont traduites sur l'ordinateur (opus-mt fr→en), et un
  prompt écrit à la main en anglais peut être retraduit en français (opus-mt en→fr).
- Instrus de 10 s à 2 min, téléchargeables en WAV (44,1 kHz, stéréo).

## Lancer en local

C'est un site statique, sans étape de compilation :

```sh
python3 -m http.server 8080
# puis ouvrir http://localhost:8080
```

`?moteur=processeur` dans l'adresse force le processeur, pour comparer avec la carte graphique.

## Déployer

N'importe quel hébergement statique convient (Vercel, Netlify, GitHub Pages…). `vercel.json` ajoute les en-têtes
`Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy: credentialless`, qui permettent au moteur d'utiliser
plusieurs cœurs du processeur quand il n'y a pas de carte graphique utilisable.

## Fichiers

- `index.html`, `static/app.js` : l'interface.
- `static/vocabulaire.js` : styles, ambiances, instruments, tonalités, en français et en anglais.
- `static/engine/worker.js` : le pipeline de génération (tokenizer → encodeur de texte T5Gemma → durée →
  transformeur de diffusion, 8 étapes « ping-pong » → décodeur audio), porté de
  [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools).
- `static/engine/translate.js` : la traduction français ↔ anglais.
- `static/iagora.css`, `static/fonts/` : le design system IAgora.

## Licences

Le code de ce dépôt est celui de l'atelier. Le modèle n'est pas redistribué ici : le navigateur le télécharge depuis
Hugging Face. Il est sous [Stability AI Community License](licences/STABILITY_AI_COMMUNITY_LICENSE.md) (usage commercial
des instrus libre en dessous d'un million de dollars de chiffre d'affaires annuel) et l'encodeur de texte sous les
[conditions d'utilisation de Gemma](licences/GEMMA_TERMS_OF_USE.md). Voir [NOTICE](NOTICE). **Powered by Stability AI.**
