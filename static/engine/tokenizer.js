// Gemma (T5Gemma) tokenizer, reimplemented on compact files: about 10 Mo in memory instead of 150 Mo for the
// generic Hugging Face tokenizer, which matters on phones. Same output as the original tokenizer.json:
// added tokens (<pad>, HTML tags, runs of tabs…) are cut out first, longest match first; in the rest,
// spaces become "▁", each piece is one word, then the byte-pair merge with the best priority is applied, leftmost first, until none applies.
// The files come from tools/compacter_tokenizer.py.

export async function loadTokenizer(baseUrl) {
  const [info, merges] = await Promise.all([
    fetch(`${baseUrl}/characters.json`).then(response => response.json()),
    fetch(`${baseUrl}/merges.bin`).then(response => response.arrayBuffer()),
  ]);
  const count = info.merges;
  const column = index => new Int32Array(merges, index * count * 4, count);
  const left = column(0);
  const right = column(1);
  const rank = column(2);
  const merged = column(3);
  const characters = new Map(Object.entries(info.characters));
  const encoder = new TextEncoder();
  const added = new Map(Object.entries(info.added));
  const longestAdded = Math.max(...[...added.keys()].map(token => token.length));

  // Rules are sorted by (left, right): binary search.
  function findRule(a, b) {
    let low = 0;
    let high = count - 1;
    while (low <= high) {
      const middle = (low + high) >>> 1;
      const difference = left[middle] - a || right[middle] - b;
      if (difference === 0) return middle;
      if (difference < 0) low = middle + 1;
      else high = middle - 1;
    }
    return -1;
  }

  function initialTokens(text) {
    const ids = [];
    let previousUnknown = false;
    for (const character of text) {
      const id = characters.get(character);
      if (id !== undefined) {
        ids.push(id);
        previousUnknown = false;
        continue;
      }
      const bytes = [...encoder.encode(character)].map(value => info.bytes[value]);
      if (bytes.every(value => value >= 0)) {
        ids.push(...bytes);
        previousUnknown = false;
      } else if (!previousUnknown) {
        ids.push(info.unknown); // consecutive unknown characters fuse into one token
        previousUnknown = true;
      }
    }
    return ids;
  }

  function encodePiece(text) {
    if (!text) return [];
    const ids = initialTokens(text.replaceAll(' ', '▁'));
    for (;;) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const rule = findRule(ids[i], ids[i + 1]);
        if (rule >= 0 && rank[rule] < bestRank) {
          best = i;
          bestRank = rank[rule];
        }
      }
      if (best < 0) return ids;
      ids.splice(best, 2, merged[findRule(ids[best], ids[best + 1])]);
    }
  }

  return {
    encode(text) {
      const ids = [];
      let start = 0;
      for (let i = 0; i < text.length; i++) {
        for (let length = Math.min(longestAdded, text.length - i); length > 0; length--) {
          const id = added.get(text.slice(i, i + length));
          if (id === undefined) continue;
          ids.push(...encodePiece(text.slice(start, i)), id);
          i += length - 1;
          start = i + 1;
          break;
        }
      }
      ids.push(...encodePiece(text.slice(start)));
      return ids;
    },
  };
}
