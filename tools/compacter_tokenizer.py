"""Convert the Gemma tokenizer.json (34 MB, ~150 MB once loaded in a browser) into the compact files used by
static/engine/tokenizer.js. Nothing is dropped: the single-character tokens, the byte-fallback tokens and every
merge rule are kept, only stored as numbers.

    python3 tools/compacter_tokenizer.py chemin/vers/tokenizer.json static/engine/tokenizer
"""
import json
import sys
from array import array
from pathlib import Path

source, destination = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads(source.read_text())
model = data["model"]
assert model["type"] == "BPE" and model["byte_fallback"] and not model.get("ignore_merges")
assert all(not token["normalized"] and not token["lstrip"] and not token["rstrip"] and not token["single_word"]
           for token in data["added_tokens"])  # matched as is in the raw text
assert data["normalizer"] == {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
vocab = model["vocab"]

characters = {token: index for token, index in vocab.items() if len(token) == 1}
# -1: no <0xNN> token for this byte (a character needing it becomes the unknown token, as in Hugging Face tokenizers).
byte_tokens = [vocab.get(f"<0x{value:02X}>", -1) for value in range(256)]

# Each merge rule (left, right) → merged token, with its priority (its position in the list).
rules = []
for rank, (left, right) in enumerate(model["merges"]):
    rules.append((vocab[left], vocab[right], rank, vocab[left + right]))
rules.sort()  # by (left, right), for binary search in the browser
columns = [array("i", (rule[column] for rule in rules)) for column in range(4)]
assert all(column.itemsize == 4 for column in columns)
with open(destination / "merges.bin", "wb") as output:
    for column in columns:
        column.tofile(output)  # little-endian on every machine this runs on

(destination / "characters.json").write_text(json.dumps(
    {"unknown": vocab[model["unk_token"]], "bytes": byte_tokens, "characters": characters, "merges": len(rules),
     "added": {token["content"]: token["id"] for token in data["added_tokens"]}},
    ensure_ascii=False, separators=(",", ":")))
print(f"{len(characters)} characters, {len(rules)} merges")
