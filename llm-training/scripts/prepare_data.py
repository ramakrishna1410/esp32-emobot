"""
Trains a small custom sentencepiece tokenizer on data/corpus.txt and
pretokenizes it into the shard .bin layout that llama2.c's tinystories.py
Task/PretokDataset loader expects for vocab_source="custom":

  data/tok{VOCAB_SIZE}.model      <- sentencepiece model
  data/tok{VOCAB_SIZE}/data00.bin <- test shard (first shard)
  data/tok{VOCAB_SIZE}/data01.bin <- train shard (rest)

Mirrors tinystories.py's train_vocab()/process_shard() logic but reads from
our local synthetic corpus instead of downloading TinyStories.
"""
import os
import sys

VOCAB_SIZE = 512  # matches the real stories260K's tok512, and byte_fallback needs >= ~300

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data")
LLAMA2C_DIR = os.path.join(HERE, "..", "vendor", "llama2c")
sys.path.insert(0, LLAMA2C_DIR)

import numpy as np
import sentencepiece as spm
from tokenizer import Tokenizer  # from vendor/llama2c

corpus_path = os.path.join(DATA_DIR, "corpus.txt")
assert os.path.isfile(corpus_path), f"run generate_corpus.py first ({corpus_path} missing)"

prefix = os.path.join(DATA_DIR, f"tok{VOCAB_SIZE}")

print("Training sentencepiece tokenizer...")
spm.SentencePieceTrainer.train(
    input=corpus_path,
    model_prefix=prefix,
    model_type="bpe",
    vocab_size=VOCAB_SIZE,
    self_test_sample_size=0,
    input_format="text",
    character_coverage=1.0,
    num_threads=os.cpu_count(),
    split_digits=True,
    allow_whitespace_only_pieces=True,
    byte_fallback=True,
    unk_surface=r" \342\201\207 ",
    normalization_rule_name="identity",
)
print(f"Tokenizer written to {prefix}.model")

# export tokenizer.bin (same format run.c's -z flag expects)
tok = Tokenizer(f"{prefix}.model")
tok.export()
print(f"Exported {prefix}.bin for C inference")

# pretokenize corpus into shard bin files
bin_dir = os.path.join(DATA_DIR, f"tok{VOCAB_SIZE}")
os.makedirs(bin_dir, exist_ok=True)

with open(corpus_path, "r", encoding="utf-8") as f:
    all_lines = [l.strip() for l in f if l.strip()]

# split into two shards: shard 0 = "test" split, shard 1 = "train" split,
# matching PretokDataset's shard_filenames[:1] / [1:] convention.
split_idx = len(all_lines) // 10  # ~10% held out as shard 0 (test)
shards = [all_lines[:split_idx], all_lines[split_idx:]]

for shard_id, shard_lines in enumerate(shards):
    all_tokens = []
    for line in shard_lines:
        tokens = tok.encode(line, bos=True, eos=False)
        all_tokens.extend(tokens)
    arr = np.array(all_tokens, dtype=np.uint16)
    out_path = os.path.join(bin_dir, f"data{shard_id:02d}.bin")
    with open(out_path, "wb") as f:
        f.write(arr.tobytes())
    print(f"Wrote {out_path}: {len(shard_lines)} lines, {len(all_tokens)} tokens")

print("Done. Data ready for train.py --vocab_source=custom --vocab_size=%d" % VOCAB_SIZE)
