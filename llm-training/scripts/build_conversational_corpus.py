"""
Parses data_source/personality_seed.txt (User:/Bot: exchanges, # comments,
blank-line separated) into data/corpus.txt for the training pipeline.

Each exchange becomes one line in the corpus, in the same delimiter format
the model will see at inference time:
  User: <trigger text>\nBot: <response>

This replaces generate_corpus.py's synthetic story text for the
conversational-model iteration of the project. Run prepare_data.py after
this to re-tokenize.
"""
import os

HERE = os.path.dirname(__file__)
SEED_PATH = os.path.join(HERE, "..", "data_source", "personality_seed.txt")
OUT_DIR = os.path.join(HERE, "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_seed(path):
    exchanges = []
    user_line = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("User:"):
                user_line = stripped[len("User:"):].strip()
            elif stripped.startswith("Bot:"):
                if user_line is None:
                    raise ValueError(f"Bot: line with no preceding User: line: {line!r}")
                exchanges.append((user_line, stripped[len("Bot:"):].strip()))
                user_line = None
            else:
                raise ValueError(f"Unexpected line (expected User:/Bot:/#/blank): {line!r}")
    return exchanges


exchanges = parse_seed(SEED_PATH)
print(f"Parsed {len(exchanges)} exchanges from {SEED_PATH}")

corpus_path = os.path.join(OUT_DIR, "corpus.txt")
with open(corpus_path, "w", encoding="utf-8") as f:
    for user, bot in exchanges:
        # Each exchange must be ONE line in corpus.txt: prepare_data.py (like
        # tinystories.py) treats one file-line as one training example and
        # gives it a single BOS token. So we use a literal two-character
        # "\n" marker (backslash + n) between turns rather than an actual
        # newline, which would split the exchange across two lines and lose
        # the User/Bot pairing. The model learns this literal marker as part
        # of its vocabulary, and firmware prompts/parses using the same
        # convention at inference time.
        f.write(f"User: {user}\\nBot: {bot}\n")

print(f"Wrote {corpus_path}")
print(f"Size: {os.path.getsize(corpus_path) / 1024:.1f} KB")
if len(exchanges) < 200:
    print(
        f"NOTE: only {len(exchanges)} exchanges — this is a starter draft, "
        "not enough for real training quality. Expand data_source/"
        "personality_seed.txt before running a serious (not smoke-test) "
        "training run."
    )
