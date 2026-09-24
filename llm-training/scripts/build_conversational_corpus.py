"""
Parses data_source/personality_seed.txt (grouped into categories by
"## === category_name ===" headers, containing User:/Bot: exchanges) into
data/corpus.txt for the training pipeline.

Beyond the literal authored exchanges, this also does within-category
cross-pairing augmentation: for each user trigger line, it pairs the
original bot response PLUS a few other bot responses from the SAME
category. This multiplies the training data several-fold without
inventing new content — every response used is still one the user wrote,
just paired with more of the semantically-similar triggers in its
category, which is exactly the variation a tiny model needs to learn
"category -> one of several natural responses" instead of memorizing 80
fixed exchanges verbatim (see llm-training/README.md, "why augmentation").

Each exchange becomes one line in the corpus, in the same delimiter format
the model will see at inference time:
  User: <trigger text>\\nBot: <response>   (literal two-char \\n, see below)
"""
import os
import random

HERE = os.path.dirname(__file__)
SEED_PATH = os.path.join(HERE, "..", "data_source", "personality_seed.txt")
OUT_DIR = os.path.join(HERE, "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

EXTRA_PAIRINGS_PER_LINE = 3  # how many additional same-category bot responses to pair with each user line
AUG_SEED = 1337


def parse_seed(path):
    """Returns {category_name: [(user, bot), ...]}, preserving order."""
    categories = {}
    current_category = "uncategorized"
    user_line = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("## ==="):
                # "## === category_name ===" -> category_name
                current_category = stripped.strip("# =").strip()
                categories.setdefault(current_category, [])
                continue
            if stripped.startswith("#"):
                continue  # ordinary comment
            if stripped.startswith("User:"):
                user_line = stripped[len("User:"):].strip()
            elif stripped.startswith("Bot:"):
                if user_line is None:
                    raise ValueError(f"Bot: line with no preceding User: line: {line!r}")
                categories.setdefault(current_category, []).append((user_line, stripped[len("Bot:"):].strip()))
                user_line = None
            else:
                raise ValueError(f"Unexpected line (expected User:/Bot:/#/blank): {line!r}")
    return categories


categories = parse_seed(SEED_PATH)
total_original = sum(len(v) for v in categories.values())
print(f"Parsed {total_original} exchanges across {len(categories)} categories from {SEED_PATH}")

rng = random.Random(AUG_SEED)
augmented = []  # list of (user, bot)
for cat_name, exchanges in categories.items():
    bot_lines = [bot for _, bot in exchanges]
    for user, bot in exchanges:
        augmented.append((user, bot))  # the original, authored pairing
        # pick up to EXTRA_PAIRINGS_PER_LINE other bot lines from the same category
        others = [b for b in bot_lines if b != bot]
        k = min(EXTRA_PAIRINGS_PER_LINE, len(others))
        for extra_bot in rng.sample(others, k):
            augmented.append((user, extra_bot))

rng.shuffle(augmented)
print(f"Augmented to {len(augmented)} exchanges (within-category cross-pairing, +{EXTRA_PAIRINGS_PER_LINE} per line)")

corpus_path = os.path.join(OUT_DIR, "corpus.txt")
with open(corpus_path, "w", encoding="utf-8") as f:
    for user, bot in augmented:
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
if total_original < 200:
    print(
        f"NOTE: only {total_original} authored exchanges (before augmentation) — "
        "consider adding more to data_source/personality_seed.txt over time; "
        "augmentation helps but isn't a substitute for more real content."
    )
