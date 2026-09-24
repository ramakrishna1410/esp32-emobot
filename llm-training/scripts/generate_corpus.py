"""
Generates a small synthetic corpus for the Phase 1 pipeline smoke test.

This is NOT the real EmoBot personality dataset (that's Phase 3, hand-curated
by the user). This exists only to exercise the train -> export -> C-inference
pipeline end-to-end in an environment where the real pretrained stories260K
checkpoint and the TinyStories dataset (both hosted on huggingface.co) are
unreachable. Deterministic and self-contained: no network access required.
"""
import os
import random

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(1337)

names = ["Lily", "Max", "Zoe", "Sam", "Emma", "Leo", "Mia", "Kai"]
moods = ["sad", "happy", "sleepy", "excited", "worried"]
things = ["a red ball", "a soft blanket", "a funny hat", "a paper boat", "a kite"]
places = ["the park", "the garden", "the kitchen", "the living room", "the yard"]

templates = [
    "{name} felt {mood} today. A little robot came over and told a silly joke, and {name} started to laugh.",
    "{name} was playing with {thing} in {place} when the small robot rolled up and said something funny.",
    "The robot noticed {name} looked {mood}, so it did a happy dance and {name} smiled again.",
    "{name} asked the robot how the weather was in the room, and the robot said it was warm and cozy.",
    "\"I wish I could come with you,\" said the robot, when {name} said they were going out.",
    "{name} came home and the robot cheered, \"Welcome back! I missed you today.\"",
    "The robot told {name} a joke about {thing}, and {name} laughed so hard they fell over.",
    "When {name} felt {mood}, the little robot gave a warm, funny story to cheer them up.",
]

lines = []
for _ in range(1200):
    t = random.choice(templates)
    line = t.format(
        name=random.choice(names),
        mood=random.choice(moods),
        thing=random.choice(things),
        place=random.choice(places),
    )
    lines.append(line)

corpus_path = os.path.join(OUT_DIR, "corpus.txt")
with open(corpus_path, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print(f"Wrote {len(lines)} lines to {corpus_path}")
print(f"Size: {os.path.getsize(corpus_path) / 1024:.1f} KB")
