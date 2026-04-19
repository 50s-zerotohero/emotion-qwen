"""
Script: 02_verify_story_lengths.py

Verify token length distribution of generated stories and neutral texts
using the Qwen3-4B tokenizer. Flags samples below 80 tokens.

Usage:
    python scripts/02_verify_story_lengths.py

Output:
    - Console report with per-emotion stats and short-sample warnings
    - Histogram saved to data/stories/token_length_histogram.png
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotion_probe.config import load_config, HF_TOKEN

PROJECT_ROOT = Path(__file__).parent.parent
STORIES_DIR = PROJECT_ROOT / "data" / "stories"

cfg = load_config()
MIN_TOKENS: int = cfg["story_generation"]["min_tokens"]  # 80
WARN_THRESHOLD = 0.05  # 5% short → recommend regeneration

# --------------------------------------------------------------------------- #
# Load tokenizer (CPU only, no model weights needed)
# --------------------------------------------------------------------------- #
print("Loading Qwen3-4B tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-4B",
    token=HF_TOKEN,
    trust_remote_code=True,
)
print("Tokenizer loaded.\n")

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
with open(STORIES_DIR / "emotion_stories.json") as f:
    emotion_stories: dict[str, list[str]] = json.load(f)

with open(STORIES_DIR / "neutral_texts.json") as f:
    neutral_texts: list[dict] = json.load(f)

# --------------------------------------------------------------------------- #
# Count tokens
# --------------------------------------------------------------------------- #
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

print("Counting tokens for emotion stories...")
emotion_lengths: dict[str, list[int]] = {}
short_emotion_samples: list[dict] = []

for emotion, stories in emotion_stories.items():
    lengths = [count_tokens(s) for s in stories]
    emotion_lengths[emotion] = lengths
    for i, (length, story) in enumerate(zip(lengths, stories)):
        if length < MIN_TOKENS:
            short_emotion_samples.append({
                "emotion": emotion, "index": i,
                "tokens": length, "text": story[:120]
            })

print("Counting tokens for neutral texts...")
neutral_lengths = [count_tokens(t["text"]) for t in neutral_texts]
short_neutral_samples = [
    {"topic": neutral_texts[i]["topic"], "index": i,
     "tokens": l, "text": neutral_texts[i]["text"][:120]}
    for i, l in enumerate(neutral_lengths) if l < MIN_TOKENS
]

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
all_emotion_lengths = [l for ls in emotion_lengths.values() for l in ls]

def stats(lengths: list[int]) -> dict:
    import statistics
    return {
        "n": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(statistics.mean(lengths), 1),
        "median": statistics.median(lengths),
        "short": sum(1 for l in lengths if l < MIN_TOKENS),
    }

print("\n" + "=" * 60)
print("EMOTION STORIES — per-emotion token stats")
print("=" * 60)
print(f"{'emotion':<12} {'n':>4} {'min':>5} {'max':>5} {'mean':>6} {'median':>7} {'<80':>5}")
print("-" * 60)
for emotion, lengths in emotion_lengths.items():
    s = stats(lengths)
    flag = " !" if s["short"] > 0 else ""
    print(f"{emotion:<12} {s['n']:>4} {s['min']:>5} {s['max']:>5} {s['mean']:>6} {int(s['median']):>7} {s['short']:>5}{flag}")

s_all = stats(all_emotion_lengths)
print("-" * 60)
print(f"{'ALL':<12} {s_all['n']:>4} {s_all['min']:>5} {s_all['max']:>5} {s_all['mean']:>6} {int(s_all['median']):>7} {s_all['short']:>5}")

short_pct = s_all["short"] / s_all["n"]
print(f"\nShort samples (<{MIN_TOKENS} tokens): {s_all['short']} / {s_all['n']} = {short_pct:.1%}")
if short_pct >= WARN_THRESHOLD:
    print(f"WARNING: {short_pct:.1%} >= {WARN_THRESHOLD:.0%} threshold → consider regenerating short samples")
else:
    print(f"OK: below {WARN_THRESHOLD:.0%} threshold")

print("\n" + "=" * 60)
print("NEUTRAL TEXTS — token stats")
print("=" * 60)
s_n = stats(neutral_lengths)
print(f"n={s_n['n']}  min={s_n['min']}  max={s_n['max']}  mean={s_n['mean']}  median={int(s_n['median'])}  <80={s_n['short']}")
n_short_pct = s_n["short"] / s_n["n"]
if n_short_pct >= WARN_THRESHOLD:
    print(f"WARNING: {n_short_pct:.1%} short neutral texts >= threshold")
else:
    print(f"OK: {n_short_pct:.1%} short neutral texts")

# --------------------------------------------------------------------------- #
# Detail of short samples
# --------------------------------------------------------------------------- #
if short_emotion_samples:
    print("\n" + "=" * 60)
    print(f"SHORT EMOTION SAMPLES ({len(short_emotion_samples)} total)")
    print("=" * 60)
    for s in short_emotion_samples:
        print(f"  [{s['emotion']}] #{s['index']} ({s['tokens']} tokens): {s['text']}...")

if short_neutral_samples:
    print("\n" + "=" * 60)
    print(f"SHORT NEUTRAL SAMPLES ({len(short_neutral_samples)} total)")
    print("=" * 60)
    for s in short_neutral_samples:
        print(f"  [{s['topic'][:40]}] #{s['index']} ({s['tokens']} tokens): {s['text']}...")

# --------------------------------------------------------------------------- #
# Histogram
# --------------------------------------------------------------------------- #
print("\nGenerating histogram...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Token Length Distribution — Qwen3-4B Tokenizer", fontsize=14)

# Emotion stories
axes[0].hist(all_emotion_lengths, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].axvline(MIN_TOKENS, color="red", linestyle="--", linewidth=1.5, label=f"min={MIN_TOKENS}")
axes[0].set_title(f"Emotion Stories (n={len(all_emotion_lengths)})")
axes[0].set_xlabel("Token count")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# Neutral texts
axes[1].hist(neutral_lengths, bins=30, color="darkorange", edgecolor="white", alpha=0.85)
axes[1].axvline(MIN_TOKENS, color="red", linestyle="--", linewidth=1.5, label=f"min={MIN_TOKENS}")
axes[1].set_title(f"Neutral Texts (n={len(neutral_lengths)})")
axes[1].set_xlabel("Token count")
axes[1].legend()

plt.tight_layout()
out_path = STORIES_DIR / "token_length_histogram.png"
plt.savefig(out_path, dpi=150)
print(f"Histogram saved to {out_path}")

print("\nDONE")
