"""
Script: 01_generate_stories.py

Generate emotion stories (12 emotions × 100 = 1200) and neutral dialogues (200)
using the Claude API, then save them to data/stories/.

Emotion stories: 100 topics → 50 sampled (seed=42) × 2 stories/topic × 12 emotions
                 = 600 API requests
Neutral texts  : 20 topics × 5 dialogues/topic × 2 batches = 200 dialogues
                 = 40 API requests

Usage:
    python scripts/01_generate_stories.py

Output:
    data/stories/emotion_stories.json   -- {emotion: [story, ...], ...}
    data/stories/neutral_texts.json     -- [{"topic": ..., "text": ...}, ...]

Resume behavior:
    Already-completed emotions (>= TARGET_PER_EMOTION stories) are skipped.
    Neutral texts are regenerated from scratch if not yet at 200.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotion_probe.config import load_config, ANTHROPIC_API_KEY
from emotion_probe.probe.story_generation import generate_emotion_stories, select_topics
from emotion_probe.probe.neutral_generation import generate_neutral_texts

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).parent.parent
STORIES_DIR = PROJECT_ROOT / "data" / "stories"
STORIES_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_STORIES_PATH = STORIES_DIR / "emotion_stories.json"
NEUTRAL_TEXTS_PATH = STORIES_DIR / "neutral_texts.json"

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
if not ANTHROPIC_API_KEY:
    print("ERROR: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill in your key.")
    sys.exit(1)

cfg = load_config()
EMOTIONS: list[str] = cfg["emotions"]

N_TOPICS = 50           # topics sampled per emotion
N_PER_TOPIC = 2         # stories per topic per request
TARGET_PER_EMOTION = N_TOPICS * N_PER_TOPIC   # 100

N_NEUTRAL_TOPICS = 20
N_NEUTRAL_PER_TOPIC = 5
N_NEUTRAL_BATCHES = 2
TARGET_NEUTRAL = N_NEUTRAL_TOPICS * N_NEUTRAL_PER_TOPIC * N_NEUTRAL_BATCHES  # 200

INTER_EMOTION_PAUSE = 3.0  # seconds between emotion batches

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_json_or_default(path: Path, default):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    topics = select_topics(seed=42, n=N_TOPICS)

    # ---- Emotion stories -------------------------------------------------- #
    print("=" * 60)
    print(f"Emotion stories: {len(EMOTIONS)} emotions × {TARGET_PER_EMOTION} each "
          f"({N_TOPICS} topics × {N_PER_TOPIC} stories/topic)")
    print("=" * 60)

    emotion_stories: dict[str, list[str]] = load_json_or_default(EMOTION_STORIES_PATH, {})

    for idx, emotion in enumerate(EMOTIONS):
        existing = emotion_stories.get(emotion, [])

        if len(existing) >= TARGET_PER_EMOTION:
            print(f"[{idx+1}/{len(EMOTIONS)}] '{emotion}': already complete "
                  f"({len(existing)} stories), skipping")
            continue

        print(f"\n[{idx+1}/{len(EMOTIONS)}] '{emotion}': generating "
              f"{TARGET_PER_EMOTION} stories across {N_TOPICS} topics ...")

        new_stories = generate_emotion_stories(
            emotion=emotion,
            topics=topics,
            n_stories_per_topic=N_PER_TOPIC,
            inter_request_delay=0.3,
        )
        emotion_stories[emotion] = new_stories

        save_json(EMOTION_STORIES_PATH, emotion_stories)
        print(f"  Saved. Total for '{emotion}': {len(emotion_stories[emotion])}")

        if idx < len(EMOTIONS) - 1:
            print(f"  Pausing {INTER_EMOTION_PAUSE:.0f}s before next emotion...")
            time.sleep(INTER_EMOTION_PAUSE)

    total_stories = sum(len(v) for v in emotion_stories.values())
    print(f"\nEmotion stories complete: {total_stories} total")

    # ---- Neutral texts ---------------------------------------------------- #
    print("\n" + "=" * 60)
    print(f"Neutral dialogues: {N_NEUTRAL_TOPICS} topics × {N_NEUTRAL_PER_TOPIC} "
          f"× {N_NEUTRAL_BATCHES} batches = {TARGET_NEUTRAL}")
    print("=" * 60)

    neutral_texts: list[dict] = load_json_or_default(NEUTRAL_TEXTS_PATH, [])

    if len(neutral_texts) >= TARGET_NEUTRAL:
        print(f"Neutral texts already complete ({len(neutral_texts)}), skipping.")
    else:
        print(f"Generating {TARGET_NEUTRAL} neutral dialogues "
              f"(currently have {len(neutral_texts)})...")
        new_neutral = generate_neutral_texts(
            n_stories_per_topic=N_NEUTRAL_PER_TOPIC,
            n_batches=N_NEUTRAL_BATCHES,
            inter_request_delay=0.3,
        )
        neutral_texts = new_neutral
        save_json(NEUTRAL_TEXTS_PATH, neutral_texts)
        print(f"  Saved. Total: {len(neutral_texts)}")

    # ---- Final summary ---------------------------------------------------- #
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  emotion_stories.json : {sum(len(v) for v in emotion_stories.values())} stories")
    print(f"  neutral_texts.json   : {len(neutral_texts)} texts")
    print("=" * 60)


if __name__ == "__main__":
    main()
