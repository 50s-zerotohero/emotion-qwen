"""
Script: 01_generate_stories.py

Generate emotion stories (12 emotions × 50 = 600) and neutral texts (200)
using the Claude API, then save them to data/stories/.

Usage:
    python scripts/01_generate_stories.py

Output:
    data/stories/emotion_stories.json   -- {emotion: [story, ...], ...}
    data/stories/neutral_texts.json     -- [{"topic": ..., "text": ...}, ...]

Resume behavior:
    Already-generated emotions/neutral texts are loaded from disk and skipped,
    so the script is safe to re-run after interruption.
"""

import sys
import json
import time
from pathlib import Path

# Allow running as `python scripts/01_generate_stories.py` from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotion_probe.config import load_config, ANTHROPIC_API_KEY
from emotion_probe.probe.story_generation import generate_emotion_stories
from emotion_probe.probe.neutral_generation import generate_neutral_texts, NEUTRAL_TOPICS

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).parent.parent
STORIES_DIR = PROJECT_ROOT / "data" / "stories"
STORIES_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_STORIES_PATH = STORIES_DIR / "emotion_stories.json"
NEUTRAL_TEXTS_PATH = STORIES_DIR / "neutral_texts.json"

# --------------------------------------------------------------------------- #
# Guard: API key must be set
# --------------------------------------------------------------------------- #
if not ANTHROPIC_API_KEY:
    print("ERROR: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill in your key.")
    sys.exit(1)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
cfg = load_config()
EMOTIONS: list[str] = cfg["emotions"]
N_PER_EMOTION: int = cfg["story_generation"]["stories_per_emotion"]
N_NEUTRAL: int = cfg["neutral_generation"]["n_neutral_texts"]
N_PER_TOPIC: int = N_NEUTRAL // len(NEUTRAL_TOPICS)  # 200 / 20 = 10

INTER_EMOTION_PAUSE = 3.0   # seconds between emotion batches

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
    # ---- Emotion stories -------------------------------------------------- #
    print("=" * 60)
    print(f"Generating emotion stories: {len(EMOTIONS)} emotions × {N_PER_EMOTION} each")
    print("=" * 60)

    emotion_stories: dict[str, list[str]] = load_json_or_default(EMOTION_STORIES_PATH, {})

    for idx, emotion in enumerate(EMOTIONS):
        existing = emotion_stories.get(emotion, [])
        remaining = N_PER_EMOTION - len(existing)

        if remaining <= 0:
            print(f"[{idx+1}/{len(EMOTIONS)}] '{emotion}': already complete ({len(existing)} stories), skipping")
            continue

        print(f"\n[{idx+1}/{len(EMOTIONS)}] '{emotion}': generating {remaining} stories "
              f"(already have {len(existing)})")

        new_stories = generate_emotion_stories(
            emotion=emotion,
            n=remaining,
            inter_request_delay=0.3,
        )
        emotion_stories[emotion] = existing + new_stories

        # Save after each emotion so progress is not lost on interruption
        save_json(EMOTION_STORIES_PATH, emotion_stories)
        print(f"  Saved. Total for '{emotion}': {len(emotion_stories[emotion])}")

        if idx < len(EMOTIONS) - 1:
            print(f"  Pausing {INTER_EMOTION_PAUSE:.0f}s before next emotion...")
            time.sleep(INTER_EMOTION_PAUSE)

    total_stories = sum(len(v) for v in emotion_stories.values())
    print(f"\nEmotion stories complete: {total_stories} total")

    # ---- Neutral texts ---------------------------------------------------- #
    print("\n" + "=" * 60)
    print(f"Generating neutral texts: {len(NEUTRAL_TOPICS)} topics × {N_PER_TOPIC} each = {N_NEUTRAL}")
    print("=" * 60)

    neutral_texts: list[dict] = load_json_or_default(NEUTRAL_TEXTS_PATH, [])

    already_done_topics: set[str] = set()
    topic_counts: dict[str, int] = {}
    for entry in neutral_texts:
        topic_counts[entry["topic"]] = topic_counts.get(entry["topic"], 0) + 1
    for topic, count in topic_counts.items():
        if count >= N_PER_TOPIC:
            already_done_topics.add(topic)

    topics_to_generate = [t for t in NEUTRAL_TOPICS if t not in already_done_topics]

    if not topics_to_generate:
        print("Neutral texts already complete, skipping.")
    else:
        print(f"Topics to generate: {len(topics_to_generate)} "
              f"(skipping {len(already_done_topics)} already done)")

        # Re-run only for missing topics
        import importlib
        import emotion_probe.probe.neutral_generation as ng_mod
        from emotion_probe.probe.neutral_generation import (
            NEUTRAL_TEXT_PROMPT, NEUTRAL_TOPICS as ALL_TOPICS
        )
        import anthropic as _anthropic
        from emotion_probe.config import ANTHROPIC_API_KEY as _key

        client = _anthropic.Anthropic(api_key=_key)

        for t_idx, topic in enumerate(topics_to_generate):
            prompt = NEUTRAL_TEXT_PROMPT.format(topic=topic)
            print(f"\n[{t_idx+1}/{len(topics_to_generate)}] topic: '{topic[:50]}'")
            for i in range(N_PER_TOPIC):
                for attempt in range(3):
                    try:
                        response = client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=400,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        text = response.content[0].text.strip()
                        neutral_texts.append({"topic": topic, "text": text})
                        break
                    except _anthropic.RateLimitError:
                        wait = 2.0 * (2 ** attempt)
                        print(f"  Rate limit, waiting {wait:.0f}s...")
                        time.sleep(wait)
                    except Exception as e:
                        print(f"  Error #{i+1} attempt {attempt+1}: {e}")
                        time.sleep(2.0)
                else:
                    print(f"  WARNING: Failed neutral text for topic '{topic}' #{i+1}")

                time.sleep(0.3)

            save_json(NEUTRAL_TEXTS_PATH, neutral_texts)
            print(f"  Saved. Running total: {len(neutral_texts)}")

    print(f"\nNeutral texts complete: {len(neutral_texts)} total")

    # ---- Final summary ---------------------------------------------------- #
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  emotion_stories.json : {sum(len(v) for v in emotion_stories.values())} stories")
    print(f"  neutral_texts.json   : {len(neutral_texts)} texts")
    print("=" * 60)


if __name__ == "__main__":
    main()
