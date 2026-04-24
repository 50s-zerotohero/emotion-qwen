"""Inspect emotion stories and neutral texts for quality check."""

import json
import random
from pathlib import Path

STORIES_PATH = Path("data/stories/emotion_stories.json")
NEUTRAL_PATH = Path("data/stories/neutral_texts.json")
STORIES_PER_EMOTION = 3
NEUTRAL_SAMPLES = 5
RANDOM_SEED = 42

SEP = "=" * 72
SUBSEP = "-" * 72


def print_stories(stories: list[str], emotion: str) -> None:
    print(f"\n{SEP}")
    print(f"  {emotion.upper()}  ({len(stories)} stories total, showing {STORIES_PER_EMOTION})")
    print(SEP)
    samples = random.sample(stories, min(STORIES_PER_EMOTION, len(stories)))
    for i, story in enumerate(samples, 1):
        print(f"\n[{i}] {story.strip()}")
        print(SUBSEP)


def print_neutral(entries: list[dict]) -> None:
    print(f"\n{SEP}")
    print(f"  NEUTRAL TEXTS  ({len(entries)} total, showing {NEUTRAL_SAMPLES})")
    print(SEP)
    samples = random.sample(entries, min(NEUTRAL_SAMPLES, len(entries)))
    for i, entry in enumerate(samples, 1):
        topic = entry.get("topic", "(no topic)")
        text = entry.get("text", str(entry)).strip()
        print(f"\n[{i}] topic: {topic}")
        print(f"    {text[:500]}{'...' if len(text) > 500 else ''}")
        print(SUBSEP)


def main() -> None:
    random.seed(RANDOM_SEED)

    with open(STORIES_PATH) as f:
        emotion_stories: dict[str, list[str]] = json.load(f)

    with open(NEUTRAL_PATH) as f:
        neutral_texts: list[dict] = json.load(f)

    print(f"\nLoaded {len(emotion_stories)} emotions, "
          f"{sum(len(v) for v in emotion_stories.values())} stories total.")
    print(f"Loaded {len(neutral_texts)} neutral texts.")

    for emotion, stories in emotion_stories.items():
        print_stories(stories, emotion)

    print_neutral(neutral_texts)


if __name__ == "__main__":
    main()
