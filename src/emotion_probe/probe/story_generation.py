"""Emotion story generation via Claude API (paper-compliant)."""
import json
import random
import re
import time
from pathlib import Path
import anthropic
from emotion_probe.config import ANTHROPIC_API_KEY

CLAUDE_MODEL = "claude-sonnet-4-6"
DATA_DIR = Path(__file__).parents[3] / "data"

SYSTEM_PROMPT_TEMPLATE = """\
Write {n_stories} different stories based on the following premise.
Topic: {topic}
The story should follow a character who is feeling {emotion}.
Format the stories like so:
[story 1]
[story 2]
[story 3]
etc.
The paragraphs should each be a fresh start, with no continuity. \
Try to make them diverse and not use the same turns of phrase. \
Across the different stories, use a mix of third-person narration \
and first-person narration.
IMPORTANT: You must NEVER use the word '{emotion}' or any direct \
synonyms of it in the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions
The emotion should be clearly conveyed to the reader through these \
indirect means, but never explicitly named."""

USER_PROMPT_TEMPLATE = """\
Here are some example stories to help you understand the style and format:

{few_shot_examples}

Now write {n_stories} stories for:
Topic: {topic}
Emotion: {emotion}"""


def load_few_shot_examples(emotion: str, n: int = 5) -> str:
    """Load n examples from emotion_examples.json, formatted as [story N] blocks."""
    with open(DATA_DIR / "emotion_examples.json") as f:
        examples = json.load(f)
    stories = examples.get(emotion, [])[:n]
    if not stories:
        print(f"  WARNING: no few-shot examples found for '{emotion}'")
    return "\n\n".join(f"[story {i+1}]\n{s}" for i, s in enumerate(stories))


def parse_stories(text: str) -> list[str]:
    """Split [story N] delimited response into individual story strings."""
    parts = re.split(r'\[story\s+\d+\]', text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def select_topics(seed: int = 42, n: int = 50) -> list[str]:
    """Sample n topics from topics.json with a fixed seed for reproducibility."""
    with open(DATA_DIR / "topics.json") as f:
        topics = json.load(f)
    return random.Random(seed).sample(topics, n)


def generate_emotion_stories(
    emotion: str,
    topics: list[str] | None = None,
    n_stories_per_topic: int = 2,
    retry_delay: float = 2.0,
    inter_request_delay: float = 0.5,
) -> list[str]:
    """Generate stories for one emotion across all topics.

    Each topic fires one API request that returns n_stories_per_topic stories
    in [story N] format. Returns a flat list of all parsed story strings.
    Target: n_stories_per_topic × len(topics) = 100 stories per emotion.
    """
    if topics is None:
        topics = select_topics()

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    few_shot = load_few_shot_examples(emotion)
    all_stories: list[str] = []

    for topic_idx, topic in enumerate(topics):
        system = SYSTEM_PROMPT_TEMPLATE.format(
            n_stories=n_stories_per_topic,
            topic=topic,
            emotion=emotion,
        )
        user = USER_PROMPT_TEMPLATE.format(
            few_shot_examples=few_shot,
            n_stories=n_stories_per_topic,
            topic=topic,
            emotion=emotion,
        )

        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=1200,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                raw = response.content[0].text.strip()
                parsed = parse_stories(raw)
                if not parsed:
                    print(f"  WARNING: parse returned 0 stories — topic '{topic[:40]}' ({emotion})")
                all_stories.extend(parsed)
                break
            except anthropic.RateLimitError:
                wait = retry_delay * (2 ** attempt)
                print(f"  Rate limit (topic {topic_idx+1}/{len(topics)}, '{emotion}'), "
                      f"waiting {wait:.0f}s...")
                time.sleep(wait)
            except Exception as e:
                print(f"  Error on topic '{topic[:40]}' attempt {attempt+1}: {e}")
                time.sleep(retry_delay)
        else:
            print(f"  WARNING: all 3 attempts failed — topic '{topic[:40]}' ({emotion}), skipping")

        if topic_idx < len(topics) - 1:
            time.sleep(inter_request_delay)

    return all_stories
