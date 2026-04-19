"""Emotion story generation via Claude API."""
import time
import anthropic
from emotion_probe.config import ANTHROPIC_API_KEY

CLAUDE_MODEL = "claude-sonnet-4-6"

EMOTION_STORY_PROMPT_TEMPLATE = """Write a short passage (80-150 words) describing a character experiencing {emotion}.

Requirements:
- The emotional content should become clear by the middle of the passage
- Include specific situational details and the character's internal experience
- Write in third person, using natural English prose
- Avoid using the word "{emotion}" itself; show the emotion through description
- Vary the setting (work, family, relationships, academics, adventure, everyday life, crisis)

Write ONLY the passage, no meta-commentary."""


def generate_emotion_stories(
    emotion: str,
    n: int = 50,
    retry_delay: float = 2.0,
    inter_request_delay: float = 0.5,
) -> list[str]:
    """Generate n stories for a single emotion using Claude API.

    Stories are requested one at a time so we can handle failures per sample.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    stories = []
    prompt = EMOTION_STORY_PROMPT_TEMPLATE.format(emotion=emotion)

    for i in range(n):
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                stories.append(text)
                break
            except anthropic.RateLimitError:
                wait = retry_delay * (2 ** attempt)
                print(f"  Rate limit hit (story {i+1}/{n}), waiting {wait:.0f}s...")
                time.sleep(wait)
            except Exception as e:
                print(f"  Error on story {i+1}/{n} attempt {attempt+1}: {e}")
                time.sleep(retry_delay)
        else:
            print(f"  WARNING: Failed to generate story {i+1} for '{emotion}' after 3 attempts")

        if i < n - 1:
            time.sleep(inter_request_delay)

    return stories
