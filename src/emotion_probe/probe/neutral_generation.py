"""Neutral dialogue generation via Claude API (paper-compliant)."""
import re
import time
import anthropic
from emotion_probe.config import ANTHROPIC_API_KEY

CLAUDE_MODEL = "claude-sonnet-4-6"

NEUTRAL_TOPICS = [
    "Setting up a development environment",
    "Converting units of measurement",
    "Sorting a list alphabetically",
    "Calculating compound interest",
    "Explaining HTTP status codes",
    "Writing a SQL query",
    "Setting up a cron job",
    "Understanding binary numbers",
    "Formatting dates in different locales",
    "Parsing JSON data",
    "How photosynthesis works",
    "Capital cities of countries",
    "Converting Celsius to Fahrenheit",
    "The speed of light",
    "Calculating area of shapes",
    "How to change a tire",
    "Setting up a VPN",
    "How to make a spreadsheet formula",
    "Sorting algorithms explained",
    "How DNS works",
]

SYSTEM_PROMPT_TEMPLATE = """\
Write {n_stories} different dialogues based on the following topic.
Topic: {topic}
The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)
The Person asks the AI a question or requests help with a task, \
and the AI provides a helpful response.
The first speaker turn should always be from Person.
Format the dialogues like so:
[optional system instructions]
Person: [line]
AI: [line]
Person: [line]
AI: [line]
[continue for 2-6 exchanges]
[dialogue 2]
etc.
IMPORTANT: Always put a blank line before each speaker turn.
Generate a diverse mix of dialogue types across the {n_stories} examples:
- Some, but not all should include a system prompt at the start
- Some should be about code or programming tasks
- Some should be factual questions (science, history, math, geography)
- Some should be work-related tasks (writing, analysis, summarization)
- Some should be practical how-to questions
- Some should be creative but neutral tasks
CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.
- NO emotional content whatsoever
- No pleasantries ("I'd be happy to help", "Great question!", etc.)
- Focus purely on information exchange and task completion"""

USER_PROMPT_TEMPLATE = "Write {n_stories} dialogues for Topic: {topic}"


def parse_dialogues(text: str) -> list[str]:
    """Split [dialogue N] delimited response into individual dialogue strings."""
    parts = re.split(r'\[dialogue\s+\d+\]', text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def generate_neutral_texts(
    n_stories_per_topic: int = 5,
    n_batches: int = 2,
    retry_delay: float = 2.0,
    inter_request_delay: float = 0.5,
) -> list[dict]:
    """Generate neutral dialogues across all NEUTRAL_TOPICS.

    Fires n_batches passes over all topics, each requesting n_stories_per_topic
    dialogues per topic. Total: len(NEUTRAL_TOPICS) × n_stories_per_topic × n_batches
    = 20 × 5 × 2 = 200 dialogues, 40 API requests.

    Returns list of {"topic": str, "text": str}.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    results: list[dict] = []

    for batch in range(n_batches):
        for topic_idx, topic in enumerate(NEUTRAL_TOPICS):
            system = SYSTEM_PROMPT_TEMPLATE.format(
                n_stories=n_stories_per_topic,
                topic=topic,
            )
            user = USER_PROMPT_TEMPLATE.format(
                n_stories=n_stories_per_topic,
                topic=topic,
            )

            for attempt in range(3):
                try:
                    response = client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=2000,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    raw = response.content[0].text.strip()
                    parsed = parse_dialogues(raw)
                    if not parsed:
                        print(f"  WARNING: parse returned 0 dialogues — "
                              f"topic '{topic[:40]}' batch {batch+1}")
                    for d in parsed:
                        results.append({"topic": topic, "text": d})
                    break
                except anthropic.RateLimitError:
                    wait = retry_delay * (2 ** attempt)
                    print(f"  Rate limit (batch {batch+1}, topic '{topic[:40]}'), "
                          f"waiting {wait:.0f}s...")
                    time.sleep(wait)
                except Exception as e:
                    print(f"  Error batch {batch+1} topic '{topic[:40]}' attempt {attempt+1}: {e}")
                    time.sleep(retry_delay)
            else:
                print(f"  WARNING: all 3 attempts failed — "
                      f"topic '{topic[:40]}' batch {batch+1}, skipping")

            if topic_idx < len(NEUTRAL_TOPICS) - 1:
                time.sleep(inter_request_delay)

        print(f"  Batch {batch+1}/{n_batches} done ({len(results)} dialogues total so far)")

    return results
