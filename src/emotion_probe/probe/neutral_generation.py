"""Neutral text generation via Claude API."""
import time
import anthropic
from emotion_probe.config import ANTHROPIC_API_KEY

CLAUDE_MODEL = "claude-sonnet-4-6"

# 20 topics × 10 texts each = 200 neutral texts
NEUTRAL_TOPICS = [
    "The construction of the Panama Canal",
    "How photosynthesis works in plants",
    "The water cycle in Earth's atmosphere",
    "The formation of stalactites and stalagmites in caves",
    "How radio waves are transmitted and received",
    "The geography of the Sahara Desert",
    "The process of steel manufacturing",
    "How tidal forces affect ocean currents",
    "The structure of the periodic table of elements",
    "How earthquakes are measured on the Richter scale",
    "The migration patterns of Arctic terns",
    "How concrete is mixed and cured",
    "The geological layers of the Grand Canyon",
    "How glass is manufactured from silica sand",
    "The life cycle of a star from nebula to supernova",
    "How the human immune system produces antibodies",
    "The agricultural history of wheat cultivation",
    "How sonar works in submarines",
    "The formation of the Himalayan mountain range",
    "How paper is made from wood pulp",
]

NEUTRAL_TEXT_PROMPT = """Write a short factual passage (80-150 words) about the following topic:

Topic: {topic}

Requirements:
- Purely informational, encyclopedic tone
- NO emotional content, feelings, or judgments
- NO first-person perspective or interpersonal dynamics
- Avoid words that express emotions
- Write ONLY the passage, no meta-commentary."""


def generate_neutral_texts(
    n_per_topic: int = 10,
    retry_delay: float = 2.0,
    inter_request_delay: float = 0.5,
) -> list[dict]:
    """Generate neutral texts covering all NEUTRAL_TOPICS.

    Returns list of {"topic": str, "text": str}.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    results = []

    for topic in NEUTRAL_TOPICS:
        prompt = NEUTRAL_TEXT_PROMPT.format(topic=topic)
        for i in range(n_per_topic):
            for attempt in range(3):
                try:
                    response = client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=400,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.content[0].text.strip()
                    results.append({"topic": topic, "text": text})
                    break
                except anthropic.RateLimitError:
                    wait = retry_delay * (2 ** attempt)
                    print(f"  Rate limit hit (topic '{topic}' #{i+1}), waiting {wait:.0f}s...")
                    time.sleep(wait)
                except Exception as e:
                    print(f"  Error on topic '{topic}' #{i+1} attempt {attempt+1}: {e}")
                    time.sleep(retry_delay)
            else:
                print(f"  WARNING: Failed neutral text for topic '{topic}' #{i+1}")

            if i < n_per_topic - 1:
                time.sleep(inter_request_delay)

        print(f"  topic '{topic[:40]}': {n_per_topic} texts done")

    return results
