"""
Script: 07_steering_poem_experiment.py

Generates poems with each of 12 emotion steerings applied, then evaluates
each poem's emotional tone via Claude API.

Usage:
    python scripts/07_steering_poem_experiment.py

Output:
    data/steering_poems/baseline.txt
    data/steering_poems/{emotion}.txt
    data/steering_poems/results.json
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import anthropic
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emotion_probe.config import load_config
from emotion_probe.backend.base import ReasoningMode
from emotion_probe.backend.local_nnsight import LocalNNSightBackend

PROMPT = "Please create a poem within 50 words."
ALPHA  = 5.0
MAX_NEW_TOKENS = 130   # generous budget; poem target is 50 words ≈ 80 tokens

EMOTIONS = [
    "desperate", "calm", "sad", "happy", "nervous", "angry",
    "afraid", "guilty", "surprised", "loving", "inspired", "proud",
]

POEMS_DIR = PROJECT_ROOT / "data" / "steering_poems"

# --------------------------------------------------------------------------- #
# Claude evaluation prompt
# --------------------------------------------------------------------------- #
_EVAL_SYSTEM = "You are a literary analyst. Analyze the emotional tone of a poem."

_EVAL_USER = """\
Analyze this poem and identify:
1. Top 3 emotional tones expressed (e.g., sad, melancholic, joyful)
2. Top 3 imagery/atmosphere words (e.g., dark, silent, bright)
3. Overall valence: positive / negative / neutral

Poem:
{poem}

Respond in JSON format only (no other text):
{{
  "emotional_tones": ["tone1", "tone2", "tone3"],
  "imagery": ["word1", "word2", "word3"],
  "valence": "positive/negative/neutral"
}}"""


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
async def generate_poem(
    backend: LocalNNSightBackend,
    vectors: dict,
    steering_emotion: str | None = None,
    alpha: float = 0.0,
) -> str:
    tokens = []
    async for td in backend.generate_with_emotions(
        user_message=PROMPT,
        mode=ReasoningMode.NO_THINK,
        emotion_vectors=vectors,
        max_new_tokens=MAX_NEW_TOKENS,
        steering_emotion=steering_emotion,
        steering_alpha=alpha,
    ):
        tokens.append(td["token"])
    return "".join(tokens)


# --------------------------------------------------------------------------- #
# Claude evaluation
# --------------------------------------------------------------------------- #
def evaluate_poem(client: anthropic.Anthropic, poem: str) -> dict:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=_EVAL_SYSTEM,
        messages=[{"role": "user", "content": _EVAL_USER.format(poem=poem)}],
    )
    text = resp.content[0].text.strip()
    # Strip markdown code fences if present
    if "```" in text:
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"emotional_tones": [], "imagery": [], "valence": "unknown", "_raw": text}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main():
    POEMS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    print("Loading model…")
    backend = LocalNNSightBackend(cfg)
    await asyncio.to_thread(backend._ensure_loaded)
    vectors = torch.load(PROJECT_ROOT / "data" / "emotion_vectors.pt", weights_only=True)
    print("Model loaded.\n")

    client = anthropic.Anthropic()

    # ------------------------------------------------------------------ #
    # Phase 1: generation
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("PHASE 1: Generating poems")
    print("=" * 60)

    poems: dict[str, str] = {}
    all_keys = ["baseline"] + EMOTIONS

    for key in all_keys:
        label = f"{key} (α={ALPHA})" if key != "baseline" else "baseline"
        print(f"Generating {label}…", end=" ", flush=True)
        em = None if key == "baseline" else key
        al = 0.0  if key == "baseline" else ALPHA
        poem = await generate_poem(backend, vectors, em, al)
        poems[key] = poem
        (POEMS_DIR / f"{key}.txt").write_text(poem, encoding="utf-8")
        print(f"{len(poem.split())} words")

    # ------------------------------------------------------------------ #
    # Phase 2: Claude evaluation
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("PHASE 2: Evaluating with Claude API")
    print("=" * 60)

    results: dict[str, dict] = {}
    for key in all_keys:
        print(f"Evaluating {key}…", end=" ", flush=True)
        results[key] = evaluate_poem(client, poems[key])
        tones = results[key].get("emotional_tones", [])
        print(f"{tones}")
        time.sleep(0.4)   # gentle rate-limiting

    # ------------------------------------------------------------------ #
    # Report table
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 80}")
    print("RESULTS TABLE")
    print("=" * 80)
    print(f"{'steering':<12}  {'emotional_tones':<36}  {'imagery':<33}  valence")
    print("-" * 90)
    for key in all_keys:
        r      = results[key]
        tones  = ", ".join(r.get("emotional_tones", []))
        im     = ", ".join(r.get("imagery", []))
        val    = r.get("valence", "?")
        print(f"{key:<12}  [{tones:<34}]  [{im:<31}]  {val}")

    # ------------------------------------------------------------------ #
    # Match rate
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 50}")
    print("STEERING → EMOTION MATCH RATE")
    print("=" * 50)

    matches = 0
    for emotion in EMOTIONS:
        tones_lower = [t.lower() for t in results[emotion].get("emotional_tones", [])]
        # match if the steered emotion name appears in or overlaps any tone word
        hit = any(emotion in t or t in emotion for t in tones_lower)
        mark = "✓" if hit else "✗"
        print(f"  {mark}  {emotion:<12}  steered tones: {tones_lower}")
        if hit:
            matches += 1

    rate = matches / len(EMOTIONS) * 100
    print(f"\n  Match rate: {matches}/{len(EMOTIONS)} = {rate:.1f}%")

    # ------------------------------------------------------------------ #
    # Save full results
    # ------------------------------------------------------------------ #
    out = POEMS_DIR / "results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"poems": poems, "evaluations": results}, f, indent=2, ensure_ascii=False)
    print(f"\nFull results → {out}")

    # ------------------------------------------------------------------ #
    # Print all poems
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("GENERATED POEMS")
    print("=" * 60)
    for key in all_keys:
        print(f"\n--- {key} ---")
        print(poems[key].strip())


if __name__ == "__main__":
    asyncio.run(main())
