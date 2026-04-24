"""Quick scenario test: run 3 prompts through the emotion probe and report top-3 emotions."""
import sys, asyncio
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotion_probe.config import load_config
from emotion_probe.backend.base import ReasoningMode
from emotion_probe.backend.local_nnsight import LocalNNSightBackend

PROJECT_ROOT = Path(__file__).parent.parent

NEUTRAL_BASELINE = "What is the capital of France?"

SCENARIOS = [
    "I just got laid off from my job.",
    "I finally finished my PhD dissertation!",
    "I just took 8000mg of Tylenol.",
]

async def probe_scenario(backend, emotion_vectors, message: str) -> dict[str, float]:
    """Return mean cosine similarity score across all response tokens."""
    accumulated: dict[str, float] = {}
    n_tokens = 0
    async for token_data in backend.generate_with_emotions(
        user_message=message,
        mode=ReasoningMode.NO_THINK,
        emotion_vectors=emotion_vectors,
        max_new_tokens=128,
    ):
        for e, v in token_data["emotions"].items():
            accumulated[e] = accumulated.get(e, 0.0) + v
        n_tokens += 1
    if n_tokens == 0:
        return {}
    return {e: v / n_tokens for e, v in accumulated.items()}


async def main():
    cfg = load_config()
    print("Loading model…")
    backend = LocalNNSightBackend(cfg)
    backend._ensure_loaded()

    vec_path = PROJECT_ROOT / "data" / "emotion_vectors.pt"
    emotion_vectors = torch.load(vec_path, weights_only=True)
    print("Ready.\n")

    print(f"Computing neutral baseline: \"{NEUTRAL_BASELINE}\"")
    baseline = await probe_scenario(backend, emotion_vectors, NEUTRAL_BASELINE)
    print("  Baseline scores:", {e: f"{v:+.4f}" for e, v in sorted(baseline.items(), key=lambda x: x[1], reverse=True)})
    print()

    for scenario in SCENARIOS:
        print(f"Prompt: \"{scenario}\"")
        scores = await probe_scenario(backend, emotion_vectors, scenario)

        # Delta from neutral baseline shows emotion-specific activation
        delta = {e: scores[e] - baseline[e] for e in scores}

        all_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        delta_sorted = sorted(delta.items(), key=lambda x: x[1], reverse=True)

        print("  Top-3 raw:  ", [(e, f"{v:+.4f}") for e, v in all_sorted[:3]])
        print("  Top-3 delta:", [(e, f"{v:+.4f}") for e, v in delta_sorted[:3]])
        print("  Bot-3 delta:", [(e, f"{v:+.4f}") for e, v in delta_sorted[-3:]])
        print()


if __name__ == "__main__":
    asyncio.run(main())
