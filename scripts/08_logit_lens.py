"""
Script: 08_logit_lens.py

Project each emotion vector through the lm_head (unembedding matrix) to see
which tokens lie in the same direction in vocabulary space.

Usage:
    python scripts/08_logit_lens.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emotion_probe.config import load_config
from emotion_probe.probe.activation_recorder import _get_model_and_tokenizer

EMOTIONS = [
    "desperate", "calm", "sad", "happy", "nervous", "angry",
    "afraid", "guilty", "surprised", "loving", "inspired", "proud",
]
TOP_K = 10


def decode_tokens(tokenizer, ids: torch.Tensor) -> list[str]:
    return [repr(tokenizer.decode([i.item()])) for i in ids]


def main():
    cfg = load_config()
    print("Loading model…")
    lm, tokenizer = _get_model_and_tokenizer(cfg)

    # Unembedding matrix: (vocab_size, hidden_dim)
    # nnsight keeps weights as meta tensors; a single trace pass materialises them.
    print("Extracting lm_head weights via trace…")
    _enc = tokenizer("Hello", return_tensors="pt")
    with lm.trace(_enc["input_ids"].to("cuda"), remote=False):
        _w_save = lm.lm_head.weight.save()
    W = (_w_save if isinstance(_w_save, torch.Tensor) else _w_save.value)
    W = W.detach().float().cpu()   # (vocab_size, hidden_dim)
    print(f"lm_head.weight shape: {W.shape}\n")

    vectors = torch.load(PROJECT_ROOT / "data" / "emotion_vectors.pt", weights_only=True)

    rows = []
    for emotion in EMOTIONS:
        vec = vectors[emotion].float()
        vec_unit = vec / vec.norm().clamp(min=1e-8)

        logits = W @ vec_unit   # (vocab_size,)

        top_ids  = logits.topk(TOP_K).indices
        bot_ids  = logits.topk(TOP_K, largest=False).indices

        top_tokens = decode_tokens(tokenizer, top_ids)
        bot_tokens = decode_tokens(tokenizer, bot_ids)

        rows.append((emotion, top_tokens, bot_tokens))

    # ------------------------------------------------------------------ #
    # Print results
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print(f"Logit Lens: emotion vectors → vocabulary space  (top/bottom {TOP_K})")
    print("=" * 70)

    for emotion, top_tokens, bot_tokens in rows:
        print(f"\n{'─'*60}")
        print(f"  {emotion.upper()}")
        print(f"  ↑ top {TOP_K}: {', '.join(top_tokens[:5])}")
        print(f"           {', '.join(top_tokens[5:])}")
        print(f"  ↓ bot {TOP_K}: {', '.join(bot_tokens[:5])}")
        print(f"           {', '.join(bot_tokens[5:])}")

    # ------------------------------------------------------------------ #
    # Compact summary table (top-5 only)
    # ------------------------------------------------------------------ #
    print(f"\n\n{'=' * 70}")
    print("SUMMARY TABLE  (top 5 tokens per emotion)")
    print("=" * 70)
    print(f"{'emotion':<12}  {'top 5 tokens (↑)':<45}  {'bottom 5 tokens (↓)'}")
    print("-" * 100)
    for emotion, top_tokens, bot_tokens in rows:
        t = "  ".join(top_tokens[:5])
        b = "  ".join(bot_tokens[:5])
        print(f"{emotion:<12}  {t:<45}  {b}")


if __name__ == "__main__":
    main()
