"""Record residual stream activations from Qwen3-4B using nnsight.

Verified behavior (nnsight 0.6.x, local mode):
  - lm.model.layers[i].output[0] shape: (seq_len, hidden_dim)  [no batch dim]
  - .save() returns torch.Tensor directly (no .value needed)
  - remote=False required to avoid ndif.us remote execution
"""

import torch
import numpy as np
from pathlib import Path
from nnsight import LanguageModel
from transformers import AutoTokenizer

from emotion_probe.config import load_config, HF_TOKEN

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "activations"
ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

_lm: LanguageModel | None = None
_tokenizer = None


def _get_model_and_tokenizer(cfg: dict):
    global _lm, _tokenizer
    if _lm is None:
        model_name = cfg["model"]["name"]
        print(f"Loading model: {model_name}")
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=HF_TOKEN, trust_remote_code=True
        )
        _lm = LanguageModel(
            model_name,
            device_map="cuda",
            dtype=torch.bfloat16,
            token=HF_TOKEN,
            trust_remote_code=True,
        )
        print("Model loaded.\n")
    return _lm, _tokenizer


def record_activations(
    texts: list[str],
    cfg: dict,
) -> np.ndarray:
    """Run texts through the model and return mean residual stream vectors.

    Args:
        texts: list of plain text strings (no ChatML wrapper)
        cfg: loaded config dict

    Returns:
        activations: shape (len(texts), hidden_dim) in float32
                     Texts with fewer than skip_first_n_tokens are skipped
                     and replaced with np.nan rows.
    """
    lm, tokenizer = _get_model_and_tokenizer(cfg)
    layer_idx: int = cfg["extraction"]["layer"]
    skip_n: int = cfg["extraction"]["skip_first_n_tokens"]

    results = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"]
        seq_len = input_ids.shape[1]

        if seq_len <= skip_n:
            print(f"  WARN: seq_len={seq_len} <= skip_n={skip_n}, skipping")
            # placeholder — caller can filter nan rows
            results.append(np.full(2560, np.nan, dtype=np.float32))
            continue

        with lm.trace(input_ids.to("cuda"), remote=False):
            hidden_save = lm.model.layers[layer_idx].output[0].save()

        # nnsight 0.6.x local: .save() returns torch.Tensor directly
        hidden = hidden_save if isinstance(hidden_save, torch.Tensor) else hidden_save.value
        # hidden shape: (seq_len, hidden_dim)
        vec = hidden[skip_n:, :].detach().float().cpu().mean(dim=0).numpy()  # (hidden_dim,)
        results.append(vec)

        # Free GPU memory eagerly
        del hidden_save, hidden
        torch.cuda.empty_cache()

    return np.stack(results, axis=0)  # (n_texts, hidden_dim)


def record_emotion_activations(cfg: dict, force: bool = False) -> None:
    """Record activations for all emotion stories and save per-emotion .pt files.

    Skips emotions whose .pt file already exists (resume-safe).
    """
    import json
    stories_path = PROJECT_ROOT / "data" / "stories" / "emotion_stories.json"
    with open(stories_path) as f:
        emotion_stories: dict[str, list[str]] = json.load(f)

    emotions: list[str] = cfg["emotions"]

    for emotion in emotions:
        out_path = ACTIVATIONS_DIR / f"{emotion}.pt"
        if out_path.exists() and not force:
            print(f"  '{emotion}': already exists, skipping")
            continue

        stories = emotion_stories.get(emotion, [])
        print(f"  '{emotion}': recording {len(stories)} stories...")
        acts = record_activations(stories, cfg)

        # Filter nan rows (texts that were too short)
        valid_mask = ~np.isnan(acts).any(axis=1)
        if not valid_mask.all():
            print(f"    WARNING: {(~valid_mask).sum()} stories skipped (too short)")
        acts = acts[valid_mask]

        torch.save(torch.tensor(acts, dtype=torch.float32), out_path)
        print(f"    Saved {acts.shape} → {out_path.name}")


def record_neutral_activations(cfg: dict, force: bool = False) -> None:
    """Record activations for neutral texts and save neutral.pt."""
    import json
    neutral_path = PROJECT_ROOT / "data" / "stories" / "neutral_texts.json"
    with open(neutral_path) as f:
        neutral_texts: list[dict] = json.load(f)

    out_path = ACTIVATIONS_DIR / "neutral.pt"
    if out_path.exists() and not force:
        print(f"  neutral: already exists, skipping")
        return

    texts = [t["text"] for t in neutral_texts]
    print(f"  neutral: recording {len(texts)} texts...")
    acts = record_activations(texts, cfg)

    valid_mask = ~np.isnan(acts).any(axis=1)
    if not valid_mask.all():
        print(f"    WARNING: {(~valid_mask).sum()} neutral texts skipped (too short)")
    acts = acts[valid_mask]

    torch.save(torch.tensor(acts, dtype=torch.float32), out_path)
    print(f"    Saved {acts.shape} → {out_path.name}")
