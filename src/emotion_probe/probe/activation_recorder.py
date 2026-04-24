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


def wrap_as_assistant(text: str, tokenizer) -> tuple[str, int]:
    """Wrap story/neutral text as an assistant reply in ChatML format.

    This aligns the activation distribution with inference time (chat UI),
    where the model generates text as an assistant response.

    Returns:
        (wrapped_text, header_token_count) where header_token_count is the
        number of tokens in the ChatML header before the actual text begins.
    """
    messages = [{"role": "user", "content": "Write a short passage."}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # includes <|im_start|>assistant\n
    )
    header_token_count = len(tokenizer(header, return_tensors="pt").input_ids[0])
    return header + text, header_token_count


def record_activations(
    texts: list[str],
    cfg: dict,
) -> np.ndarray:
    """Run texts through the model and return mean residual stream vectors.

    Each text is wrapped as an assistant reply (ChatML format) to match the
    inference-time distribution. The mean is taken over positions
    [header_tokens + skip_first_n_tokens : end] to skip both the ChatML
    header and the first N content tokens (per paper: emotion content becomes
    clear after token 50).

    Args:
        texts: list of plain text strings (ChatML wrapping applied here)
        cfg: loaded config dict

    Returns:
        activations: shape (len(texts), hidden_dim) in float32.
                     Texts with fewer usable tokens are replaced with nan rows.
    """
    lm, tokenizer = _get_model_and_tokenizer(cfg)
    layer_idx: int = cfg["extraction"]["layer"]
    skip_n: int = cfg["extraction"]["skip_first_n_tokens"]

    # Pre-compute header token count once (same for all texts)
    _, header_n = wrap_as_assistant("", tokenizer)

    results = []
    for text in texts:
        chat_text, _ = wrap_as_assistant(text, tokenizer)
        enc = tokenizer(chat_text, return_tensors="pt")
        input_ids = enc["input_ids"]
        seq_len = input_ids.shape[1]

        # Skip header tokens + first skip_n content tokens
        skip_total = header_n + skip_n

        if seq_len <= skip_total:
            print(f"  WARN: seq_len={seq_len} <= skip_total={skip_total} "
                  f"(header={header_n} + content_skip={skip_n}), skipping")
            results.append(np.full(2560, np.nan, dtype=np.float32))
            continue

        with lm.trace(input_ids.to("cuda"), remote=False):
            hidden_save = lm.model.layers[layer_idx].output[0].save()

        hidden = hidden_save if isinstance(hidden_save, torch.Tensor) else hidden_save.value
        # Average over positions after header + first skip_n content tokens
        vec = hidden[skip_total:, :].detach().float().cpu().mean(dim=0).numpy()
        results.append(vec)

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
