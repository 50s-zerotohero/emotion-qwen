"""LocalNNSightBackend: token-by-token generation with emotion probing via nnsight.

Verified nnsight 0.6.x behavior:
  - lm.model.layers[i].output[0]: shape (seq_len, hidden_dim)  [no batch dim]
  - lm.lm_head.output:            shape (1, seq_len, vocab_size) [has batch dim]
  - remote=False required for local execution
  - .save() returns torch.Tensor directly (no .value)

Generation uses a manual loop (no KV cache) to allow per-token activation capture.
~1-2 sec/token is acceptable for this research prototype.
"""

import asyncio
import torch
from pathlib import Path
from typing import AsyncIterator

from nnsight import LanguageModel
from transformers import AutoTokenizer

from emotion_probe.config import load_config, HF_TOKEN
from emotion_probe.backend.base import EmotionProbeBackend, ReasoningMode, TokenWithEmotions

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Section boundary markers
_THINK_END        = "</think>"
_SCRATCHPAD_START = "<SCRATCHPAD_REASONING>"
_SCRATCHPAD_END   = "</SCRATCHPAD_REASONING>"


def _unwrap(t):
    """Handle both nnsight proxy (.value) and direct torch.Tensor from local trace."""
    return t if isinstance(t, torch.Tensor) else t.value


def _build_input_ids(
    tokenizer,
    user_message: str,
    mode: ReasoningMode,
) -> torch.Tensor:
    """Build tokenized input_ids with appropriate assistant prefix for each mode."""
    messages = [{"role": "user", "content": user_message}]

    # apply_chat_template adds <|im_start|>...<|im_end|> formatting
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # appends <|im_start|>assistant\n
        return_tensors="pt",
    )
    # apply_chat_template returns BatchEncoding or Tensor depending on version
    base = enc["input_ids"] if hasattr(enc, "__getitem__") and not isinstance(enc, torch.Tensor) else enc
    # ensure shape (1, seq_len)
    if base.ndim == 1:
        base = base.unsqueeze(0)

    if mode == ReasoningMode.NO_THINK:
        # Force model to skip thinking by pre-filling empty think block
        prefix = tokenizer.encode("<think></think>\n", add_special_tokens=False)
        prefix_t = torch.tensor([prefix], dtype=torch.long)
        return torch.cat([base, prefix_t], dim=1)

    elif mode == ReasoningMode.THINK:
        # Let model generate <think> naturally — no prefix
        return base

    elif mode == ReasoningMode.SCRATCHPAD:
        # Empty think + open scratchpad block for model to fill
        prefix = tokenizer.encode(
            "<think></think>\n<SCRATCHPAD_REASONING>\n",
            add_special_tokens=False,
        )
        prefix_t = torch.tensor([prefix], dtype=torch.long)
        return torch.cat([base, prefix_t], dim=1)

    raise ValueError(f"Unknown mode: {mode}")


def _initial_section(mode: ReasoningMode) -> str:
    if mode == ReasoningMode.THINK:
        return "think"
    if mode == ReasoningMode.SCRATCHPAD:
        return "scratchpad"
    return "response"  # NO_THINK: already past think block


class LocalNNSightBackend(EmotionProbeBackend):
    def __init__(self, cfg: dict | None = None):
        self._cfg = cfg or load_config()
        self._layer_idx: int = self._cfg["extraction"]["layer"]
        self._lm: LanguageModel | None = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._lm is not None:
            return
        model_name = self._cfg["model"]["name"]
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=HF_TOKEN, trust_remote_code=True
        )
        self._lm = LanguageModel(
            model_name,
            device_map="cuda",
            dtype=torch.bfloat16,
            token=HF_TOKEN,
            trust_remote_code=True,
        )

    def get_layer_for_probing(self) -> int:
        return self._layer_idx

    async def generate_with_emotions(
        self,
        user_message: str,
        mode: ReasoningMode,
        emotion_vectors: dict[str, torch.Tensor],
        max_new_tokens: int = 512,
    ) -> AsyncIterator[TokenWithEmotions]:
        # Move emotion vectors to CPU float32 once
        ev_cpu = {e: v.float().cpu() for e, v in emotion_vectors.items()}

        # Load model lazily
        await asyncio.to_thread(self._ensure_loaded)
        lm        = self._lm
        tokenizer = self._tokenizer
        layer_idx = self._layer_idx

        eos_id = tokenizer.eos_token_id
        cfg_temp: float = self._cfg["reasoning"]["temperature"]

        # Build prompt (1, prompt_len)
        input_ids = _build_input_ids(tokenizer, user_message, mode).to("cuda")

        section = _initial_section(mode)
        marker_buffer = ""   # rolling window for multi-token boundary detection

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # --- single forward pass capturing hidden states + logits ---
                with lm.trace(input_ids, remote=False):
                    hidden_save = lm.model.layers[layer_idx].output[0].save()
                    logit_save  = lm.lm_head.output.save()

                hidden = _unwrap(hidden_save)  # (seq_len, hidden_dim)
                logits = _unwrap(logit_save)   # (1, seq_len, vocab_size)
                if logits.ndim == 3:
                    logits = logits[0]         # (seq_len, vocab_size)

                # --- sample next token ---
                last_logits = logits[-1, :].detach().float().cpu()
                if cfg_temp <= 0.0:
                    next_id = int(last_logits.argmax())
                else:
                    probs   = torch.softmax(last_logits / cfg_temp, dim=-1)
                    next_id = int(torch.multinomial(probs, 1).item())

                token_str = tokenizer.decode([next_id])

                # --- compute emotion scores from last-position hidden state ---
                vec = hidden[-1, :].detach().float().cpu()  # (hidden_dim,)
                scores = {e: float(torch.dot(vec, ev)) for e, ev in ev_cpu.items()}

                # --- section tracking ---
                marker_buffer += token_str
                if len(marker_buffer) > 80:
                    marker_buffer = marker_buffer[-40:]

                if section == "think" and _THINK_END in marker_buffer:
                    section = "response"
                    marker_buffer = ""
                elif section == "scratchpad" and _SCRATCHPAD_END in marker_buffer:
                    section = "response"
                    marker_buffer = ""

                yield TokenWithEmotions(token=token_str, section=section, emotions=scores)

                # --- append token and check stop ---
                next_tensor = torch.tensor([[next_id]], device="cuda")
                input_ids   = torch.cat([input_ids, next_tensor], dim=1)

                if next_id == eos_id:
                    break

                # yield control to event loop between tokens
                await asyncio.sleep(0)

        del hidden_save, hidden, logit_save, logits
        torch.cuda.empty_cache()
