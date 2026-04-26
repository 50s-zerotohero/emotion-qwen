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
    system_prompt: str = "",
) -> torch.Tensor:
    """Build tokenized input_ids with appropriate assistant prefix for each mode."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    def _enc_to_tensor(enc) -> torch.Tensor:
        """Normalize apply_chat_template output to (1, seq_len) tensor."""
        t = enc["input_ids"] if hasattr(enc, "__getitem__") and not isinstance(enc, torch.Tensor) else enc
        return t.unsqueeze(0) if t.ndim == 1 else t

    if mode == ReasoningMode.NO_THINK:
        # enable_thinking=False makes the template inject <think>\n\n</think>\n\n
        # as the assistant prefix, so the model skips the thinking phase entirely.
        enc = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", enable_thinking=False,
        )
        return _enc_to_tensor(enc)

    elif mode == ReasoningMode.THINK:
        # Standard Qwen3 template: model generates <think>...</think> naturally.
        enc = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        )
        return _enc_to_tensor(enc)

    elif mode == ReasoningMode.SCRATCHPAD:
        # Inject <think></think> (via enable_thinking=False) then open scratchpad.
        base = _enc_to_tensor(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", enable_thinking=False,
        ))
        scratchpad_prefix = tokenizer.encode(
            "<SCRATCHPAD_REASONING>\n", add_special_tokens=False,
        )
        prefix_t = torch.tensor([scratchpad_prefix], dtype=torch.long)
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
        system_prompt: str = "",
        steering_emotion: str | None = None,
        steering_alpha: float = 0.0,
    ) -> AsyncIterator[TokenWithEmotions]:
        # Normalize emotion vectors to unit length → cosine similarity probe.
        ev_cpu = {
            e: (v / v.norm().clamp(min=1e-8)).float().cpu()
            for e, v in emotion_vectors.items()
        }

        # Precompute steering vector on CUDA (outside the trace loop for efficiency)
        sv_gpu: torch.Tensor | None = None
        do_steer = steering_emotion and steering_alpha != 0.0 and steering_emotion in ev_cpu
        if do_steer:
            sv_gpu = ev_cpu[steering_emotion].to("cuda")  # (hidden_dim,)

        # Load model lazily
        await asyncio.to_thread(self._ensure_loaded)
        lm        = self._lm
        tokenizer = self._tokenizer
        layer_idx = self._layer_idx

        eos_id = tokenizer.eos_token_id
        cfg_temp: float = self._cfg["reasoning"]["temperature"]

        # Build prompt (1, prompt_len)
        input_ids = _build_input_ids(tokenizer, user_message, mode, system_prompt).to("cuda")

        section = _initial_section(mode)
        marker_buffer = ""   # rolling window for multi-token boundary detection

        with torch.no_grad():
            for _step in range(max_new_tokens):
                # --- single forward pass capturing hidden states + logits ---
                with lm.trace(input_ids, remote=False):
                    hidden_save = lm.model.layers[layer_idx].output[0].save()

                    if do_steer:
                        # Scale alpha by hidden-state norm so the slider is intuitive:
                        #   effective Δ = alpha × (hidden_norm / 10)
                        _h_norm   = lm.model.layers[layer_idx].output[0][-1, :].norm()
                        _effective = steering_alpha * _h_norm / 10.0
                        lm.model.layers[layer_idx].output[0][:] = (
                            lm.model.layers[layer_idx].output[0]
                            + _effective * sv_gpu
                        )

                    logit_save = lm.lm_head.output.save()

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
                # Normalize hidden state → cosine similarity against each probe vector.
                vec = hidden[-1, :].detach().float().cpu()
                vec = vec / vec.norm().clamp(min=1e-8)
                scores = {e: float(torch.dot(vec, ev)) for e, ev in ev_cpu.items()}

                # --- section tracking ---
                marker_buffer += token_str
                if len(marker_buffer) > 80:
                    marker_buffer = marker_buffer[-40:]

                if section == "response" and mode == ReasoningMode.THINK:
                    # THINK mode: model may emit <think> before thinking content
                    if "<think>" in marker_buffer:
                        section = "think"
                        marker_buffer = ""
                elif section == "think" and _THINK_END in marker_buffer:
                    section = "response"
                    marker_buffer = ""
                elif section == "scratchpad" and _SCRATCHPAD_END in marker_buffer:
                    section = "response"
                    marker_buffer = ""

                # --- stop on EOS without yielding the EOS token itself ---
                if next_id == eos_id:
                    break

                yield TokenWithEmotions(token=token_str, section=section, emotions=scores)

                # --- append token and continue ---
                next_tensor = torch.tensor([[next_id]], device="cuda")
                input_ids   = torch.cat([input_ids, next_tensor], dim=1)

                # yield control to event loop between tokens
                await asyncio.sleep(0)

        del hidden_save, hidden, logit_save, logits
        torch.cuda.empty_cache()
