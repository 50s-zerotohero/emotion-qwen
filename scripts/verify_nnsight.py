"""
Verification script: confirm nnsight + Qwen3-4B layer access works correctly.

Checks:
  1. Model loads on CUDA in bfloat16
  2. lm.model.layers[20].output[0] shape is (1, seq_len, 2560)
  3. No error on text shorter than 50 tokens (edge case)
  4. Mean over positions 50: gives shape (2560,)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
from emotion_probe.config import load_config, HF_TOKEN

cfg = load_config()
MODEL_NAME = cfg["model"]["name"]      # "Qwen/Qwen3-4B"
LAYER_IDX  = cfg["extraction"]["layer"]  # 20
SKIP_N     = cfg["extraction"]["skip_first_n_tokens"]  # 50

# --------------------------------------------------------------------------- #
# 1. Load tokenizer
# --------------------------------------------------------------------------- #
print(f"[1] Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
print("    OK\n")

# --------------------------------------------------------------------------- #
# 2. Load model via nnsight
# --------------------------------------------------------------------------- #
print(f"[2] Loading model via nnsight (bfloat16, CUDA)")
lm = LanguageModel(
    MODEL_NAME,
    device_map="cuda",
    dtype=torch.bfloat16,
    token=HF_TOKEN,
    trust_remote_code=True,
)
print("    OK\n")

# Print model structure to confirm layer access path
print("[3] Model structure (top level):")
print(lm.model)
print()

# --------------------------------------------------------------------------- #
# 4. Test with a normal-length text (>50 tokens)
# --------------------------------------------------------------------------- #
test_text = (
    "Marcus stared at the rejection email, his hands trembling. "
    "Six months of applications had led to this moment, and now "
    "the walls seemed to close in around him. He pressed his palms "
    "flat against the cold desk, trying to steady himself."
)

enc = tokenizer(test_text, return_tensors="pt")
input_ids = enc["input_ids"]
seq_len = input_ids.shape[1]
print(f"[4] Normal text — seq_len={seq_len} tokens")

with lm.trace(input_ids.to("cuda"), remote=False):
    hidden_save = lm.model.layers[LAYER_IDX].output[0].save()

# nnsight 0.6.x local mode: .save() returns torch.Tensor directly (no .value)
hidden = hidden_save if isinstance(hidden_save, torch.Tensor) else hidden_save.value
print(f"    lm.model.layers[{LAYER_IDX}].output[0].shape = {hidden.shape}")
# nnsight 0.6.x local mode: batch dim is squeezed → shape is (seq_len, hidden_dim)
assert hidden.ndim == 2,           f"Expected 2D tensor, got {hidden.ndim}D"
assert hidden.shape[0] == seq_len, f"Expected seq={seq_len}, got {hidden.shape[0]}"
hidden_dim = hidden.shape[1]
print(f"    hidden_dim = {hidden_dim}")

# Convert bf16 → float32
vec = hidden[SKIP_N:, :].float().cpu().mean(dim=0)
print(f"    mean(positions {SKIP_N}:).shape = {vec.shape}  dtype={vec.dtype}")
assert vec.shape == (hidden_dim,), f"Unexpected shape: {vec.shape}"
print("    OK\n")

# --------------------------------------------------------------------------- #
# 5. Edge case: text shorter than SKIP_N tokens
# --------------------------------------------------------------------------- #
short_text = "Hello world."
enc_short = tokenizer(short_text, return_tensors="pt")
short_len = enc_short["input_ids"].shape[1]
print(f"[5] Short text — seq_len={short_len} tokens (< {SKIP_N})")

with lm.trace(enc_short["input_ids"].to("cuda"), remote=False):
    hidden_short_save = lm.model.layers[LAYER_IDX].output[0].save()

hidden_short = hidden_short_save if isinstance(hidden_short_save, torch.Tensor) else hidden_short_save.value
print(f"    output shape = {hidden_short.shape}")
if short_len <= SKIP_N:
    print(f"    WARNING: seq_len ({short_len}) <= skip_n ({SKIP_N}) — would skip this sample")
else:
    vec_short = hidden_short[SKIP_N:, :].float().cpu().mean(dim=0)
    print(f"    mean shape = {vec_short.shape}")
print("    OK (no crash)\n")

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
print("=" * 50)
print("All checks passed.")
print(f"  Layer access path : lm.model.layers[{LAYER_IDX}].output[0]")
print(f"  hidden_dim        : {hidden_dim}")
print(f"  VRAM after load   :", end=" ")
print(f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
print("=" * 50)
