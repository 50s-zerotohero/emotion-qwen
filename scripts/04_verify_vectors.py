"""
Script: 04_verify_vectors.py

Verify emotion vectors by computing 12x12 cosine similarity matrices
for both raw (pre-projection) and noise-removed vectors.

Usage:
    python scripts/04_verify_vectors.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ACTIVATIONS_DIR = DATA_DIR / "activations"

# --------------------------------------------------------------------------- #
# Load vectors
# --------------------------------------------------------------------------- #
raw_dict: dict[str, torch.Tensor] = torch.load(
    ACTIVATIONS_DIR / "raw_emotion_means.pt", weights_only=True
)
final_dict: dict[str, torch.Tensor] = torch.load(
    DATA_DIR / "emotion_vectors.pt", weights_only=True
)

emotions = list(raw_dict.keys())
n = len(emotions)

def cosine_matrix(vec_dict: dict[str, torch.Tensor]) -> np.ndarray:
    vecs = torch.stack([vec_dict[e] for e in emotions])  # (12, 2560)
    norms = vecs.norm(dim=1, keepdim=True)
    normed = vecs / norms
    return (normed @ normed.T).numpy()

raw_cos  = cosine_matrix(raw_dict)
final_cos = cosine_matrix(final_dict)

# --------------------------------------------------------------------------- #
# Pretty-print helper
# --------------------------------------------------------------------------- #
def print_matrix(mat: np.ndarray, title: str):
    col_w = 8
    label_w = 11
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    # header
    header = " " * label_w + "".join(f"{e[:col_w-1]:>{col_w}}" for e in emotions)
    print(header)
    print("-" * len(header))
    for i, ei in enumerate(emotions):
        row = f"{ei:<{label_w}}"
        for j in range(n):
            val = mat[i, j]
            if i == j:
                row += f"{'---':>{col_w}}"
            else:
                row += f"{val:>{col_w}.3f}"
        print(row)

print_matrix(raw_cos,   "RAW (before noise removal)")
print_matrix(final_cos, "FINAL (after noise removal)")

# --------------------------------------------------------------------------- #
# Spot-check pairs
# --------------------------------------------------------------------------- #
def idx(e: str) -> int:
    return emotions.index(e)

PAIRS = {
    "opposing": [("happy","sad"), ("calm","desperate"), ("loving","angry")],
    "similar":  [("afraid","nervous"), ("happy","proud"), ("happy","inspired")],
}

print(f"\n{'='*60}")
print("  PAIR ANALYSIS")
print(f"{'='*60}")
for kind, pairs in PAIRS.items():
    print(f"\n  [{kind}]")
    print(f"  {'pair':<22} {'raw':>8}  {'final':>8}  {'delta':>8}")
    print(f"  {'-'*50}")
    for a, b in pairs:
        r = raw_cos[idx(a), idx(b)]
        f = final_cos[idx(a), idx(b)]
        d = f - r
        print(f"  {a+' / '+b:<22} {r:>8.4f}  {f:>8.4f}  {d:>+8.4f}")

# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def off_diag(mat):
    mask = ~np.eye(n, dtype=bool)
    return mat[mask]

for label, mat in [("RAW", raw_cos), ("FINAL", final_cos)]:
    vals = off_diag(mat)
    print(f"\n  {label} off-diagonal stats:")
    print(f"    mean={vals.mean():.4f}  std={vals.std():.4f}  "
          f"min={vals.min():.4f}  max={vals.max():.4f}")

# --------------------------------------------------------------------------- #
# Save heatmap images
# --------------------------------------------------------------------------- #
print("\nGenerating heatmaps...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Emotion Vector Cosine Similarity", fontsize=14)

for ax, mat, title in [
    (axes[0], raw_cos,   "Raw (before noise removal)"),
    (axes[1], final_cos, "Final (after noise removal)"),
]:
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(emotions, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(emotions, fontsize=9)
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
out = DATA_DIR / "activations" / "cosine_similarity.png"
plt.savefig(out, dpi=150)
print(f"Heatmap saved → {out}")
print("\nDONE")
