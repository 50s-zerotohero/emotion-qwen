"""Compute and save emotion vectors from per-emotion activations.

Pipeline (per SPEC.md):
  1. Load per-emotion activations → mean per emotion  (n_emotions, hidden_dim)
  2. Subtract grand mean across all 12 emotions
  3. Save raw (pre-projection) means to raw_emotion_means.pt
  4. Apply noise basis projection
  5. Save final vectors to emotion_vectors.pt
"""

import torch
import numpy as np
from pathlib import Path

from emotion_probe.probe.noise_removal import project_out

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "activations"
DATA_DIR = PROJECT_ROOT / "data"


def compute_emotion_vectors(cfg: dict, noise_basis: np.ndarray) -> dict[str, torch.Tensor]:
    """Compute final emotion vectors (noise-removed).

    Args:
        cfg: loaded config dict
        noise_basis: shape (k, hidden_dim) from compute_and_save_noise_basis()

    Returns:
        dict mapping emotion name → 1D float32 tensor of shape (hidden_dim,)
    """
    emotions: list[str] = cfg["emotions"]

    # ---- Step 1: load per-emotion mean activations ---- #
    means = {}
    for emotion in emotions:
        act_path = ACTIVATIONS_DIR / f"{emotion}.pt"
        if not act_path.exists():
            raise FileNotFoundError(f"Missing activations: {act_path}")
        acts = torch.load(act_path, weights_only=True).numpy()  # (n_stories, hidden_dim)
        means[emotion] = acts.mean(axis=0)                       # (hidden_dim,)
        print(f"  '{emotion}': mean over {acts.shape[0]} stories → shape {means[emotion].shape}")

    mean_matrix = np.stack([means[e] for e in emotions], axis=0)  # (12, hidden_dim)

    # ---- Step 2: subtract grand mean (deviation from category mean) ---- #
    grand_mean = mean_matrix.mean(axis=0)                          # (hidden_dim,)
    mean_matrix_centered = mean_matrix - grand_mean                # (12, hidden_dim)

    # ---- Step 3: save raw (pre-projection) means ---- #
    raw_dict = {e: torch.tensor(mean_matrix_centered[i], dtype=torch.float32)
                for i, e in enumerate(emotions)}
    raw_path = ACTIVATIONS_DIR / "raw_emotion_means.pt"
    torch.save(raw_dict, raw_path)
    print(f"\n  Saved raw emotion means → {raw_path.name}")

    # ---- Step 4: project out noise basis ---- #
    if cfg["noise_removal"]["enabled"]:
        clean_matrix = project_out(mean_matrix_centered, noise_basis)  # (12, hidden_dim)
        print(f"  Noise removal: projected out {noise_basis.shape[0]} PCs")
    else:
        clean_matrix = mean_matrix_centered
        print("  Noise removal: disabled")

    # ---- Step 5: save final vectors ---- #
    final_dict = {e: torch.tensor(clean_matrix[i], dtype=torch.float32)
                  for i, e in enumerate(emotions)}
    final_path = DATA_DIR / "emotion_vectors.pt"
    torch.save(final_dict, final_path)
    print(f"  Saved final emotion vectors → {final_path.name}")

    return final_dict
