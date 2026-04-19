"""Noise removal via PCA projection on neutral text activations.

Implements the method from SPEC.md: compute top PCs of neutral activations
up to variance_explained threshold, then project them out of emotion vectors.
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "activations"


def compute_noise_basis(
    neutral_activations: np.ndarray,
    variance_explained: float = 0.5,
) -> np.ndarray:
    """Compute top PCs of neutral activations up to cumulative variance threshold.

    Args:
        neutral_activations: shape (n_samples, hidden_dim)
        variance_explained: cumulative variance ratio threshold

    Returns:
        basis: shape (k, hidden_dim), unit-normalized rows
    """
    pca = PCA()
    pca.fit(neutral_activations)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_explained) + 1)
    print(f"  PCA: keeping top {k} PCs (cumulative variance = {cumvar[k-1]:.3f})")
    basis = pca.components_[:k]
    # Ensure unit vectors (PCA components are already unit-normed, but just in case)
    norms = np.linalg.norm(basis, axis=1, keepdims=True)
    basis = basis / norms
    return basis.astype(np.float32)


def project_out(
    vectors: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Remove basis directions from vectors.

    Args:
        vectors: shape (n, d) or (d,)
        basis: shape (k, d), each row is a unit vector

    Returns:
        clean vectors, same shape as input
    """
    clean = vectors.copy().astype(np.float64)
    for u in basis.astype(np.float64):
        if clean.ndim == 2:
            projections = clean @ u          # (n,)
            clean = clean - np.outer(projections, u)
        else:
            projection = clean @ u           # scalar
            clean = clean - projection * u
    return clean.astype(np.float32)


def compute_and_save_noise_basis(cfg: dict, force: bool = False) -> np.ndarray:
    """Load neutral.pt, run PCA, save basis to neutral_pca_basis.pt.

    Returns the basis array.
    """
    basis_path = ACTIVATIONS_DIR / "neutral_pca_basis.pt"
    if basis_path.exists() and not force:
        print("  noise_basis: already exists, loading from disk")
        return torch.load(basis_path, weights_only=True).numpy()

    neutral_path = ACTIVATIONS_DIR / "neutral.pt"
    if not neutral_path.exists():
        raise FileNotFoundError(f"neutral.pt not found at {neutral_path}")

    neutral_acts = torch.load(neutral_path, weights_only=True).numpy()  # (200, hidden_dim)
    print(f"  neutral activations shape: {neutral_acts.shape}")

    variance_explained: float = cfg["noise_removal"]["variance_explained"]
    basis = compute_noise_basis(neutral_acts, variance_explained)

    torch.save(torch.tensor(basis, dtype=torch.float32), basis_path)
    print(f"  Saved basis {basis.shape} → {basis_path.name}")
    return basis
