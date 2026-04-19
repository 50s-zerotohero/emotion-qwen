"""
Script: 03_extract_vectors.py

Full pipeline:
  1. Record residual stream activations for all emotion stories (GPU)
  2. Record activations for neutral texts (GPU)
  3. Compute noise basis via PCA on neutral activations (CPU)
  4. Compute emotion vectors with noise removal (CPU)

Resume-safe: each emotion's .pt is saved immediately and skipped on re-run.

Usage:
    python scripts/03_extract_vectors.py [--force]

Options:
    --force   Re-compute even if output files already exist
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotion_probe.config import load_config
from emotion_probe.probe.activation_recorder import (
    record_emotion_activations,
    record_neutral_activations,
)
from emotion_probe.probe.noise_removal import compute_and_save_noise_basis
from emotion_probe.probe.emotion_vectors import compute_emotion_vectors

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="Re-compute existing files")
args = parser.parse_args()

cfg = load_config()

# --------------------------------------------------------------------------- #
# Step 1 & 2: Record activations (GPU)
# --------------------------------------------------------------------------- #
print("=" * 60)
print("Step 1: Recording emotion story activations")
print("=" * 60)
record_emotion_activations(cfg, force=args.force)

print()
print("=" * 60)
print("Step 2: Recording neutral text activations")
print("=" * 60)
record_neutral_activations(cfg, force=args.force)

# --------------------------------------------------------------------------- #
# Step 3: Noise basis (CPU)
# --------------------------------------------------------------------------- #
print()
print("=" * 60)
print("Step 3: Computing noise basis via PCA")
print("=" * 60)
noise_basis = compute_and_save_noise_basis(cfg, force=args.force)

# --------------------------------------------------------------------------- #
# Step 4: Emotion vectors (CPU)
# --------------------------------------------------------------------------- #
print()
print("=" * 60)
print("Step 4: Computing emotion vectors")
print("=" * 60)
emotion_vectors = compute_emotion_vectors(cfg, noise_basis)

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
print()
print("=" * 60)
print("DONE")
print(f"  {len(emotion_vectors)} emotion vectors computed")
for name, vec in emotion_vectors.items():
    print(f"    {name:<12} norm={vec.norm().item():.4f}")
print("=" * 60)
