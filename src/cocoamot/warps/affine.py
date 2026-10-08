"""2x3 affine helpers for camera warps."""

from __future__ import annotations

import numpy as np


def apply_affine(H: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine warp to a single (x, y) point."""
    x, y = pt
    return np.array([H[0, 0] * x + H[0, 1] * y + H[0, 2], H[1, 0] * x + H[1, 1] * y + H[1, 2]])


def invert_affine(H: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine warp (returns a 2x3)."""
    M = np.eye(3)
    M[:2] = H
    return np.linalg.inv(M)[:2]
