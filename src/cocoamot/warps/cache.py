"""Per-sequence camera-warp computation with an on-disk cache.

`steps` has shape (N-1, 2, 3); `steps[t-1]` maps a point in frame t to frame t+1 (1-indexed frames).
The cache file is named `emgi_warps_{method}{_m<dilate%> if masked}_ds{downscale}.npz` and stores a
signature of the estimation settings; a cache whose frame count or signature differs is recomputed.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ..io import list_frames, read_det_boxes
from .estimators import build_exclusion_mask, make_estimator


def warp_cache_name(method: str, downscale: int = 2, masked: bool = False, dilate: float = 0.10) -> str:
    """File name of the warp cache for a set of estimation settings."""
    tag = f"_m{int(round(dilate * 100))}" if masked else ""
    return f"emgi_warps_{method}{tag}_ds{downscale}.npz"


def compute_warps(
    img_dir: str,
    method: str = "sparseOptFlow",
    downscale: int = 2,
    cache_path: str | None = None,
    mask: bool = True,
    det_txt: str | None = None,
    dilate: float = 0.10,
    device: str | None = None,
    cache_dir: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Compute (or load from cache) the per-step camera warps of a sequence.

    Every method goes through the per-pair interface of `estimators` (`estimate(imgA, imgB, mask) -> 2x3`).
    The classic methods (sparseOptFlow, orb, sift, ecc) are bit-for-bit identical to the streaming tracker
    GMC when masking is disabled.

    Args:
        img_dir: sequence image folder (`<seq>/img1`).
        method: sparseOptFlow[_nomask], orb, sift, ecc, loftr, disk_lightglue, xfeat.
        downscale: working-resolution downscale factor.
        cache_path: explicit cache file; defaults to `<cache_dir>/<name>`.
        mask: exclude object regions (from `det_txt`, dilated by `dilate`); forced off for `*_nomask`.
        det_txt: MOT det.txt used for masking; defaults to `<seq>/det/det.txt`.
        dilate: fractional box enlargement for the mask (0.10 => +10%).
        device: torch device for the learned matchers.
        cache_dir: folder for the cache file; defaults to the sequence folder (parent of `img_dir`).
    """
    log = print if verbose else (lambda *a, **k: None)
    files = list_frames(img_dir)
    n = len(files)
    masked = mask and not method.endswith("_nomask")
    parent = os.path.dirname(os.path.normpath(img_dir))

    if masked and det_txt is None:
        cand = os.path.join(parent, "det", "det.txt")
        if os.path.isfile(cand):
            det_txt = cand
        else:
            log(f"[warps] masking requested but no det.txt found at {cand}; proceeding un-masked")
            masked = False

    if cache_path is None:
        cache_path = os.path.join(cache_dir or parent, warp_cache_name(method, downscale, masked, dilate))

    sig = f"{method}|ds{downscale}|mask={masked}|dilate={dilate}"
    if os.path.isfile(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        stored = str(data["sig"]) if "sig" in data else None
        # Without masking the dilation is irrelevant: accept caches written with any dilate value.
        prefix = f"{method}|ds{downscale}|mask=False|"
        same_sig = stored is None or stored == sig or (not masked and stored.startswith(prefix))
        if int(data["nframes"]) == n and same_sig:
            log(f"[warps] loaded cache {cache_path}  ({n} frames)")
            return data["steps"]
        log("[warps] cache mismatch (frames/sig); recomputing")

    dets = read_det_boxes(det_txt) if masked else {}
    est = make_estimator(method, downscale=downscale, device=device)
    log(f"[warps] estimating '{method}' (downscale={downscale}, mask={masked}) over {n} frames ...")

    steps = np.zeros((n - 1, 2, 3), dtype=np.float64)
    prev = cv2.imread(files[0])
    if prev is None:
        raise FileNotFoundError(f"Could not read image {files[0]}")
    img_h, img_w = prev.shape[:2]
    for j in range(n - 1):
        cur = cv2.imread(files[j + 1])
        if cur is None:
            raise FileNotFoundError(f"Could not read image {files[j + 1]}")
        m = build_exclusion_mask((img_h, img_w), dets.get(j + 1), dilate) if masked else None
        steps[j] = est.estimate(prev, cur, m).H  # warp frame (j+1) -> frame (j+2)
        prev = cur
        if (j + 2) % 100 == 0 or j + 2 == n:
            log(f"[warps]   {j + 2}/{n}   fails={est.fail_count}")

    log(f"[warps] done: {est.n_calls} pairs, {est.fail_count} identity fallbacks, {est.total_time:.1f}s")
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez(cache_path, steps=steps, nframes=np.int64(n), sig=sig)
    log(f"[warps] cached -> {cache_path}")
    return steps
