#!/usr/bin/env python3
"""Intrinsic evaluation of EMGI warp accuracy against the GT ego-motion.

The scene is static (pods filmed from a moving camera), so the GT box centers move ONLY
because of the camera. That makes the GT itself a reference for ego-motion: the true
position of an object after K frames is simply its GT center at frame t+K. We therefore
propagate each object with the estimated warps and measure how far it lands from its GT
position — at 1 frame, and after chaining K warps.

For each method and sequence we report:
  * 1-frame center-displacement error (mean, median), in pixels;
  * accumulated error after chaining K warps, K in {1,4,8,15,30,60};
  * fraction of pairs where the method failed / returned identity;
  * compute time per pair (median, ms).

The error-vs-K curve is the decision criterion: a matcher whose curve stays flat allows a
larger EMGI `max_gap`.

No tracking / TrackEval is run here — this is warp accuracy only.

Example:
    python tools/warp_eval.py \\
        --seqs data/synthetic/seq_0001 data/real/seq_0007 \\
        --methods sparseOptFlow_nomask sparseOptFlow loftr disk_lightglue xfeat \\
        --out warp_eval
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from cocoamot.io import read_det_boxes
from cocoamot.warps.estimators import build_exclusion_mask, make_estimator

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_KS = [1, 4, 8, 15, 30, 60]


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------
def list_frames(img_dir: str) -> list[str]:
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    return [os.path.join(img_dir, f) for f in files]


def read_gt_centers(gt_txt: str, min_vis: float = 0.0) -> dict[int, dict[int, np.ndarray]]:
    """Read MOT gt.txt -> per-frame {id: center xy}. Skips rows with the consider-flag off."""
    per_frame: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    with open(gt_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            fr, tid = int(float(p[0])), int(float(p[1]))
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            consider = float(p[6]) if len(p) > 6 else 1.0
            vis = float(p[8]) if len(p) > 8 else 1.0
            if consider == 0 or vis < min_vis:
                continue
            per_frame[fr][tid] = np.array([x + w / 2.0, y + h / 2.0])
    return per_frame


# --------------------------------------------------------------------------------------
# Warp computation with per-pair diagnostics (cached to disk)
# --------------------------------------------------------------------------------------
def affine_batch(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to an (M, 2) array of points."""
    return pts @ H[:, :2].T + H[:, 2]


def get_warps(
    seq: str, method: str, downscale: int, dilate: float, device: str | None, max_frames: int | None, cache_root: str | None = None
):
    """Return (steps, ok, times_ms) for a sequence+method, computing and caching as needed.

    steps[j] maps frame (j+1) -> frame (j+2). ok[j] is False on identity fallback.
    times_ms[j] is the wall time for that pair. Masking is on unless the method is *_nomask.
    """
    img_dir = os.path.join(seq, "img1")
    files = list_frames(img_dir)
    if max_frames:
        files = files[:max_frames]
    n = len(files)
    masked = not method.endswith("_nomask")
    tag = f"_m{int(round(dilate * 100))}" if masked else ""
    mf = f"_f{max_frames}" if max_frames else ""
    cache_dir = os.path.join(cache_root, os.path.basename(seq.rstrip("/"))) if cache_root else seq
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"emgi_warpeval_{method}{tag}_ds{downscale}{mf}.npz")

    if os.path.isfile(cache):
        d = np.load(cache)
        if int(d["nframes"]) == n:
            return d["steps"], d["ok"], d["times_ms"]

    det_txt = os.path.join(seq, "det", "det.txt")
    dets = read_det_boxes(det_txt) if (masked and os.path.isfile(det_txt)) else {}
    est = make_estimator(method, downscale=downscale, device=device)

    steps = np.zeros((n - 1, 2, 3))
    ok = np.zeros(n - 1, dtype=bool)
    times_ms = np.zeros(n - 1)
    prev = cv2.imread(files[0])
    h, w = prev.shape[:2]
    print(f"  [{method}] {n} frames, mask={masked} ...", flush=True)
    for j in range(n - 1):
        cur = cv2.imread(files[j + 1])
        m = build_exclusion_mask((h, w), dets.get(j + 1), dilate) if masked else None
        t0 = time.perf_counter()
        out = est.estimate(prev, cur, m)
        times_ms[j] = (time.perf_counter() - t0) * 1000.0
        steps[j] = out.H
        ok[j] = out.ok
        prev = cur
    np.savez(cache, steps=steps, ok=ok, times_ms=times_ms, nframes=np.int64(n))
    print(f"  [{method}] done, {(~ok).sum()}/{n - 1} identity fallbacks", flush=True)
    return steps, ok, times_ms


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def compute_errors(steps: np.ndarray, gt: dict[int, dict[int, np.ndarray]], Ks: list[int], n_frames: int):
    """Propagate GT objects through the warps and collect position errors at each K.

    Returns dict K -> np.ndarray of per-(object, start-frame) errors (pixels).
    """
    Kmax = max(Ks)
    Kset = set(Ks)
    errors: dict[int, list] = {k: [] for k in Ks}
    # Start frames are 1-indexed to match MOT / the warp indexing (steps[f-1]: f -> f+1).
    for t in range(1, n_frames):
        if t not in gt:
            continue
        ids = list(gt[t].keys())
        if not ids:
            continue
        pos = np.array([gt[t][i] for i in ids])  # (M, 2), propagated positions
        for k in range(1, Kmax + 1):
            f = t + k
            if f > n_frames:
                break
            pos = affine_batch(steps[f - 2], pos)  # warp frame (f-1) -> f
            if k in Kset and f in gt:
                gtf = gt[f]
                for idx, i in enumerate(ids):
                    if i in gtf:
                        errors[k].append(float(np.linalg.norm(pos[idx] - gtf[i])))
    return {k: np.array(v) for k, v in errors.items()}


def median_gt_motion(gt: dict[int, dict[int, np.ndarray]], n_frames: int) -> float:
    """Median per-frame GT center displacement magnitude (how much the camera moves)."""
    mags = []
    for t in range(1, n_frames):
        if t not in gt or (t + 1) not in gt:
            continue
        common = set(gt[t]) & set(gt[t + 1])
        for i in common:
            mags.append(np.linalg.norm(gt[t + 1][i] - gt[t][i]))
    return float(np.median(mags)) if mags else float("nan")


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Intrinsic EMGI warp-accuracy evaluation (GT ego-motion).")
    ap.add_argument("--seqs", nargs="+", required=True, help="Sequence dirs (each with img1/, gt/gt.txt, det/det.txt)")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["sparseOptFlow_nomask", "sparseOptFlow", "orb", "loftr", "disk_lightglue", "xfeat"],
    )
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--dilate", type=float, default=0.10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_frames", type=int, default=None, help="Cap frames per sequence (speed; default: full)")
    ap.add_argument("--Ks", type=int, nargs="+", default=DEFAULT_KS)
    ap.add_argument("--min_vis", type=float, default=0.0, help="Min GT visibility to include an object")
    ap.add_argument("--out", default="warp_eval", help="Output basename (.csv and .md)")
    ap.add_argument("--cache_root", default=None, help="Cache warps in <cache_root>/<seq>/ (default: the sequence folder)")
    return ap.parse_args()


def main():
    args = parse_args()
    rows = []  # dicts for CSV/markdown
    for seq in args.seqs:
        gt = read_gt_centers(os.path.join(seq, "gt", "gt.txt"), args.min_vis)
        files = list_frames(os.path.join(seq, "img1"))
        n = min(len(files), args.max_frames) if args.max_frames else len(files)
        motion = median_gt_motion(gt, n)
        print(f"\n=== {seq}  ({n} frames, median GT motion {motion:.2f} px/frame) ===", flush=True)
        for method in args.methods:
            steps, ok, times_ms = get_warps(seq, method, args.downscale, args.dilate, args.device, args.max_frames, args.cache_root)
            errs = compute_errors(steps, gt, args.Ks, n)
            row = {
                "seq": os.path.basename(seq.rstrip("/")),
                "method": method,
                "motion_px": round(motion, 2),
                "fail_frac": round(float((~ok).mean()), 4),
                "ms_per_pair": round(float(np.median(times_ms)), 2),
                "err1_mean": round(float(errs[1].mean()), 3) if len(errs[1]) else float("nan"),
                "err1_med": round(float(np.median(errs[1])), 3) if len(errs[1]) else float("nan"),
            }
            for k in args.Ks:
                e = errs[k]
                row[f"errK{k}_med"] = round(float(np.median(e)), 3) if len(e) else float("nan")
            rows.append(row)
            print(
                f"  {method:22s} fail={row['fail_frac']:.3f} {row['ms_per_pair']:7.1f}ms/pair "
                f"err1(med)={row['err1_med']:.2f}  "
                + "  ".join(f"K{k}={row[f'errK{k}_med']:.1f}" for k in args.Ks),
                flush=True,
            )

    # Write CSV
    import csv

    keys = list(rows[0].keys())
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # Write a per-sequence markdown table
    Kcols = [f"errK{k}_med" for k in args.Ks]
    with open(args.out + ".md", "w") as f:
        f.write("# EMGI warp-accuracy evaluation (median center error, px)\n\n")
        by_seq = defaultdict(list)
        for r in rows:
            by_seq[r["seq"]].append(r)
        for seq, rs in by_seq.items():
            f.write(f"## {seq}  (median GT motion {rs[0]['motion_px']} px/frame)\n\n")
            f.write("| method | fail% | ms/pair | err@1 | " + " | ".join(f"K={k}" for k in args.Ks) + " |\n")
            f.write("|---|---|---|---|" + "---|" * len(args.Ks) + "\n")
            for r in rs:
                f.write(
                    f"| {r['method']} | {r['fail_frac'] * 100:.1f} | {r['ms_per_pair']:.0f} | "
                    f"{r['err1_med']:.2f} | " + " | ".join(f"{r[c]:.1f}" for c in Kcols) + " |\n"
                )
            f.write("\n")
    print(f"\n[done] wrote {args.out}.csv and {args.out}.md", flush=True)


if __name__ == "__main__":
    main()
