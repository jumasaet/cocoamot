"""EMGI — Ego-Motion Gap Interpolation.

Post-process that fills *closed* gaps of a MOTChallenge tracker output by propagating each box with the
camera motion (GMC warps) instead of a straight line. Targets are static (cacao pods) and only the camera
moves, so between two observations a box follows the accumulated camera warp.

Fill modes (for ablation):
  warp   : forward + backward propagation of the box center through the warps, fused by distance to
           each anchor (cancels the drift of chaining warps).
  lerp   : linear interpolation of the center, ignoring the warps (baseline).
  static : hold the last-seen box from the first anchor (baseline).

In every mode the box size is linearly interpolated between the two anchors (object depth, not camera
motion). The tracker is not touched; EMGI operates only on the output .txt.

Usage:
    # one results file
    python -m cocoamot.emgi --input_txt res.txt --img_dir seq/img1 --output_txt out.txt --max_gap 8 --no_mask
    # every res_* folder of every sequence of a dataset (writes <seq>/emgi_<tracker>_<mode>_<gmc>_<gap>/)
    python -m cocoamot.emgi --batch --dataset data/real --max_gap 8 --no_mask
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from .io import box_to_center_size, center_size_to_box, list_frames, read_mot, write_mot
from .warps import apply_affine, compute_warps, invert_affine


def fill_track(frames, steps, inv_steps, mode, img_w, img_h, max_gap):
    """Fill the closed gaps of a single track.

    Args:
        frames: dict[frame] -> box [x, y, w, h] for the observed frames of this id.
        steps: per-step forward warps; steps[t-1] maps frame t -> t+1.
        inv_steps: inverse of `steps`; inv_steps[t-1] maps frame t+1 -> t.
        mode: 'warp' | 'lerp' | 'static'.
        img_w, img_h: frame size, for the out-of-frame safeguard.
        max_gap: max number of missing frames a gap may have to be filled (None = no limit).

    Yields:
        (a, b, frame, box): the gap boundaries and the interpolated box.
    """
    obs = sorted(frames.keys())
    for a, b in zip(obs[:-1], obs[1:]):
        gap_len = b - a - 1
        if gap_len <= 0:
            continue
        if max_gap is not None and gap_len > max_gap:
            continue

        c_a, s_a = box_to_center_size(frames[a])
        c_b, s_b = box_to_center_size(frames[b])

        fwd = {a: c_a.copy()}
        bwd = {b: c_b.copy()}
        if mode == "warp":
            c = c_a.copy()
            for f in range(a, b):  # step f -> f+1
                c = apply_affine(steps[f - 1], c)
                fwd[f + 1] = c.copy()
            c = c_b.copy()
            for f in range(b - 1, a - 1, -1):  # step f+1 -> f (inverse)
                c = apply_affine(inv_steps[f - 1], c)
                bwd[f] = c.copy()

        for t in range(a + 1, b):
            w_fwd = (b - t) / (b - a)
            w_bwd = (t - a) / (b - a)

            if mode == "static":
                center, size = c_a.copy(), s_a.copy()
            else:
                size = w_fwd * s_a + w_bwd * s_b
                center = w_fwd * fwd[t] + w_bwd * bwd[t] if mode == "warp" else w_fwd * c_a + w_bwd * c_b

            cx, cy = center
            if cx < 0 or cy < 0 or cx > img_w or cy > img_h:
                continue  # center left the frame

            yield a, b, t, center_size_to_box(center, size)


def interpolate_file(
    input_txt: str,
    output_txt: str,
    steps: np.ndarray | None,
    mode: str = "warp",
    img_w: int = 0,
    img_h: int = 0,
    max_gap: int | None = None,
    conf: float = 0.5,
) -> tuple[int, int]:
    """Run EMGI on one results file. Returns (gaps_filled, boxes_added).

    Output = original lines verbatim + interpolated lines, sorted by (frame, id).
    """
    inv_steps = np.stack([invert_affine(H) for H in steps]) if mode == "warp" else None
    tracks, raw_lines = read_mot(input_txt)

    filled = []
    gaps_filled = set()
    for tid, frames in tracks.items():
        for a, b, t, box in fill_track(frames, steps, inv_steps, mode, img_w, img_h, max_gap):
            filled.append((t, tid, box))
            gaps_filled.add((tid, a, b))

    out_rows = []
    for line in raw_lines:
        p = line.split(",")
        out_rows.append((int(float(p[0])), int(float(p[1])), line))
    for t, tid, box in filled:
        x, y, w, h = box
        out_rows.append((t, tid, f"{t},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"))
    out_rows.sort(key=lambda r: (r[0], r[1]))

    write_mot(output_txt, [r[2] for r in out_rows])
    return len(gaps_filled), len(filled)


def frame_size(img_dir: str, img_w: int | None = None, img_h: int | None = None) -> tuple[int, int]:
    """Frame (width, height): given explicitly, or read from the first image of the sequence."""
    if img_w and img_h:
        return img_w, img_h
    h, w = cv2.imread(list_frames(img_dir)[0]).shape[:2]
    return w, h


def output_folder_name(tracker: str, mode: str, gmc_method: str, max_gap: int | None) -> str:
    """Batch-mode output folder: emgi_<tracker>_<mode>[_<gmc>][_<gap:02d>] (gmc only in warp mode)."""
    name = f"emgi_{tracker}_{mode}" + (f"_{gmc_method}" if mode == "warp" else "")
    return name + (f"_{max_gap:02d}" if max_gap is not None else "")


def _warps_for(img_dir: str, args) -> np.ndarray | None:
    if args.mode != "warp":
        return None
    return compute_warps(
        img_dir,
        method=args.gmc_method,
        downscale=args.downscale,
        cache_path=getattr(args, "warp_cache", None),
        mask=not args.no_mask,
        det_txt=getattr(args, "det_txt", None),
        dilate=args.dilate,
        device=args.device,
        cache_dir=args.cache_dir,
    )


def run_batch(args) -> None:
    """Apply EMGI to every `res_*/<seq>.txt` of every sequence under `args.dataset`."""
    for seq_name in sorted(os.listdir(args.dataset)):
        seq_dir = os.path.join(args.dataset, seq_name)
        img_dir = os.path.join(seq_dir, "img1")
        if not os.path.isdir(img_dir):
            continue
        res_folders = sorted(
            d for d in os.listdir(seq_dir) if d.startswith("res_") and os.path.isdir(os.path.join(seq_dir, d))
        )
        if not res_folders:
            print(f"[emgi] no res_* folders in {seq_name}; skipping")
            continue
        img_w, img_h = frame_size(img_dir, args.img_w, args.img_h)
        steps = _warps_for(img_dir, args)
        for res_folder in res_folders:
            input_txt = os.path.join(seq_dir, res_folder, f"{seq_name}.txt")
            if not os.path.isfile(input_txt):
                continue
            tracker = res_folder.replace("res_", "")
            out_txt = os.path.join(
                seq_dir, output_folder_name(tracker, args.mode, args.gmc_method, args.max_gap), f"{seq_name}.txt"
            )
            g, n = interpolate_file(input_txt, out_txt, steps, args.mode, img_w, img_h, args.max_gap, args.conf)
            print(f"[emgi] {seq_name} | {tracker}: gaps_filled={g} boxes_added={n} -> {out_txt}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="EMGI — Ego-Motion Gap Interpolation (MOT post-process).")
    ap.add_argument("--batch", action="store_true", help="Process every res_* folder of every sequence in --dataset")
    ap.add_argument("--dataset", help="[batch] dataset root with <seq>/img1 and <seq>/res_*/<seq>.txt")
    ap.add_argument("--input_txt", help="[single] input MOT results .txt")
    ap.add_argument("--img_dir", help="[single] sequence image folder (<seq>/img1)")
    ap.add_argument("--output_txt", help="[single] output MOT .txt with gaps filled")
    ap.add_argument("--mode", choices=["warp", "lerp", "static"], default="warp", help="Gap-fill mode")
    ap.add_argument("--max_gap", type=int, default=None, help="Max missing frames per gap to fill (default: no limit)")
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence assigned to interpolated boxes")
    ap.add_argument(
        "--gmc_method", default="sparseOptFlow", help="sparseOptFlow[_nomask], orb, sift, ecc, loftr, disk_lightglue, xfeat"
    )
    ap.add_argument("--downscale", type=int, default=2, help="Warp estimation downscale factor")
    ap.add_argument("--no_mask", action="store_true", help="Disable detection-based background masking")
    ap.add_argument("--det_txt", default=None, help="[single] det.txt for masking (default: <seq>/det/det.txt)")
    ap.add_argument("--dilate", type=float, default=0.10, help="Mask box dilation fraction (0.10 => +10%%)")
    ap.add_argument("--device", default=None, help="Torch device for learned matchers (default: auto)")
    ap.add_argument("--warp_cache", default=None, help="[single] explicit warp cache .npz")
    ap.add_argument("--cache_dir", default=None, help="Folder for warp caches (default: the sequence folder)")
    ap.add_argument("--img_w", type=int, default=None, help="Frame width (default: read from first image)")
    ap.add_argument("--img_h", type=int, default=None, help="Frame height (default: read from first image)")
    args = ap.parse_args(argv)
    if args.batch and not args.dataset:
        ap.error("--batch requires --dataset")
    if not args.batch and not (args.input_txt and args.img_dir and args.output_txt):
        ap.error("single mode requires --input_txt, --img_dir and --output_txt (or use --batch --dataset)")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.batch:
        if not os.path.isdir(args.dataset):
            sys.exit(f"Dataset directory not found: {args.dataset}")
        run_batch(args)
        return
    img_w, img_h = frame_size(args.img_dir, args.img_w, args.img_h)
    steps = _warps_for(args.img_dir, args)
    g, n = interpolate_file(args.input_txt, args.output_txt, steps, args.mode, img_w, img_h, args.max_gap, args.conf)
    print(f"[emgi] mode={args.mode} gaps_filled={g} boxes_added={n} -> {args.output_txt}")


if __name__ == "__main__":
    main()
