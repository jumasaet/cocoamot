"""EMTS — Ego-Motion Track Stitching.

Reconnects tracks broken by identity switches. The last known position of every track is projected
forward with the camera warps (GMC); a track that starts within `max_time_gap` frames is a candidate when
the warped box overlaps its first box with IoU >= `min_iou`. Candidates are assigned with the Hungarian
algorithm (cost = 1 - warped IoU) and merged with union-find, keeping the oldest id.

Usage:
    # one results file
    python -m cocoamot.emts --input_txt res.txt --img_dir seq/img1 --output_txt out.txt
    # every res_* folder (not already stitched) of every sequence of a dataset (writes <seq>/res_<trk>_stit/)
    python -m cocoamot.emts --batch --dataset data/real
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from .io import read_mot_records, write_mot
from .warps import apply_affine, compute_warps

DEFAULT_MIN_IOU = 0.25
DEFAULT_GMC = "sparseOptFlow"


def iou(box1, box2) -> float:
    """IoU of two top-left [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    if inter == 0:
        return 0.0
    return inter / (w1 * h1 + w2 * h2 - inter)


def stitch_tracks(tracks: dict, steps: np.ndarray, max_time_gap: int = 60, min_iou: float = DEFAULT_MIN_IOU):
    """Stitch broken tracks. Returns (output lines sorted by (frame, id), number of merges)."""
    heads, tails = {}, {}
    for tid, frames_dict in tracks.items():
        frames = sorted(frames_dict.keys())
        heads[tid] = frames[0]
        tails[tid] = frames[-1]

    dead_tracks = list(tails.keys())
    new_tracks = list(heads.keys())
    cost_matrix = np.ones((len(dead_tracks), len(new_tracks))) * 1000.0

    for i, t_dead in enumerate(dead_tracks):
        f_end = tails[t_dead]
        box_end = tracks[t_dead][f_end]["box"]
        curr_center = np.array([box_end[0] + box_end[2] / 2.0, box_end[1] + box_end[3] / 2.0])

        projected_centers = {}
        for f in range(f_end, f_end + max_time_gap + 1):
            projected_centers[f] = curr_center.copy()
            if f - 1 < len(steps):
                curr_center = apply_affine(steps[f - 1], curr_center)

        for j, t_new in enumerate(new_tracks):
            if t_dead == t_new:
                continue
            f_start = heads[t_new]
            if 0 < f_start - f_end <= max_time_gap:
                p_center = projected_centers[f_start]
                p_box = np.array(
                    [p_center[0] - box_end[2] / 2.0, p_center[1] - box_end[3] / 2.0, box_end[2], box_end[3]]
                )
                sim = iou(p_box, tracks[t_new][f_start]["box"])
                if sim >= min_iou:
                    cost_matrix[i, j] = 1.0 - sim

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    parent = {tid: tid for tid in tracks}

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    merges = 0
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1.0:
            root_dead, root_new = find(dead_tracks[r]), find(new_tracks[c])
            if root_dead != root_new:
                parent[root_new] = root_dead  # the oldest id survives
                merges += 1

    out = []
    for tid, frames_dict in tracks.items():
        final_id = find(tid)
        for frame, rec in frames_dict.items():
            parts = rec["line"].split(",")
            parts[1] = str(final_id)
            out.append((frame, final_id, ",".join(parts)))
    out.sort(key=lambda x: (x[0], x[1]))
    return [r[2] for r in out], merges


def stitch_file(input_txt: str, output_txt: str, steps: np.ndarray, max_time_gap: int = 60, min_iou: float = DEFAULT_MIN_IOU) -> int:
    """Run EMTS on one results file. Returns the number of merges (nothing is written for an empty input)."""
    tracks = read_mot_records(input_txt)
    if not tracks:
        return 0
    lines, merges = stitch_tracks(tracks, steps, max_time_gap, min_iou)
    write_mot(output_txt, lines)
    return merges


def output_folder_name(res_folder: str, gmc_method: str = DEFAULT_GMC, min_iou: float = DEFAULT_MIN_IOU) -> str:
    """Batch-mode output folder: <res_folder>_stit[_<gmc> if not sparseOptFlow][_<iou*100:02d> if not 0.25]."""
    name = f"{res_folder}_stit"
    if gmc_method != DEFAULT_GMC:
        name += f"_{gmc_method}"
    if min_iou != DEFAULT_MIN_IOU:
        name += "_" + f"{min_iou:.2f}".split(".")[1]
    return name


def _warps_for(img_dir: str, args) -> np.ndarray:
    return compute_warps(
        img_dir,
        method=args.gmc_method,
        downscale=args.downscale,
        cache_path=getattr(args, "warp_cache", None),
        mask=False,
        device=args.device,
        cache_dir=args.cache_dir,
    )


def run_batch(args) -> None:
    """Apply EMTS to every not-yet-stitched `res_*/<seq>.txt` of every sequence under `args.dataset`."""
    for seq_name in sorted(os.listdir(args.dataset)):
        seq_dir = os.path.join(args.dataset, seq_name)
        img_dir = os.path.join(seq_dir, "img1")
        if not os.path.isdir(img_dir):
            continue
        res_folders = sorted(
            d
            for d in os.listdir(seq_dir)
            if d.startswith("res_") and "_stit" not in d and os.path.isdir(os.path.join(seq_dir, d))
        )
        if not res_folders:
            continue
        steps = _warps_for(img_dir, args)
        for folder in res_folders:
            input_txt = os.path.join(seq_dir, folder, f"{seq_name}.txt")
            if not os.path.isfile(input_txt):
                continue
            out_folder = output_folder_name(folder, args.gmc_method, args.min_iou)
            n = stitch_file(input_txt, os.path.join(seq_dir, out_folder, f"{seq_name}.txt"), steps, args.max_time_gap, args.min_iou)
            print(f"[emts] {seq_name} | {folder}: {n} ids merged -> {out_folder}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="EMTS — Ego-Motion Track Stitching (MOT post-process).")
    ap.add_argument("--batch", action="store_true", help="Process every res_* folder of every sequence in --dataset")
    ap.add_argument("--dataset", help="[batch] dataset root with <seq>/img1 and <seq>/res_*/<seq>.txt")
    ap.add_argument("--input_txt", help="[single] input MOT results .txt")
    ap.add_argument("--img_dir", help="[single] sequence image folder (<seq>/img1), used to compute/load the warps")
    ap.add_argument("--output_txt", help="[single] output MOT .txt")
    ap.add_argument("--max_time_gap", type=int, default=60, help="Max frames between a track end and a new track start")
    ap.add_argument("--min_iou", type=float, default=DEFAULT_MIN_IOU, help="Min warped IoU to merge two tracks")
    ap.add_argument("--gmc_method", default=DEFAULT_GMC, help="Warp method (see cocoamot.warps)")
    ap.add_argument("--downscale", type=int, default=2, help="Warp estimation downscale factor")
    ap.add_argument("--device", default=None, help="Torch device for learned matchers (default: auto)")
    ap.add_argument("--warp_cache", default=None, help="[single] explicit warp cache .npz")
    ap.add_argument("--cache_dir", default=None, help="Folder for warp caches (default: the sequence folder)")
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
    n = stitch_file(args.input_txt, args.output_txt, _warps_for(args.img_dir, args), args.max_time_gap, args.min_iou)
    print(f"[emts] {n} ids merged -> {args.output_txt}")


if __name__ == "__main__":
    main()
