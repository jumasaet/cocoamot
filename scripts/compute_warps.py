#!/usr/bin/env python3
"""Compute (and cache) the per-sequence camera warps used by EMTS and EMGI.

    python scripts/compute_warps.py --dataset data/synthetic --methods sparseOptFlow,orb --no_mask
"""

import argparse
import os

from cocoamot.warps import compute_warps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Dataset root with <seq>/img1/")
    ap.add_argument("--seqs", default=None, help="Comma-separated sequences (default: every folder with img1/)")
    ap.add_argument("--methods", default="sparseOptFlow", help="sparseOptFlow[_nomask],orb,sift,ecc,loftr,disk_lightglue,xfeat")
    ap.add_argument("--downscale", type=int, default=2, help="Warp estimation downscale factor")
    ap.add_argument("--no_mask", action="store_true", help="Disable detection-based background masking")
    ap.add_argument("--dilate", type=float, default=0.10, help="Mask box dilation fraction (0.10 => +10%%)")
    ap.add_argument("--device", default=None, help="Torch device for learned matchers (default: auto)")
    ap.add_argument("--cache_root", default=None, help="Write caches to <cache_root>/<seq>/ (default: the sequence folder)")
    args = ap.parse_args()

    seqs = args.seqs.split(",") if args.seqs else sorted(
        d for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d, "img1"))
    )
    for method in args.methods.split(","):
        for seq in seqs:
            compute_warps(
                os.path.join(args.dataset, seq, "img1"),
                method=method,
                downscale=args.downscale,
                mask=not args.no_mask,
                dilate=args.dilate,
                device=args.device,
                cache_dir=os.path.join(args.cache_root, seq) if args.cache_root else None,
            )


if __name__ == "__main__":
    main()
