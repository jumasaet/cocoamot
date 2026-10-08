#!/usr/bin/env python3
"""Ground-truth preparation (never modifies its input).

    # CVAT "video" XML -> MOT gt.txt (real sequences)
    python scripts/prepare_gt.py cvat --xml annotations_final.xml --gt_out data/real/seq_0006/gt/gt.txt

    # a stricter synthetic evaluation GT: ignore boxes < 5% of the frame or > 60% occluded
    # (the released gt/gt.txt uses --min_wh 0.02 --max_occ 0.8)
    python scripts/prepare_gt.py ignore --dataset data/synthetic --dst gt/gt_strict.txt --min_wh 0.05 --max_occ 0.6

    # one occlusion bucket (only 20-40% occluded boxes are evaluated)
    python scripts/prepare_gt.py interval --dataset data/synthetic --dst gt/gt_occ_20_40.txt \\
        --min_occ 0.2 --max_occ 0.4 --min_wh 0.02

Frame size comes from each sequence's seqinfo.ini unless --img_w/--img_h are given.
"""

import argparse
import os

from cocoamot import gt
from cocoamot.experiment import seq_size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cvat", help="CVAT XML -> MOT gt.txt")
    c.add_argument("--xml", required=True)
    c.add_argument("--gt_out", required=True)
    c.add_argument("--det_out", default=None, help="Also write the GT boxes as oracle detections")

    for name in ("ignore", "interval"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, help="Dataset root with <seq>/seqinfo.ini")
        p.add_argument("--seqs", default=None, help="Comma-separated sequences (default: all with seqinfo.ini)")
        p.add_argument("--src", default="gt/gt.txt", help="Source GT, relative to the sequence folder")
        p.add_argument("--dst", required=True, help="Output GT, relative to the sequence folder (or absolute with {seq})")
        p.add_argument("--min_wh", type=float, default=0.0, help="Min normalized box width/height")
        p.add_argument("--img_w", type=int, default=None)
        p.add_argument("--img_h", type=int, default=None)
        if name == "ignore":
            p.add_argument("--max_occ", type=float, default=1.0, help="Max occlusion (1 - visibility)")
        else:
            p.add_argument("--min_occ", type=float, required=True)
            p.add_argument("--max_occ", type=float, required=True)
    args = ap.parse_args()

    if args.cmd == "cvat":
        n = gt.cvat_to_mot(args.xml, args.gt_out, args.det_out)
        print(f"[cvat] {n} boxes -> {args.gt_out}")
        return

    seqs = args.seqs.split(",") if args.seqs else sorted(
        d for d in os.listdir(args.dataset) if os.path.isfile(os.path.join(args.dataset, d, "seqinfo.ini"))
    )
    for seq in seqs:
        sd = os.path.join(args.dataset, seq)
        w, h = (args.img_w, args.img_h) if args.img_w and args.img_h else seq_size(sd)
        dst = args.dst.format(seq=seq) if os.path.isabs(args.dst) else os.path.join(sd, args.dst)
        src = os.path.join(sd, args.src)
        if os.path.abspath(src) == os.path.abspath(dst):
            ap.error(f"--dst must differ from --src (refusing to overwrite {src})")
        if args.cmd == "ignore":
            ign, tot = gt.apply_ignore(src, dst, w, h, args.min_wh, args.max_occ)
        else:
            ign, tot = gt.apply_interval(src, dst, w, h, args.min_occ, args.max_occ, args.min_wh)
        print(f"[{args.cmd}] {seq}: {tot - ign} evaluated / {ign} ignored -> {dst}")


if __name__ == "__main__":
    main()
