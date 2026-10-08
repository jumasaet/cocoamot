#!/usr/bin/env python3
"""Run a YOLO detector over MOT sequences and write <seq>/det/det.txt (or --out_root/<seq>/det.txt).

    python scripts/detect.py --model weights/real_yolo12m/best.pt --dataset data/real --seqs seq_0006
"""

import argparse
import os

from cocoamot.detection import detect_sequence


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Detector weights (.pt)")
    ap.add_argument("--dataset", required=True, help="Dataset root with <seq>/img1/")
    ap.add_argument("--seqs", default=None, help="Comma-separated sequences (default: every folder with img1/)")
    ap.add_argument("--out_root", default=None, help="Write <out_root>/<seq>/det.txt instead of <seq>/det/det.txt")
    ap.add_argument("--conf", type=float, default=None, help="Confidence threshold (default: Ultralytics default 0.25)")
    ap.add_argument("--imgsz", type=int, default=None, help="Inference size (default: the checkpoint's training size)")
    ap.add_argument("--device", default=None, help="Torch device (default: auto)")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    kwargs = {k: v for k, v in (("conf", args.conf), ("imgsz", args.imgsz), ("device", args.device)) if v is not None}
    seqs = args.seqs.split(",") if args.seqs else sorted(
        d for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d, "img1"))
    )
    for seq in seqs:
        out = os.path.join(args.out_root, seq, "det.txt") if args.out_root else os.path.join(args.dataset, seq, "det", "det.txt")
        n = detect_sequence(model, os.path.join(args.dataset, seq, "img1"), out, **kwargs)
        print(f"[detect] {seq}: {n} boxes -> {out}")


if __name__ == "__main__":
    main()
