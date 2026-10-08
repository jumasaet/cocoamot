#!/usr/bin/env python3
"""Run Ultralytics trackers over cached detections (<seq>/det/det.txt), MOTChallenge output.

Every tracker consumes the same detections, so the comparison isolates association and motion. Output:
<seq>/res_<tracker>/<seq>.txt, or <out_root>/<tracker>/<seq>.txt with --out_root.

    python scripts/track.py --dataset data/real --trackers botsort,bytetrack --seqs seq_0006
"""

import argparse
import os

from cocoamot.tracking import TRACKER_NAMES, load_tracker_cfg, track_sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Dataset root with <seq>/img1 and <seq>/det/det.txt")
    ap.add_argument("--trackers", default=",".join(TRACKER_NAMES), help=f"Comma-separated subset of {list(TRACKER_NAMES)}")
    ap.add_argument("--seqs", default=None, help="Comma-separated sequences (default: all with det/det.txt)")
    ap.add_argument("--gmc", default=None, help="Force gmc_method on GMC-capable trackers (default: from the YAML)")
    ap.add_argument("--tracker_cfg_dir", default=os.path.join(REPO, "configs", "trackers"), help="Folder with <tracker>.yaml")
    ap.add_argument("--out_root", default=None, help="Write <out_root>/<tracker>/<seq>.txt")
    args = ap.parse_args()

    trackers = [t.strip() for t in args.trackers.split(",") if t.strip()]
    unknown = [t for t in trackers if t not in TRACKER_NAMES]
    if unknown:
        ap.error(f"unknown trackers {unknown}")
    seqs = args.seqs.split(",") if args.seqs else sorted(
        d for d in os.listdir(args.dataset) if os.path.isfile(os.path.join(args.dataset, d, "det", "det.txt"))
    )
    for tracker in trackers:
        for seq in seqs:
            seq_dir = os.path.join(args.dataset, seq)
            out = (
                os.path.join(args.out_root, tracker, f"{seq}.txt")
                if args.out_root
                else os.path.join(seq_dir, f"res_{tracker}", f"{seq}.txt")
            )
            cfg = load_tracker_cfg(os.path.join(args.tracker_cfg_dir, f"{tracker}.yaml"), args.gmc)
            n = track_sequence(tracker, cfg, seq_dir, out)
            print(f"[track] {tracker:<11} {seq}: {n} rows -> {out}")


if __name__ == "__main__":
    main()
