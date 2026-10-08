#!/usr/bin/env python3
"""Evaluate MOT results with TrackEval (HOTA, CLEAR, Identity) against a dataset's GT.

Each --entry names one result folder explicitly (no hidden folder mapping):

    python scripts/evaluate.py --gt_root data/real --seqs seq_0001,seq_0006 \\
        --entry botsort:base=outputs/main/real/tracks/botsort \\
        --entry botsort:emts=outputs/main/real/default/emts/botsort \\
        --out_csv outputs/eval/metrics.csv
"""

import argparse
import os

from cocoamot import evaluation


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_root", required=True, help="Dataset root with <seq>/seqinfo.ini and <seq>/gt/gt.txt")
    ap.add_argument("--seqs", required=True, help="Comma-separated sequences")
    ap.add_argument("--entry", action="append", required=True, help="TRACKER:VARIANT=FOLDER with <seq>.txt files")
    ap.add_argument("--gt_file", default=evaluation.GT_FILE, help="GT path template with {root} and {seq}")
    ap.add_argument("--benchmark", default="CocoaMOT", help="TrackEval benchmark name")
    ap.add_argument("--work_dir", default=None, help="TrackEval workspace (default: next to --out_csv)")
    ap.add_argument("--out_csv", required=True, help="Per-sequence metrics (+ COMBINED rows)")
    args = ap.parse_args()

    entries = []
    for e in args.entry:
        key, folder = e.split("=", 1)
        tracker, variant = key.split(":", 1) if ":" in key else (key, "base")
        entries.append((tracker, variant, folder))
    work = args.work_dir or os.path.join(os.path.dirname(os.path.abspath(args.out_csv)), "trackeval")
    rows = evaluation.evaluate(work, args.benchmark, args.gt_root, args.seqs.split(","), entries, args.gt_file)
    evaluation.write_csv(args.out_csv, rows)
    for r in rows:
        if r["Sequence"] == "COMBINED":
            print(f"{r['Tracker']:<11} {r['Variant']:<10} HOTA={r['HOTA']:.4f} IDF1={r['IDF1']:.4f} MOTA={r['MOTA']:.4f} IDSW={r['IDSW']}")
    print(f"-> {args.out_csv}")


if __name__ == "__main__":
    main()
