#!/usr/bin/env python3
"""Occlusion analysis: combine per-bucket metrics into a 0-100% table and run paired Wilcoxon tests.

    python scripts/significance.py --bucket 00_20=m_00_20.csv ... --bucket 80_100=m_80_100.csv \\
        --test_buckets 00_20,20_40,40_60,60_80 --out_dir outputs/occlusion/significance

Inputs are the per-sequence CSVs of scripts/evaluate.py (variants: base, emts, emgi, emts_emgi).
--legacy reads the historical ';'-separated, decimal-comma format.
"""

import argparse
import os

from cocoamot.significance import load_bucket_csv, significance


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", action="append", required=True, help="LABEL=CSV (e.g. 00_20=metrics_00_20.csv)")
    ap.add_argument("--test_buckets", default=None, help="Buckets with per-bucket tests (default: all)")
    ap.add_argument("--variants", default="emgi,emts,emts_emgi", help="Variants compared against the baseline")
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--legacy", action="store_true", help="Inputs use ';' separators and ',' decimals")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    buckets = {}
    for b in args.bucket:
        label, path = b.split("=", 1)
        buckets[label] = load_bucket_csv(path, legacy=args.legacy)
    test = args.test_buckets.split(",") if args.test_buckets else list(buckets)
    comb, summary = significance(buckets, test, args.variants.split(","), args.baseline)

    os.makedirs(args.out_dir, exist_ok=True)
    comb.to_csv(os.path.join(args.out_dir, "combined_00_100.csv"), index=False, float_format="%.15g")
    summary.to_csv(os.path.join(args.out_dir, "significance_summary.csv"), index=False, float_format="%.6g")
    print(summary.to_string(index=False))
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
