#!/usr/bin/env python3
"""Single entry point: reproduce one paper table/figure from its config.

    python scripts/run_experiment.py --config configs/experiments/main.yaml
    python scripts/run_experiment.py --config ... --steps evaluate      # re-evaluate only

The dataset root defaults to the repository's data/ folder (see data/README.md to download the images);
override it with --data_root or the COCOAMOT_DATA environment variable.
"""

import argparse
import os

from cocoamot.experiment import DEFAULT_STEPS, STEPS, Experiment, load_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/experiments/<exp>.yaml")
    ap.add_argument(
        "--data_root",
        default=os.environ.get("COCOAMOT_DATA", os.path.join(REPO, "data")),
        help="Dataset root (default: env COCOAMOT_DATA, else <repo>/data)",
    )
    ap.add_argument("--out", default=None, help="Output folder (default: outputs/<config name>)")
    ap.add_argument("--steps", default=",".join(DEFAULT_STEPS), help=f"Comma-separated subset of {list(STEPS)}")
    ap.add_argument("--datasets", default=None, help="Comma-separated subset of the config's datasets")
    ap.add_argument("--sequences", default=None, help="Comma-separated subset of sequences (debugging)")
    args = ap.parse_args()
    if not os.path.isdir(args.data_root):
        ap.error(f"dataset root not found: {args.data_root}")

    cfg = load_config(args.config)
    if args.datasets:
        keep = args.datasets.split(",")
        cfg["datasets"] = {k: v for k, v in cfg["datasets"].items() if k in keep}
    if args.sequences:
        keep = args.sequences.split(",")
        for ds in cfg["datasets"].values():
            ds["sequences"] = [s for s in ds["sequences"] if s in keep]
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        ap.error(f"unknown steps {sorted(unknown)}")
    out = args.out or os.path.join(REPO, "outputs", cfg.get("name", os.path.splitext(os.path.basename(args.config))[0]))
    Experiment(cfg, os.path.abspath(args.data_root), os.path.abspath(out), REPO).run(steps)


if __name__ == "__main__":
    main()
