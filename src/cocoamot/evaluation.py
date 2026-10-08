"""Evaluate MOT results with TrackEval (MotChallenge2DBox; HOTA, CLEAR, Identity).

Builds a self-contained TrackEval workspace from explicit {name: results_dir} entries (no hand-edited
folder mappings), runs the same configuration as TrackEval's `scripts/run_mot_challenge.py`
(--METRICS HOTA CLEAR Identity, single process), and collects per-sequence metrics.

GT class 7 is a TrackEval distractor class ("static_person"): boxes marked 7 are ignored, not penalized.
"""

from __future__ import annotations

import csv
import os
import shutil

import numpy as np

# TrackEval (pinned commit) still uses the aliases removed in NumPy 1.24; they were plain synonyms.
for _alias, _type in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _type)

# TrackEval column -> reported column (HOTA is the AUC over localization thresholds).
METRICS = {
    "HOTA___AUC": "HOTA",
    "IDF1": "IDF1",
    "MOTA": "MOTA",
    "IDSW": "IDSW",
    "CLR_Re": "CLR_Re",
    "CLR_Pr": "CLR_Pr",
    "CLR_TP": "CLR_TP",
    "CLR_FN": "CLR_FN",
    "CLR_FP": "CLR_FP",
    "DetA___AUC": "DetA",
    "AssA___AUC": "AssA",
}
SPLIT = "train"


GT_FILE = "{root}/{seq}/gt/gt.txt"


def build_workspace(
    work_dir: str, benchmark: str, gt_root: str, seqs: list[str], results: dict[str, str], gt_file: str = GT_FILE
) -> None:
    """Lay out GT, seqmap and tracker results the way TrackEval's MotChallenge2DBox expects.

    Args:
        gt_root: dataset root with <seq>/seqinfo.ini.
        results: {tracker_name: folder containing <seq>.txt}.
        gt_file: GT path template with {root} and {seq} (e.g. an occlusion-bucket GT outside the dataset).
    """
    bench = f"{benchmark}-{SPLIT}"
    gt_dir = os.path.join(work_dir, "gt", "mot_challenge")
    for seq in seqs:
        dst = os.path.join(gt_dir, bench, seq)
        os.makedirs(os.path.join(dst, "gt"), exist_ok=True)
        shutil.copyfile(gt_file.format(root=gt_root, seq=seq), os.path.join(dst, "gt", "gt.txt"))
        shutil.copyfile(os.path.join(gt_root, seq, "seqinfo.ini"), os.path.join(dst, "seqinfo.ini"))
    os.makedirs(os.path.join(gt_dir, "seqmaps"), exist_ok=True)
    with open(os.path.join(gt_dir, "seqmaps", f"{bench}.txt"), "w") as f:
        f.write("name\n" + "".join(f"{s}\n" for s in seqs))
    for name, res_dir in results.items():
        dst = os.path.join(work_dir, "trackers", "mot_challenge", bench, name, "data")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)
        for seq in seqs:
            src = os.path.join(res_dir, f"{seq}.txt")
            if not os.path.isfile(src):
                raise FileNotFoundError(f"Missing results for '{name}': {src}")
            shutil.copyfile(src, os.path.join(dst, f"{seq}.txt"))


def run_trackeval(work_dir: str, benchmark: str, trackers: list[str]) -> None:
    """Run TrackEval on a workspace built by `build_workspace` (writes <tracker>/pedestrian_detailed.csv)."""
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": False,
            "NUM_PARALLEL_CORES": 1,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
        }
    )
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": os.path.join(work_dir, "gt", "mot_challenge"),
            "TRACKERS_FOLDER": os.path.join(work_dir, "trackers", "mot_challenge"),
            "BENCHMARK": benchmark,
            "SPLIT_TO_EVAL": SPLIT,
            "TRACKERS_TO_EVAL": trackers,
            "PRINT_CONFIG": False,
        }
    )
    metric_config = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(metric_config), trackeval.metrics.CLEAR(metric_config), trackeval.metrics.Identity(metric_config)]
    evaluator = trackeval.Evaluator(eval_config)
    _, messages = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics)
    failed = {t: m for t, m in messages.get("MotChallenge2DBox", {}).items() if m != "Success"}
    if failed:
        raise RuntimeError(f"TrackEval failed: {failed}")


def read_detailed(work_dir: str, benchmark: str, tracker: str) -> list[dict]:
    """Read <tracker>/pedestrian_detailed.csv -> rows {Sequence, <reported metrics>} (incl. COMBINED)."""
    path = os.path.join(work_dir, "trackers", "mot_challenge", f"{benchmark}-{SPLIT}", tracker, "pedestrian_detailed.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r = {k.strip(): v for k, v in r.items()}
            row = {"Sequence": r["seq"]}
            for col, out in METRICS.items():
                v = float(r[col])
                row[out] = int(v) if col in ("IDSW", "CLR_TP", "CLR_FN", "CLR_FP") else v
            rows.append(row)
    return rows


def evaluate(
    work_dir: str,
    benchmark: str,
    gt_root: str,
    seqs: list[str],
    entries: list[tuple[str, str, str]],
    gt_file: str = GT_FILE,
) -> list[dict]:
    """Evaluate (tracker, variant, results_dir) entries. Returns rows {Tracker, Variant, Sequence, metrics...}."""
    names = {f"{t}__{v}": (t, v, d) for t, v, d in entries}
    build_workspace(work_dir, benchmark, gt_root, seqs, {n: d for n, (_, _, d) in names.items()}, gt_file)
    run_trackeval(work_dir, benchmark, list(names))
    out = []
    for name, (t, v, _) in names.items():
        for row in read_detailed(work_dir, benchmark, name):
            out.append({"Tracker": t, "Variant": v, **row})
    return out


def write_csv(path: str, rows: list[dict]) -> None:
    """Write metric rows as a standard CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
