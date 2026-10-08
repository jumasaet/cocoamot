"""Experiment runner: one YAML in configs/experiments/ -> every number of one paper table/figure.

Pipeline per dataset:  [detect] -> track -> warps -> EMTS -> EMGI -> evaluate -> [significance]

Variants (per run):
    base       raw tracker output
    emts       EMTS(base)
    emgi       EMGI(base)
    emts_emgi  EMGI(EMTS(base))   (stitch first, then fill gaps)

Output layout (nothing is written inside the dataset):
    <out>/<dataset>/tracks/<tracker>/<seq>.txt                 tracker outputs (shared by all runs)
    <out>/<dataset>/<run>/<variant>/<tracker>/<seq>.txt        post-processed outputs
    <out>/<dataset>/<run>/metrics[_<bucket>].csv              TrackEval metrics per sequence + COMBINED
    <out>/<dataset>/<run>/significance_{combined,summary}.csv  occlusion analysis (if configured)
    <out>/cache/warps/<dataset>/<seq>/emgi_warps_*.npz         camera-warp cache
"""

from __future__ import annotations

import configparser
import copy
import os

import yaml

from . import evaluation, gt, significance
from .emgi import interpolate_file
from .emts import stitch_file
from .tracking import load_tracker_cfg, track_sequence
from .warps import compute_warps

STEPS = ("detect", "track", "warps", "emts", "emgi", "evaluate", "significance")
DEFAULT_STEPS = ("track", "warps", "emts", "emgi", "evaluate", "significance")
VARIANTS = ("base", "emts", "emgi", "emts_emgi")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def seq_size(seq_dir: str) -> tuple[int, int]:
    """(width, height) from seqinfo.ini."""
    ini = configparser.ConfigParser()
    ini.read(os.path.join(seq_dir, "seqinfo.ini"))
    s = {k.lower(): v for k, v in ini["Sequence"].items()}
    return int(s["imwidth"]), int(s["imheight"])


class Experiment:
    def __init__(self, cfg: dict, data_root: str, out_dir: str, repo_root: str, log=print):
        self.cfg = cfg
        self.data_root = data_root
        self.out = out_dir
        self.repo = repo_root
        self.log = log
        self.warps_cfg = {"method": "sparseOptFlow", "downscale": 2, "mask": False, "dilate": 0.10, **cfg.get("warps", {})}
        self.cache_root = os.path.join(out_dir, "cache", "warps")

    # -- paths -------------------------------------------------------------------------------
    def ds_root(self, ds: dict) -> str:
        return os.path.join(self.data_root, ds["path"])

    def seq_dir(self, ds: dict, seq: str) -> str:
        return os.path.join(self.ds_root(ds), seq)

    def tracks_dir(self, name: str, tracker: str) -> str:
        return os.path.join(self.out, name, "tracks", tracker)

    def variant_dir(self, name: str, run: str, variant: str, tracker: str) -> str:
        if variant == "base":
            return self.tracks_dir(name, tracker)
        return os.path.join(self.out, name, run, variant, tracker)

    def trackers(self, ds: dict) -> list[str]:
        """Ultralytics trackers run on a dataset (dataset override > global)."""
        return list(ds.get("trackers", self.cfg.get("trackers", [])))

    def external(self, ds: dict) -> list[str]:
        """External trackers (results provided as files) of a dataset (dataset override > global)."""
        return list(ds.get("external_trackers", self.cfg.get("external_trackers", [])))

    def run_trackers(self, ds: dict, run: dict) -> list[str]:
        """Trackers a run post-processes and evaluates (run override > every tracker of the dataset)."""
        everything = self.trackers(ds) + self.external(ds)
        return [t for t in run.get("trackers", everything) if t in everything]

    def runs(self) -> list[dict]:
        return self.cfg.get("runs") or [{"name": "default"}]

    # -- steps -------------------------------------------------------------------------------
    def detect(self, name: str, ds: dict) -> None:
        det_cfg = ds.get("detection") or self.cfg.get("detection")
        if not det_cfg:
            self.log(f"[detect] {name}: no detection config; using the provided det/det.txt")
            return
        from ultralytics import YOLO

        from .detection import detect_sequence

        model = YOLO(os.path.join(self.data_root, det_cfg["model"]))
        kwargs = {k: det_cfg[k] for k in ("conf", "imgsz", "device") if det_cfg.get(k) is not None}
        for seq in ds["sequences"]:
            out = os.path.join(self.out, name, "det", seq, "det.txt")
            n = detect_sequence(model, os.path.join(self.seq_dir(ds, seq), "img1"), out, **kwargs)
            self.log(f"[detect] {name}/{seq}: {n} boxes -> {out}")

    def det_path(self, name: str, ds: dict, seq: str) -> str:
        regen = os.path.join(self.out, name, "det", seq, "det.txt")
        return regen if os.path.isfile(regen) else os.path.join(self.seq_dir(ds, seq), "det", "det.txt")

    def track(self, name: str, ds: dict) -> None:
        cfg_dir = os.path.join(self.repo, self.cfg.get("tracker_cfg_dir", "configs/trackers"))
        for tracker in self.trackers(ds):
            tcfg_path = os.path.join(cfg_dir, f"{tracker}.yaml")
            for seq in ds["sequences"]:  # same order as the original runs: tracker-major, then sequences
                out = os.path.join(self.tracks_dir(name, tracker), f"{seq}.txt")
                n = track_sequence(tracker, load_tracker_cfg(tcfg_path), self.seq_dir(ds, seq), out, self.det_path(name, ds, seq))
                self.log(f"[track] {name}/{seq} {tracker}: {n} rows")
        for tracker in self.external(ds):
            missing = [s for s in ds["sequences"] if not os.path.isfile(os.path.join(self.tracks_dir(name, tracker), f"{s}.txt"))]
            if missing:
                raise FileNotFoundError(
                    f"External tracker '{tracker}' has no results for {name}: {missing}. "
                    f"Place them in {self.tracks_dir(name, tracker)}/<seq>.txt (see third_party/README.md)."
                )

    def warp_methods(self) -> set[str]:
        methods = {self.warps_cfg["method"]}
        for run in self.runs():
            for stage in ("emts", "emgi"):
                if run.get(stage, {}).get("gmc_method"):
                    methods.add(run[stage]["gmc_method"])
        return methods

    def warps(self, name: str, ds: dict, method: str | None = None, mask: bool | None = None):
        """Compute (or load) the warps of every sequence; returns {seq: steps}."""
        method = method or self.warps_cfg["method"]
        mask = self.warps_cfg["mask"] if mask is None else mask
        out = {}
        for seq in ds["sequences"]:
            sd = self.seq_dir(ds, seq)
            out[seq] = compute_warps(
                os.path.join(sd, "img1"),
                method=method,
                downscale=self.warps_cfg["downscale"],
                mask=mask,
                det_txt=self.det_path(name, ds, seq),
                dilate=self.warps_cfg["dilate"],
                device=self.warps_cfg.get("device"),
                cache_dir=os.path.join(self.cache_root, name, seq),
                verbose=False,
            )
        return out

    def emts(self, name: str, ds: dict, run: dict) -> None:
        p = {"max_time_gap": 60, "min_iou": 0.25, **run.get("emts", {})}
        steps = self.warps(name, ds, p.get("gmc_method"), mask=False)
        for tracker in self.run_trackers(ds, run):
            for seq in ds["sequences"]:
                src = os.path.join(self.tracks_dir(name, tracker), f"{seq}.txt")
                dst = os.path.join(self.variant_dir(name, run["name"], "emts", tracker), f"{seq}.txt")
                n = stitch_file(src, dst, steps[seq], p["max_time_gap"], p["min_iou"])
                self.log(f"[emts] {name}/{run['name']}/{seq} {tracker}: {n} merges")

    def emgi(self, name: str, ds: dict, run: dict, variants: list[str]) -> None:
        p = {"mode": "warp", "max_gap": 8, "conf": 0.5, **run.get("emgi", {})}
        steps = self.warps(name, ds, p.get("gmc_method")) if p["mode"] == "warp" else {s: None for s in ds["sequences"]}
        pairs = [(src, dst) for src, dst in (("base", "emgi"), ("emts", "emts_emgi")) if dst in variants]
        for tracker in self.run_trackers(ds, run):
            for seq in ds["sequences"]:
                w, h = seq_size(self.seq_dir(ds, seq))
                for src_v, dst_v in pairs:
                    src = os.path.join(self.variant_dir(name, run["name"], src_v, tracker), f"{seq}.txt")
                    dst = os.path.join(self.variant_dir(name, run["name"], dst_v, tracker), f"{seq}.txt")
                    g, n = interpolate_file(src, dst, steps[seq], p["mode"], w, h, p["max_gap"], p["conf"])
                    self.log(f"[emgi] {name}/{run['name']}/{seq} {tracker} {dst_v}: {n} boxes in {g} gaps")

    def gt_variants(self, name: str, ds: dict) -> dict[str, str]:
        """GT path templates to evaluate against: {"": main GT} or one per occlusion bucket."""
        ev = self.cfg.get("evaluation", {})
        protocol = ds.get("gt_protocol")
        main = os.path.join(self.ds_root(ds), "{seq}", "gt", "gt.txt")
        if protocol:  # regenerate the evaluation GT from the raw GT (synthetic benchmark)
            main = os.path.join(self.out, name, "gt", "main", "{seq}.txt")
            for seq in ds["sequences"]:
                w, h = seq_size(self.seq_dir(ds, seq))
                gt.apply_ignore(os.path.join(self.seq_dir(ds, seq), protocol["src"]), main.format(seq=seq), w, h,
                                protocol["min_wh"], protocol["max_occ"])
        if not ev.get("occlusion_buckets"):
            return {"": main}
        out = {}
        b = ev["occlusion_buckets"]
        for lo, hi in b["intervals"]:
            label = f"{int(round(lo * 100)):02d}_{int(round(hi * 100)):02d}"
            tmpl = os.path.join(self.out, name, "gt", f"occ_{label}", "{seq}.txt")
            for seq in ds["sequences"]:
                w, h = seq_size(self.seq_dir(ds, seq))
                gt.apply_interval(os.path.join(self.seq_dir(ds, seq), b["src"]), tmpl.format(seq=seq), w, h, lo, hi, b["min_wh"])
            out[label] = tmpl
        return out

    def evaluate(self, name: str, ds: dict, run: dict, variants: list[str]) -> dict[str, str]:
        """Evaluate every (tracker, variant); returns {bucket_label: metrics csv}."""
        bench = self.cfg.get("evaluation", {}).get("benchmark", "CocoaMOT")
        entries = [(t, v, self.variant_dir(name, run["name"], v, t)) for t in self.run_trackers(ds, run) for v in variants]
        csvs = {}
        for label, tmpl in self.gt_variants(name, ds).items():
            run_dir = os.path.join(self.out, name, run["name"])
            work = os.path.join(run_dir, "trackeval" + (f"_{label}" if label else ""))
            rows = evaluation.evaluate(work, bench, self.ds_root(ds), ds["sequences"], entries, gt_file=tmpl)
            path = os.path.join(run_dir, f"metrics{'_' + label if label else ''}.csv")
            evaluation.write_csv(path, rows)
            csvs[label] = path
            self.log(f"[evaluate] {name}/{run['name']} {label or 'main'} -> {path}")
            for r in rows:
                if r["Sequence"] == "COMBINED":
                    self.log(f"    {r['Tracker']:<11} {r['Variant']:<10} HOTA={r['HOTA']:.4f} IDF1={r['IDF1']:.4f} IDSW={r['IDSW']}")
        return csvs

    def significance(self, name: str, run: dict, csvs: dict[str, str], variants: list[str]) -> None:
        sig = self.cfg.get("significance")
        if not sig or set(csvs) == {""}:
            return
        buckets = {label: significance.load_bucket_csv(p) for label, p in csvs.items()}
        comb, summary = significance.significance(
            buckets, sig.get("test_buckets", list(buckets)), [v for v in variants if v != "base"], "base"
        )
        run_dir = os.path.join(self.out, name, run["name"])
        comb.to_csv(os.path.join(run_dir, "significance_combined.csv"), index=False, float_format="%.15g")
        summary.to_csv(os.path.join(run_dir, "significance_summary.csv"), index=False, float_format="%.6g")
        self.log(f"[significance] {name}/{run['name']} -> {run_dir}/significance_*.csv")

    # -- driver ------------------------------------------------------------------------------
    def run(self, steps=DEFAULT_STEPS) -> None:
        for name, ds in self.cfg["datasets"].items():
            if "detect" in steps:
                self.detect(name, ds)
            if "track" in steps:
                self.track(name, ds)
            if "warps" in steps:
                for m in sorted(self.warp_methods()):
                    self.warps(name, ds, m, mask=False)
                    if self.warps_cfg["mask"]:
                        self.warps(name, ds, m, mask=True)
            for run in self.runs():
                run = copy.deepcopy(run)
                variants = run.get("variants", list(VARIANTS))
                if "emts" in steps and ("emts" in variants or "emts_emgi" in variants):
                    self.emts(name, ds, run)
                if "emgi" in steps and ({"emgi", "emts_emgi"} & set(variants)):
                    self.emgi(name, ds, run, variants)
                csvs = self.evaluate(name, ds, run, variants) if "evaluate" in steps else {}
                if "significance" in steps:
                    self.significance(name, run, csvs, variants)
