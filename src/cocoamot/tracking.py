"""Run Ultralytics trackers over cached detections (det/det.txt) and write MOTChallenge results.

Every tracker consumes the *same* detections, so the comparison isolates association and motion.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .io import list_frames, load_detections

TRACKER_NAMES = ("botsort", "bytetrack", "ocsort", "deepocsort")
# Tracker-YAML keys that only exist with patches/ultralytics_ours.patch applied.
PATCH_KEYS = ("static_prior_beta", "min_wh_clamp", "use_reid_firewall", "use_diou_rescue", "use_omega_penalty")


def _tracker_class(name: str):
    from ultralytics.trackers import BOTSORT, OCSORT, BYTETracker, DeepOCSORT

    return {"botsort": BOTSORT, "bytetrack": BYTETracker, "ocsort": OCSORT, "deepocsort": DeepOCSORT}[name]


def load_tracker_cfg(yaml_path: str, gmc_override: str | None = None):
    """Load a tracker YAML into the namespace Ultralytics trackers expect; optionally force `gmc_method`."""
    from ultralytics.utils import YAML, IterableSimpleNamespace

    cfg = IterableSimpleNamespace(**YAML.load(yaml_path))
    if any(k in vars(cfg) for k in PATCH_KEYS):
        from ultralytics.trackers.byte_tracker import BYTETracker

        if not hasattr(BYTETracker, "_init_ablation_panel"):
            raise RuntimeError(
                f"{yaml_path} uses the ablation panel ({', '.join(k for k in PATCH_KEYS if k in vars(cfg))}) but the "
                "installed ultralytics is not patched; those keys would be silently ignored. "
                "Apply patches/ultralytics_ours.patch (see README)."
            )
    if gmc_override is not None and hasattr(cfg, "gmc_method"):
        cfg.gmc_method = gmc_override
    return cfg


def track_sequence(tracker: str, cfg, seq_dir: str, out_file: str, det_path: str | None = None) -> int:
    """Run one tracker over one sequence's cached detections. Returns the number of rows written.

    A fresh tracker is built per sequence (clean id state). Output columns:
    frame, id, x, y, w, h, conf, 1, -1, -1.
    """
    from ultralytics.engine.results import Boxes

    dets_by_frame = load_detections(det_path or os.path.join(seq_dir, "det", "det.txt"))
    trk = _tracker_class(tracker)(args=cfg)

    rows = []
    for frame_idx, img_path in enumerate(list_frames(os.path.join(seq_dir, "img1")), start=1):
        img = cv2.imread(img_path)
        if img is None:
            continue
        raw = dets_by_frame.get(frame_idx, [])
        data = np.array(raw, dtype=np.float32) if raw else np.empty((0, 6), dtype=np.float32)
        for t in trk.update(Boxes(data, img.shape[:2]), img):
            x1, y1, x2, y2, track_id, conf, _cls, _ = t
            rows.append(f"{frame_idx},{int(track_id)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.4f},1,-1,-1")

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w") as f:
        f.writelines(r + "\n" for r in rows)
    return len(rows)
