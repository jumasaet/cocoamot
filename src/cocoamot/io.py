"""MOTChallenge I/O shared by every stage of the pipeline.

MOT text format (one box per line, 1-indexed frames, top-left boxes):
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, cls/x, vis/y, z
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_frames(img_dir: str) -> list[str]:
    """Return the image paths of a sequence sorted by file name (= frame order, frame 1 first)."""
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
    if not files:
        raise FileNotFoundError(f"No images found in {img_dir} (images are distributed separately, see data/README.md)")
    return [os.path.join(img_dir, f) for f in files]


def read_mot(path: str) -> tuple[dict[int, dict[int, np.ndarray]], list[str]]:
    """Read a MOT results file.

    Returns:
        tracks: dict[id] -> dict[frame] -> np.array([x, y, w, h]) (top-left x, y).
        raw_lines: the original non-empty lines, verbatim, in file order.
    """
    tracks: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    raw_lines: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_lines.append(line)
            p = line.split(",")
            frame, tid = int(float(p[0])), int(float(p[1]))
            tracks[tid][frame] = np.array([float(p[2]), float(p[3]), float(p[4]), float(p[5])])
    return tracks, raw_lines


def read_mot_records(path: str) -> dict[int, dict[int, dict]]:
    """Read a MOT results file keeping each row: dict[id] -> dict[frame] -> {"box", "conf", "line"}."""
    tracks: dict[int, dict[int, dict]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            frame, tid = int(float(p[0])), int(float(p[1]))
            box = np.array([float(p[2]), float(p[3]), float(p[4]), float(p[5])])
            tracks[tid][frame] = {"box": box, "conf": float(p[6]), "line": line}
    return tracks


def write_mot(path: str, lines: list[str]) -> None:
    """Write MOT lines (without trailing newlines), creating the parent directory."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def load_detections(det_path: str) -> dict[int, list[list[float]]]:
    """Read a MOT det.txt grouped by frame as [x1, y1, x2, y2, conf, cls=0] rows (tracker input format)."""
    dets_by_frame: dict[int, list[list[float]]] = {}
    with open(det_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame_idx = int(float(parts[0]))
            bb_left, bb_top, w, h, conf = (float(x) for x in parts[2:7])
            dets_by_frame.setdefault(frame_idx, []).append([bb_left, bb_top, bb_left + w, bb_top + h, conf, 0])
    return dets_by_frame


def read_det_boxes(det_txt: str) -> dict[int, np.ndarray]:
    """Read a MOT det.txt -> dict[frame] -> (N, 4) array of top-left [x, y, w, h] boxes (for warp masking)."""
    per_frame: dict[int, list] = defaultdict(list)
    with open(det_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            per_frame[int(float(p[0]))].append([float(p[2]), float(p[3]), float(p[4]), float(p[5])])
    return {fr: np.array(v, dtype=np.float64) for fr, v in per_frame.items()}


def box_to_center_size(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MOT top-left box [x, y, w, h] -> (center xy, size wh)."""
    x, y, w, h = box
    return np.array([x + w / 2.0, y + h / 2.0]), np.array([w, h])


def center_size_to_box(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """(center xy, size wh) -> MOT top-left box [x, y, w, h]."""
    cx, cy = center
    w, h = size
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h])
