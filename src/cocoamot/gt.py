"""Ground-truth preparation: CVAT XML -> MOT, and the evaluation-protocol filters.

Filters mark boxes with the TrackEval distractor class (7) so they are ignored, not penalized. They never
modify their input: they read a source gt.txt and write a new one. An existing ignore mark is kept by
`apply_ignore` and recomputed by `apply_interval`, so both can start from the released (already filtered)
gt.txt. Rows with fewer than 10 columns are dropped by the filters (the synthetic GT has 10 columns:
..., conf, class, visibility, z).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

IGNORE_CLASS = 7


def cvat_to_mot(xml_path: str, gt_txt: str, det_txt: str | None = None) -> int:
    """Convert a CVAT "video" XML export to a MOT gt.txt (frame, id, x, y, w, h, 1, 1, 1.0).

    Frames and track ids become 1-indexed; boxes with outside=1 are skipped. If `det_txt` is given, the same
    boxes are also written as oracle detections (conf 1.0). Returns the number of boxes.
    """
    root = ET.parse(xml_path).getroot()
    gt_lines, det_lines = [], []
    for track in root.findall("track"):
        track_id = int(track.get("id")) + 1
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame")) + 1
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            w, h = float(box.get("xbr")) - xtl, float(box.get("ybr")) - ytl
            gt_lines.append((frame, f"{frame},{track_id},{xtl:.2f},{ytl:.2f},{w:.2f},{h:.2f},1,1,1.0"))
            det_lines.append((frame, f"{frame},-1,{xtl:.2f},{ytl:.2f},{w:.2f},{h:.2f},1.0,-1,-1,-1"))
    gt_lines.sort(key=lambda x: x[0])
    det_lines.sort(key=lambda x: x[0])
    for path, lines in ((gt_txt, gt_lines), (det_txt, det_lines)):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                f.writelines(line + "\n" for _, line in lines)
    return len(gt_lines)


def _rewrite(src: str, dst: str, rule) -> tuple[int, int]:
    """Apply `rule(parts) -> parts` to every 10-column row of `src`; write `dst`. Returns (ignored, total)."""
    out, ignored, total = [], 0, 0
    with open(src) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            total += 1
            parts = rule(parts)
            ignored += int(parts[7]) == IGNORE_CLASS
            out.append(",".join(parts) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w") as f:
        f.writelines(out)
    return ignored, total


def apply_ignore(src: str, dst: str, img_w: int, img_h: int, min_wh: float = 0.0, max_occ: float = 1.0,
                 ignore_class: int = IGNORE_CLASS) -> tuple[int, int]:
    """Mark as ignore every box smaller than min_wh (normalized w or h) or more occluded than max_occ."""
    min_w, min_h, min_vis = min_wh * img_w, min_wh * img_h, 1.0 - max_occ

    def rule(p):
        if int(p[7]) != ignore_class and (float(p[4]) < min_w or float(p[5]) < min_h or float(p[8]) < min_vis):
            p[7] = str(ignore_class)
        return p

    return _rewrite(src, dst, rule)


def apply_interval(src: str, dst: str, img_w: int, img_h: int, min_occ: float, max_occ: float, min_wh: float = 0.0,
                   ignore_class: int = IGNORE_CLASS) -> tuple[int, int]:
    """Occlusion bucket: evaluate (class 1) only boxes with min_occ <= 1 - visibility <= max_occ and size >= min_wh.

    Both interval ends are inclusive (as in the original analysis), so a box exactly on a boundary belongs to
    the two adjacent buckets.
    """
    min_w, min_h = min_wh * img_w, min_wh * img_h

    def rule(p):
        occ = 1.0 - float(p[8])
        p[7] = "1" if (float(p[4]) >= min_w and float(p[5]) >= min_h and min_occ <= occ <= max_occ) else str(ignore_class)
        return p

    return _rewrite(src, dst, rule)
