"""Run a YOLO detector over a sequence and write a MOT det.txt (frame, -1, x, y, w, h, conf, -1, -1, -1)."""

from __future__ import annotations

import os

from .io import list_frames


def detect_sequence(model, img_dir: str, out_file: str, **predict_kwargs) -> int:
    """Detect every frame of `img_dir` with an Ultralytics `model`. Returns the number of boxes written.

    `predict_kwargs` (e.g. conf, imgsz, device) are passed to the model; when omitted, Ultralytics uses the
    checkpoint defaults (imgsz from training, conf=0.25), which is how the published det.txt were produced.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    n = 0
    with open(out_file, "w") as f:
        for frame_idx, img_path in enumerate(list_frames(img_dir), start=1):
            results = model(img_path, verbose=False, **predict_kwargs)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                f.write(f"{frame_idx},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.4f},-1,-1,-1\n")
                n += 1
    return n
