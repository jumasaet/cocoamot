# CocoaMOT dataset

CocoaMOT is a paired **synthetic–real benchmark for multi-object tracking of cacao pods** filmed from a moving
ground vehicle. The pods are static; all image motion is camera motion. The detectors distinguish two classes
(**healthy / diseased**); the tracking annotations merge both into a single cacao class.

This folder holds the lightweight files of every sequence (ground truth, detections, sequence info). The images
are distributed separately because of their size — see [Downloading the images](#downloading-the-images).

```
data/
├── synthetic/seq_0001 … seq_0010/        rendered in Isaac Sim
├── real/verified/seq_0001, 0006, 0007, 0008/        the 4 real sequences used in the paper
└── real/pre_annotated/seq_0002, 0003, 0004, 0005/   extra real sequences, NOT used in the paper
        each sequence:  gt/gt.txt   det/det.txt   seqinfo.ini   img1/ (images, downloaded separately)
```

## Splits

| Split | Sequences | Resolution | FPS | Frames | Boxes | IDs | Source |
|---|---|---|---|---|---|---|---|
| `synthetic/` | 10 (`seq_0001`–`seq_0010`) | 1024×768 | 20 | 7,990 | 229,832 (54,815 evaluated + 175,017 ignored) | 744 | Isaac Sim; ground truth and per-object visibility from the renderer |
| `real/verified/` | 4 (`seq_0001`, `0006`, `0007`, `0008`) | 1920×1080 | 20 | 3,309 | 51,880 | 638 | Pixel 8; manual annotation verified by 2 annotators |
| `real/pre_annotated/` | 4 (`seq_0002`, `0003`, `0004`, `0005`) | 1920×1080 | 20 | 3,070 | 67,584 | 8,589 tracklets | Same camera; pre-annotated via CVAT, **pending manual correction** |

`real/pre_annotated/` is included to support future work and semi-supervised approaches. It is **NOT used in any
experiment reported in the paper**. Its `gt.txt` is the raw pre-annotation: detector outputs (confidence ≥ 0.25)
linked into tracklets by a greedy IoU tracker, exactly as imported into CVAT. Expect missed and false boxes, and
fragmented identities — the 8,589 ids are short tracklets, not object identities.

<details>
<summary>Per-sequence statistics</summary>

| Sequence | Frames | GT boxes | IDs | Evaluated / ignored | Detections |
|---|---|---|---|---|---|
| synthetic/seq_0001 | 950 | 9,646 | 35 | 4,249 / 5,397 | 4,540 |
| synthetic/seq_0002 | 360 | 2,532 | 10 | 1,108 / 1,424 | 1,427 |
| synthetic/seq_0003 | 480 | 3,287 | 19 | 2,214 / 1,073 | 2,584 |
| synthetic/seq_0004 | 540 | 2,451 | 11 | 1,464 / 987 | 1,799 |
| synthetic/seq_0005 | 900 | 46,478 | 150 | 2,851 / 43,627 | 6,052 |
| synthetic/seq_0006 | 1,010 | 38,151 | 119 | 5,984 / 32,167 | 8,818 |
| synthetic/seq_0007 | 1,170 | 64,574 | 219 | 11,942 / 52,632 | 17,193 |
| synthetic/seq_0008 | 620 | 12,534 | 47 | 5,007 / 7,527 | 6,553 |
| synthetic/seq_0009 | 830 | 26,794 | 74 | 5,496 / 21,298 | 8,074 |
| synthetic/seq_0010 | 1,130 | 23,385 | 60 | 14,500 / 8,885 | 16,923 |
| real/verified/seq_0001 | 765 | 31,111 | 363 | all / 0 | 25,829 |
| real/verified/seq_0006 | 839 | 3,571 | 58 | all / 0 | 3,337 |
| real/verified/seq_0007 | 872 | 10,369 | 101 | all / 0 | 9,076 |
| real/verified/seq_0008 | 833 | 6,829 | 116 | all / 0 | 5,452 |
| real/pre_annotated/seq_0002 | 740 | 19,364 | 2,272 | all / 0 | 19,364 |
| real/pre_annotated/seq_0003 | 426 | 5,876 | 521 | all / 0 | 5,876 |
| real/pre_annotated/seq_0004 | 865 | 19,642 | 2,388 | all / 0 | 19,642 |
| real/pre_annotated/seq_0005 | 1,039 | 22,702 | 3,408 | all / 0 | 22,702 |

</details>

## File formats

All files follow the **MOTChallenge** text format: one box per line, comma-separated, frames and ids 1-indexed,
boxes in pixels with a top-left origin.

**`gt/gt.txt`**

| # | Column | Meaning |
|---|---|---|
| 1 | `frame` | Frame number (1 = first image of `img1/`, sorted by file name) |
| 2 | `id` | Track identity, unique within the sequence |
| 3–4 | `x`, `y` | Top-left corner of the box |
| 5–6 | `w`, `h` | Box width and height |
| 7 | `conf` | Consider flag, always `1` |
| 8 | `class` | `1` = cacao pod (healthy + diseased combined); `7` = ignore (synthetic only) |
| 9 | `visibility` | Visible fraction of the object in [0, 1] |
| 10 | `z` | Unused, `-1` (synthetic only) |

- **Synthetic:** 10 columns. `visibility` is available per object, from the Isaac Sim renderer. `class = 7` marks
  boxes excluded from evaluation: smaller than 2 % of the frame (width < 20.48 px or height < 15.36 px) or more
  than 80 % occluded (`visibility` < 0.2). Class 7 is a TrackEval distractor class: those boxes are ignored, not
  penalized. Setting every `class` back to `1` recovers the unfiltered ground truth.
- **Real:** 9 columns (no `z`). Visibility is **not** annotated: the column is the constant `1.0`. There is no
  ignore class; every box is evaluated.

**`det/det.txt`** — `frame, -1, x, y, w, h, conf, -1, -1, -1`: detector outputs, `conf` is the detection
confidence (boxes with confidence ≥ 0.25).

> The det/det.txt files are the canonical detector outputs used in all experiments. Re-running the provided
> detector weights may produce slightly different detections (see Section 3.2.1 of the paper); use the cached
> det.txt for exact reproduction of reported metrics.

Exception: in `real/pre_annotated/`, `det.txt` holds the same boxes as `gt.txt` (the pre-annotation *is* the
detector output). Their confidences were not stored, so `conf` is set to `1.0000` for every box.

**`seqinfo.ini`** — sequence name, image folder (`img1`), frame rate, length, image size and extension
(`.png` synthetic, `.jpg` real).

## Downloading the images

Images are hosted at **[PLACEHOLDER_URL]** (Zenodo / Hugging Face), one archive per split. Each archive already
contains the folder structure of this directory, so it is extracted on top of `data/`:

```bash
# run from the repository root
wget [PLACEHOLDER_URL]/cocoamot_synthetic_images.zip
wget [PLACEHOLDER_URL]/cocoamot_real_verified_images.zip
wget [PLACEHOLDER_URL]/cocoamot_real_pre_annotated_images.zip      # optional, not needed for the paper

unzip cocoamot_synthetic_images.zip          -d data/   # -> data/synthetic/seq_XXXX/img1/000001.png ...
unzip cocoamot_real_verified_images.zip      -d data/   # -> data/real/verified/seq_XXXX/img1/000001.jpg ...
unzip cocoamot_real_pre_annotated_images.zip -d data/   # -> data/real/pre_annotated/seq_XXXX/img1/000001.jpg ...
```

Images are named with six digits starting at `000001`, so that sorting by name gives the frame order. Check the
download — every sequence must have exactly `seqlength` images:

```bash
for d in data/synthetic/seq_* data/real/*/seq_*; do
  n=$(find "$d/img1" -name '*.png' -o -name '*.jpg' | wc -l)
  len=$(grep -i '^seqlength' "$d/seqinfo.ini" | tr -dc 0-9)
  [ "$n" = "$len" ] && echo "ok   $d ($n)" || echo "FAIL $d ($n images, expected $len)"
done
```

Until the images are downloaded, each `img1/` only contains a `DOWNLOAD.md` pointer.

## License

The CocoaMOT dataset (images, ground truth and detections — synthetic, `real/verified/` and
`real/pre_annotated/` alike) is released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

The real sequences were captured and annotated by the authors. A detector checkpoint provided by the AIRIX
project (`weights/real_yolo12m/best.pt`, not part of this dataset) was used only to speed up pre-annotation;
every real annotation was then corrected and manually verified in full by the authors. That checkpoint has
its own, separate and more restrictive license, which does **not** apply to this dataset — see
[ACKNOWLEDGMENTS.md](../ACKNOWLEDGMENTS.md) and the [root README's License section](../README.md#license).
