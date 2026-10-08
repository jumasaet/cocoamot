# CocoaMOT: ego-motion gap interpolation and track stitching for static objects

Code for the paper **TODO: title** (WACV 2027, TODO: authors / link).

Static objects (cacao pods) filmed from a moving ground vehicle: every image motion is camera motion. Two
tracker-agnostic post-processing steps exploit this:

- **EMTS — Ego-Motion Track Stitching** (`src/cocoamot/emts.py`): reconnects broken tracks by projecting the
  end of each track with the camera warps and matching it (warped IoU + Hungarian) to tracks that start later.
- **EMGI — Ego-Motion Gap Interpolation** (`src/cocoamot/emgi.py`): fills the closed gaps of every track by
  propagating the box with the camera warps (forward/backward fusion) instead of a straight line.

Both run on MOTChallenge `.txt` outputs of any tracker (BoT-SORT, ByteTrack, OC-SORT, Deep OC-SORT, AGT).

```
configs/trackers/       tracker settings used in the paper (BoT-SORT verified, see "BoT-SORT" below)
configs/experiments/    one YAML per paper table/figure
patches/                ultralytics_ours.patch (applied on top of ultralytics==8.4.84)
src/cocoamot/           io, warps/ (estimators, cache), emts, emgi, tracking, detection, gt, evaluation, significance
scripts/                CLI entry points; run_experiment.py reproduces a full table
tools/                  warp accuracy, dataset statistics, bounding-box statistics
third_party/README.md   AGT and AppleMOTS
data/                   the CocoaMOT dataset: GT, detections, sequence info (images downloaded separately)
```

## Installation

Python ≥ 3.10. Tested with Python 3.11, PyTorch 2.14 (CUDA 12.6), Ultralytics 8.4.84.

```bash
conda create -n cocoamot python=3.11 -y && conda activate cocoamot
pip install torch torchvision                      # pick the build for your CUDA: https://pytorch.org
pip install -e ".[eval]"                           # ultralytics==8.4.84 + TrackEval (pinned commit)
pip install -e ".[learned]"                        # optional: LoFTR / DISK+LightGlue / XFeat warps (GMC ablation)

# apply the tracker patch to the installed ultralytics (required for every BoT-SORT result)
patch -p1 -d "$(python -c 'import ultralytics, os; print(os.path.dirname(os.path.dirname(ultralytics.__file__)))')" \
    < patches/ultralytics_ours.patch
python -c "from ultralytics.trackers.byte_tracker import BYTETracker; assert hasattr(BYTETracker, '_init_ablation_panel'); print('patch OK')"
```

The patch targets the PyPI release `ultralytics==8.4.84` (identical, for the five patched files, to commit
`cc8576c36f881e474a034ba5e1a534f89720bbf0`). For a source checkout use `git apply patches/ultralytics_ours.patch`.
The scripts refuse to run a tracker config that uses the patch keys on an unpatched install.

## Data and weights

The **CocoaMOT dataset** lives in [`data/`](data/README.md): ground truth, detections and sequence info of every
sequence are part of this repository; only the images are downloaded separately.

| Artifact | Where | License |
|---|---|---|
| CocoaMOT-synthetic: 10 sequences (GT, detections) | [`data/synthetic/`](data/README.md) — images: TODO Zenodo/HF URL | CC-BY-4.0 |
| CocoaMOT-real, verified: the 4 sequences used in the paper | [`data/real/verified/`](data/README.md) — images: TODO URL | CC-BY-4.0 |
| CocoaMOT-real, pre-annotated: 4 extra sequences, **not used in the paper** | [`data/real/pre_annotated/`](data/README.md) — images: TODO URL | CC-BY-4.0 |
| AppleMOTS (6 test sequences) | public dataset, goes to `data/applemot/` — see [third_party/README.md](third_party/README.md) | AppleMOTS license |
| Real detector `weights/real_yolo12m/best.pt` (YOLOv12m, 1024 px), provided by AIRIX | TODO: URL | **AIRIX terms: academic/non-commercial only, prior written consent required for other use** — see [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) |
| Synthetic detector `weights/synthetic_yolov8m/best.pt` (YOLOv8m), trained by the authors | TODO: URL | CC-BY-4.0 |

The real CocoaMOT sequences were captured and annotated by the authors; the AIRIX-provided detector weights
were used only to speed up pre-annotation, which was then fully corrected and verified by the authors (see
[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md)). The dataset's CC-BY-4.0 license is independent of the separate,
more restrictive license on that one weights file.

### Zenodo archive

The full dataset (18 sequences, 21.1 GB: images + annotations, zipped per sequence) is archived on Zenodo:
**[[CocoaMOT Dataset](https://doi.org/10.5281/zenodo.23201170)]**. See the README inside that deposit for the file format and per-split details — it's the
same content as [data/README.md](data/README.md), reproduced there so the deposit is self-contained.

```
data/                                     see data/README.md for formats, statistics and image download
├── synthetic/seq_0001 … seq_0010/        gt/gt.txt  det/det.txt  seqinfo.ini  img1/ (download)
├── real/verified/seq_0001, 0006, 0007, 0008/
├── real/pre_annotated/seq_0002 … 0005/
└── applemot/0006, 0007, 0008, 0010, 0011, 0012/     (third-party, arrange it yourself)
weights/real_yolo12m/best.pt, weights/synthetic_yolov8m/best.pt
```

**The published `det/det.txt` are the canonical input.** All tracking results start from them, so the detector
is not needed to reproduce the tables. The released real detector is equivalent but *not* the exact model that
produced the real `det.txt` (AP50 0.819 vs 0.828 on the real GT); re-detecting with it (`--steps detect,...`)
gives slightly different tracking numbers.

## Reproducing the paper

Download the images first ([data/README.md](data/README.md)). One entry point per table/figure; results go to
`outputs/<experiment>/<dataset>/<run>/metrics*.csv` (per sequence + `COMBINED`; HOTA is TrackEval's `HOTA___AUC`).

```bash
python scripts/run_experiment.py --config configs/experiments/main.yaml
```

The dataset root defaults to `data/`; use `--data_root` or the `COCOAMOT_DATA` environment variable to point
elsewhere.

| Paper | Content | Command |
|---|---|---|
| Table TODO | Main results: 4 trackers + AGT × {base, EMTS, EMGI, EMTS+EMGI} on synthetic, real, AppleMOTS | `python scripts/run_experiment.py --config configs/experiments/main.yaml` |
| Table TODO | Camera-motion estimator ablation (sparseOptFlow, ORB, SIFT, ECC, LoFTR, DISK+LightGlue, XFeat) | `python scripts/run_experiment.py --config configs/experiments/gmc_ablation.yaml` |
| Table/Fig. TODO | EMGI max-gap sweep (2–60) and warp vs. linear interpolation | `python scripts/run_experiment.py --config configs/experiments/gap_ablation.yaml` |
| Table/Fig. TODO | EMTS warped-IoU threshold τ sweep (0.00–0.55), real set | `python scripts/run_experiment.py --config configs/experiments/tau_ablation.yaml` |
| Table/Fig. TODO | Metrics per occlusion bucket + combined 0–100 % + Wilcoxon / Holm | `python scripts/run_experiment.py --config configs/experiments/occlusion.yaml` |
| Table TODO | Warp accuracy vs. ground-truth ego-motion (error after K frames, failures, ms/pair) | `python tools/warp_eval.py --seqs data/synthetic/seq_0001 data/real/verified/seq_0007 --out outputs/warp_eval --cache_root outputs/warp_eval_cache` |
| Table TODO | Dataset statistics | `python tools/dataset_statistics.py data/real/verified --fps 20` |
| Fig. TODO | Bounding-box size distribution | `python tools/bbox_stats.py data/real/verified --out_plots outputs/bbox_plots` |

AGT is an external tracker: put its outputs in `outputs/<experiment>/<dataset>/tracks/agt/` first
([third_party/README.md](third_party/README.md)). Useful flags: `--steps evaluate` (re-evaluate only),
`--datasets real`, `--sequences seq_0006`, `--out <dir>`.

### Running stages by hand

```bash
R=data/real/verified
python scripts/detect.py      --model weights/real_yolo12m/best.pt --dataset $R --out_root outputs/manual/det
python scripts/track.py       --dataset $R --trackers botsort --out_root outputs/manual/tracks
python scripts/compute_warps.py --dataset $R --methods sparseOptFlow --no_mask --cache_root outputs/manual/warps
python scripts/run_emts.py    --input_txt outputs/manual/tracks/botsort/seq_0006.txt --img_dir $R/seq_0006/img1 \
                              --cache_dir outputs/manual/warps/seq_0006 --output_txt outputs/manual/emts/botsort/seq_0006.txt
python scripts/run_emgi.py    --input_txt outputs/manual/emts/botsort/seq_0006.txt --img_dir $R/seq_0006/img1 \
                              --cache_dir outputs/manual/warps/seq_0006 --max_gap 8 --no_mask --output_txt outputs/manual/emts_emgi/botsort/seq_0006.txt
python scripts/evaluate.py    --gt_root $R --seqs seq_0006 \
                              --entry botsort:base=outputs/manual/tracks/botsort --entry botsort:emts_emgi=outputs/manual/emts_emgi/botsort \
                              --out_csv outputs/manual/metrics.csv
python scripts/prepare_gt.py interval --dataset data/synthetic --dst gt/gt_occ_20_40.txt --min_occ 0.2 --max_occ 0.4 --min_wh 0.02
python scripts/significance.py --bucket 00_20=... --bucket 80_100=... --test_buckets 00_20,20_40,40_60,60_80 --out_dir outputs/sig
```

## Evaluation protocol

- TrackEval `MotChallenge2DBox`, metrics HOTA, CLEAR, Identity (IoU 0.5), commit `12c8791`.
- Synthetic GT (`data/synthetic/*/gt/gt.txt`): boxes smaller than 2 % of the frame (width or height) or more than
  80 % occluded are class 7, a TrackEval distractor class (ignored, not penalized). Real and AppleMOTS GT are
  used as annotated.
- Occlusion buckets: only boxes with occlusion in [lo, hi] (both inclusive) and size ≥ 2 % are evaluated.

## BoT-SORT

All BoT-SORT results use `configs/trackers/botsort.yaml`: Ultralytics BoT-SORT (sparse optical-flow GMC, no ReID)
**plus the static-object velocity prior `static_prior_beta: 0.2`** (and `min_wh_clamp: 5.0`) from the patch.
For vanilla BoT-SORT set both to `0.0`. ByteTrack, OC-SORT and Deep OC-SORT use the stock Ultralytics 8.4.84 configs.

## Reproducibility notes

- Tracking, EMTS and EMGI are deterministic given the same detections and library versions. RANSAC in the
  camera-motion estimate uses OpenCV's fixed-seed RNG; different OpenCV/NumPy builds can change a few boxes by
  a rounding unit (0.01 px).
- Exact Wilcoxon p-values with tied |deltas| (integer metrics such as IDSW) depend on SciPy: SciPy ≥ 1.13 handles
  ties exactly, SciPy ≤ 1.12 used the tie-free distribution.

## License

- **Code:** MIT ([LICENSE](LICENSE)), except files derived from Ultralytics, which are AGPL-3.0
  ([LICENSES/AGPL-3.0.txt](LICENSES/AGPL-3.0.txt)): `patches/ultralytics_ours.patch`,
  `src/cocoamot/warps/estimators.py` and `configs/trackers/*.yaml`. This code depends on Ultralytics (AGPL-3.0).
- **CocoaMOT dataset** (`data/`: synthetic and real, images and annotations, including `pre_annotated/`) and
  the **synthetic detector weights** (`weights/synthetic_yolov8m/best.pt`, trained by the authors):
  [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
- **Real detector weights** (`weights/real_yolo12m/best.pt`, provided by AIRIX): a separate, more restrictive
  license — academic/non-commercial research use only, prior written consent from AIRIX required for any
  other use. See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md). These terms apply **only** to this one file; they
  do not extend to the dataset, which is CC-BY-4.0 regardless of which weights produced its detections.

## Citation
If you use CacaoMOT in your research, please cite: 

```bibtex
@dataset{saeteros2026cacaomot,
  author    = {Saeteros, Juan and
               Arévalo Ronquillo, Nick Joel and
               Estrada Santana, Michael Bryan and
               Vintimilla, Boris and
               Romero, Dennis},
  title     = {{CacaoMOT: A Paired Synthetic-Real Multi-Object Tracking
                Dataset for Cocoa Pods}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23201170},
  url       = {https://doi.org/10.5281/zenodo.23201170}
}
```




