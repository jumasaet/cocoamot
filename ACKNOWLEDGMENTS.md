# Acknowledgments

## AIRIX

We gratefully acknowledge the **AIRIX project** for providing the pre-trained real-domain detector weights
(`weights/real_yolo12m/best.pt`, YOLOv12m) used in this work.

**Terms of use — detector weights only:** "The provided model weights... are released strictly for academic,
educational, and non-commercial research purposes. Any commercial deployment, redistribution, or derivation
requires prior written consent from the AIRIX project." These terms apply **exclusively** to the AIRIX-provided
real detector weights (`weights/real_yolo12m/best.pt`). They do **not** apply to the CocoaMOT dataset
(synthetic or real) or to the synthetic detector weights, which the authors release under CC-BY-4.0 — see the
[License section of the README](README.md#license).

We additionally thank AIRIX: these weights allowed us to efficiently pre-annotate the real CocoaMOT video
sequences. The resulting pre-annotations were subsequently corrected and manually verified in full by the
authors of this paper. The real CocoaMOT dataset — video capture, the pre-annotation pipeline, and the final,
verified annotations — is original work of the authors, not of AIRIX.

To learn more about their initiatives, visit the [AIRIX Project](https://www.airixtech.com/).



## Third-party software and data

- [Ultralytics](https://github.com/ultralytics/ultralytics) 8.4.84 (AGPL-3.0): detectors and trackers
  (BoT-SORT, ByteTrack, OC-SORT, Deep OC-SORT). Patched by `patches/ultralytics_ours.patch`.
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) (MIT), commit `12c8791`: HOTA / CLEAR / Identity.
- [kornia](https://github.com/kornia/kornia) (Apache-2.0): LoFTR and DISK + LightGlue matchers.
- [XFeat](https://github.com/verlab/accelerated_features) (Apache-2.0), loaded with `torch.hub`.
- AGT and AppleMOTS: see `third_party/README.md`.
