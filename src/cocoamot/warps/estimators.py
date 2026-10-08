# SPDX-License-Identifier: AGPL-3.0-only
# Derived from Ultralytics GMC (ultralytics/trackers/utils/gmc.py, AGPL-3.0, https://ultralytics.com/license).
"""Warp estimators — a common interface for per-pair camera-motion estimation.

Every estimator implements the same contract:

    out = estimator.estimate(img_a_bgr, img_b_bgr, mask=None)   # -> WarpOutput
    out.H         # 2x3 float64 affine mapping a point in frame A to frame B (full-res)
    out.ok        # False if the estimator fell back to identity (a failure)
    out.n_inliers # inliers used by RANSAC (-1 when the method does not expose it)

`mask` (optional) is a full-resolution uint8 array, 255 = background (use), 0 = object
(exclude). It is the *uniform* masking channel: the same mask is honored by the
classic and the learned methods so the background-warp is estimated from static regions
only. Build it with `build_exclusion_mask` from the frame's detections.

Method families
---------------
Classic (reuse Ultralytics `GMC`, per-pair, so the output is bit-for-bit identical to the
streaming tracker GMC when no mask is used):
    sparseOptFlow          Lucas-Kanade optical flow  (MASKED — step 2 changes its output)
    sparseOptFlow_nomask   the exact original, no masking (to isolate the mask effect)
    orb, sift              feature matching + RANSAC   (masked)
    ecc                    Enhanced Correlation Coeff. (masked)
    <any>_nomask           the un-masked variant of any classic method

Learned (kornia / accelerated_features):
    loftr                  LoFTR (detector-free dense matcher, outdoor weights)
    disk_lightglue         DISK keypoints + LightGlue matcher
    xfeat                  XFeat (verlab/accelerated_features via torch.hub)

Learned methods: extract correspondences -> filter by confidence/mask -> scale to full-res
-> cv2.estimateAffinePartial2D(RANSAC) -> 2x3. On matcher error or too few inliers they
return identity and increment `fail_count`.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import cv2
import numpy as np

from ultralytics.trackers.utils.gmc import GMC

IDENTITY = np.eye(2, 3, dtype=np.float64)


@dataclass
class WarpOutput:
    """Result of a single pairwise warp estimation."""

    H: np.ndarray  # (2, 3) affine, frame A -> frame B, in full-resolution pixel coords
    ok: bool  # False => fell back to identity (failure)
    n_inliers: int  # RANSAC inliers, or -1 if not exposed by the method


# --------------------------------------------------------------------------------------
# Mask construction (uniform across every method)
# --------------------------------------------------------------------------------------
def build_exclusion_mask(
    shape_hw: tuple[int, int],
    boxes_xywh: np.ndarray | None,
    dilate: float = 0.10,
) -> np.ndarray | None:
    """Build a background mask (255 = keep/background, 0 = object) from MOT boxes.

    Args:
        shape_hw: (height, width) of the full-resolution frame.
        boxes_xywh: (N, 4) top-left boxes [x, y, w, h]; None/empty -> all-background mask.
        dilate: fractional box enlargement (0.10 => +10% on w and h, centered).

    Returns:
        uint8 (H, W) mask, or None if no boxes were given (caller may treat as "no mask").
    """
    h, w = shape_hw
    mask = np.full((h, w), 255, dtype=np.uint8)
    if boxes_xywh is None or len(boxes_xywh) == 0:
        return None
    for x, y, bw, bh in np.asarray(boxes_xywh, dtype=np.float64):
        cx, cy = x + bw / 2.0, y + bh / 2.0
        bw *= 1.0 + dilate
        bh *= 1.0 + dilate
        x1 = max(0, int(round(cx - bw / 2.0)))
        y1 = max(0, int(round(cy - bh / 2.0)))
        x2 = min(w, int(round(cx + bw / 2.0)))
        y2 = min(h, int(round(cy + bh / 2.0)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
    return mask


# --------------------------------------------------------------------------------------
# Classic estimators — subclass GMC so the un-masked path is byte-identical to the tracker
# --------------------------------------------------------------------------------------
class _MaskedGMC(GMC):
    """GMC that honors an external background mask (`self.ext_mask`, full-res, 255=keep).

    Every override reduces to the *exact* upstream code when `ext_mask is None`, so
    `<method>_nomask` reproduces the streaming GMC bit-for-bit. The only additions are:
    intersecting the downscaled `ext_mask` into each method's keypoint mask, and recording
    the RANSAC inlier count in `self.last_n_inliers`.
    """

    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2) -> None:
        super().__init__(method=method, downscale=downscale)
        self.ext_mask: np.ndarray | None = None
        self.last_n_inliers: int = -1

    def _ds_mask(self, width: int, height: int) -> np.ndarray | None:
        """Downscale `ext_mask` to (height, width) for a working-resolution frame."""
        if self.ext_mask is None:
            return None
        return cv2.resize(self.ext_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    def apply_sparseoptflow(self, raw_frame: np.ndarray) -> np.ndarray:
        """Sparse optical flow with optional background masking (else identical to GMC)."""
        height, width, c = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if c == 3 else raw_frame
        H = np.eye(2, 3)
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        sof_mask = self._ds_mask(frame.shape[1], frame.shape[0])
        keypoints = cv2.goodFeaturesToTrack(frame, mask=sof_mask, **self.feature_params)

        if not self.initializedFirstFrame or self.prevKeyPoints is None:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.initializedFirstFrame = True
            return H

        matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)
        prevPoints, currPoints = [], []
        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])
        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        if (prevPoints.shape[0] > 4) and (prevPoints.shape[0] == currPoints.shape[0]):
            H, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            self.last_n_inliers = int(inliers.sum()) if inliers is not None else -1
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            self.last_n_inliers = 0

        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        return H

    def apply_ecc(self, raw_frame: np.ndarray) -> np.ndarray:
        """ECC with optional background masking (else identical to GMC)."""
        height, width, c = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if c == 3 else raw_frame
        H = np.eye(2, 3, dtype=np.float32)
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.initializedFirstFrame = True
            return H

        ecc_mask = self._ds_mask(frame.shape[1], frame.shape[0])
        try:
            (_, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, ecc_mask, 1)
        except Exception:
            self.last_n_inliers = 0
            return H
        self.last_n_inliers = -1  # ECC is dense; no inlier concept
        return H

    def apply_features(self, raw_frame: np.ndarray, detections: list | None = None) -> np.ndarray:
        """ORB/SIFT with optional background masking (else identical to GMC)."""
        height, width, c = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if c == 3 else raw_frame
        H = np.eye(2, 3)
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        mask = np.zeros_like(frame)
        mask[int(0.02 * height) : int(0.98 * height), int(0.02 * width) : int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1] : tlbr[3], tlbr[0] : tlbr[2]] = 0
        ds = self._ds_mask(width, height)
        if ds is not None:
            mask = cv2.bitwise_and(mask, ds)

        keypoints = self.detector.detect(frame, mask)
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            self.initializedFirstFrame = True
            return H

        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)
        matches, spatialDistances = [], []
        maxSpatialDistance = 0.25 * np.array([width, height])
        if len(knnMatches) == 0:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt
                spatialDistance = (
                    prevKeyPointLocation[0] - currKeyPointLocation[0],
                    prevKeyPointLocation[1] - currKeyPointLocation[1],
                )
                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and (
                    np.abs(spatialDistance[1]) < maxSpatialDistance[1]
                ):
                    spatialDistances.append(spatialDistance)
                    matches.append(m)

        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)
        inliers = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances

        goodMatches, prevPoints, currPoints = [], [], []
        for i in range(len(matches)):
            if inliers[i, 0] and inliers[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)
        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        if prevPoints.shape[0] > 4:
            H, ransac_inl = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            self.last_n_inliers = int(ransac_inl.sum()) if ransac_inl is not None else -1
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            self.last_n_inliers = 0

        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)
        return H


class ClassicEstimator:
    """Per-pair wrapper over `_MaskedGMC`. `<method>_nomask` disables masking entirely."""

    CLASSIC = {"sparseOptFlow", "orb", "sift", "ecc"}

    def __init__(self, method: str, downscale: int = 2):
        self.method = method
        self.use_mask = not method.endswith("_nomask")
        self.base_method = method[: -len("_nomask")] if not self.use_mask else method
        if self.base_method not in self.CLASSIC:
            raise ValueError(f"Unknown classic method: {method}")
        self.downscale = downscale
        self.n_calls = 0
        self.fail_count = 0
        self.total_time = 0.0

    def estimate(self, img_a_bgr: np.ndarray, img_b_bgr: np.ndarray, mask: np.ndarray | None = None) -> WarpOutput:
        """Estimate the A->B warp for one pair (identical to streaming GMC when un-masked)."""
        t0 = time.perf_counter()
        gmc = _MaskedGMC(method=self.base_method, downscale=self.downscale)
        gmc.ext_mask = mask if self.use_mask else None
        gmc.apply(img_a_bgr)  # initialize on frame A
        H = np.asarray(gmc.apply(img_b_bgr), dtype=np.float64)  # warp A -> B
        self.total_time += time.perf_counter() - t0
        self.n_calls += 1
        # GMC signals failure by returning identity; treat an exact identity as a failure.
        ok = not np.array_equal(H, IDENTITY)
        if not ok:
            self.fail_count += 1
        return WarpOutput(H=H, ok=ok, n_inliers=int(gmc.last_n_inliers))


# --------------------------------------------------------------------------------------
# Learned estimators — kornia LoFTR / DISK+LightGlue, XFeat via torch.hub
# --------------------------------------------------------------------------------------
_MODEL_CACHE: dict = {}  # (kind, device) -> loaded model(s); shared across estimators


def _get_device(device: str | None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class LearnedEstimator:
    """Correspondence-based warp from a learned matcher: match -> filter -> RANSAC affine."""

    KINDS = {"loftr", "disk_lightglue", "xfeat"}

    def __init__(
        self,
        kind: str,
        downscale: int = 2,
        device: str | None = None,
        min_conf: float = 0.5,
        min_inliers: int = 15,
        ransac_thresh: float = 3.0,
        max_kpts: int = 2048,
    ):
        if kind not in self.KINDS:
            raise ValueError(f"Unknown learned method: {kind}")
        self.kind = kind
        self.downscale = downscale
        self.device = _get_device(device)
        self.min_conf = min_conf
        self.min_inliers = min_inliers
        self.ransac_thresh = ransac_thresh
        self.max_kpts = max_kpts
        self.n_calls = 0
        self.fail_count = 0
        self.total_time = 0.0
        self._load()

    # -- model loading -----------------------------------------------------------------
    def _load(self):
        key = (self.kind, self.device)
        if key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[key]
            return
        import torch

        if self.kind == "loftr":
            import kornia.feature as KF

            m = KF.LoFTR(pretrained="outdoor").eval().to(self.device)
            self._model = m
        elif self.kind == "disk_lightglue":
            import kornia.feature as KF

            disk = KF.DISK.from_pretrained("depth").eval().to(self.device)
            lg = KF.LightGlueMatcher("disk").eval().to(self.device)
            self._model = (disk, lg)
        elif self.kind == "xfeat":
            self._model = torch.hub.load(
                "verlab/accelerated_features", "XFeat", pretrained=True, top_k=4096, trust_repo=True
            )
        _MODEL_CACHE[key] = self._model

    # -- helpers -----------------------------------------------------------------------
    def _resize(self, img_bgr: np.ndarray):
        """Return (resized RGB uint8, sx, sy) where (x_full, y_full) = (x_ds*sx, y_ds*sy)."""
        h, w = img_bgr.shape[:2]
        wds, hds = w // self.downscale, h // self.downscale
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (wds, hds))
        return rgb, w / wds, h / hds

    def _match(self, rgb_a: np.ndarray, rgb_b: np.ndarray):
        """Run the matcher; return (pts_a, pts_b, conf) in downscaled-image coords."""
        import torch

        if self.kind == "loftr":
            import kornia as K
            import kornia.feature as KF  # noqa: F401

            def gray(rgb):
                t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(self.device) / 255.0
                return K.color.rgb_to_grayscale(t)

            with torch.inference_mode():
                out = self._model({"image0": gray(rgb_a), "image1": gray(rgb_b)})
            pa = out["keypoints0"].cpu().numpy()
            pb = out["keypoints1"].cpu().numpy()
            conf = out["confidence"].cpu().numpy()
            return pa, pb, conf

        if self.kind == "disk_lightglue":
            import kornia.feature as KF

            disk, lg = self._model

            def to_t(rgb):
                return torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(self.device) / 255.0

            ta, tb = to_t(rgb_a), to_t(rgb_b)
            with torch.inference_mode():
                fa = disk(ta, self.max_kpts, pad_if_not_divisible=True)[0]
                fb = disk(tb, self.max_kpts, pad_if_not_divisible=True)[0]
                lafs_a = KF.laf_from_center_scale_ori(fa.keypoints[None])
                lafs_b = KF.laf_from_center_scale_ori(fb.keypoints[None])
                hw_a = torch.tensor(ta.shape[-2:], device=self.device)
                hw_b = torch.tensor(tb.shape[-2:], device=self.device)
                _, idxs = lg(fa.descriptors, fb.descriptors, lafs_a, lafs_b, hw1=hw_a, hw2=hw_b)
            if idxs.shape[0] == 0:
                return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))
            idxs = idxs.cpu().numpy()
            pa = fa.keypoints.cpu().numpy()[idxs[:, 0]]
            pb = fb.keypoints.cpu().numpy()[idxs[:, 1]]
            return pa, pb, np.ones(len(pa))  # LightGlue matches are already mutually filtered

        # xfeat
        mk0, mk1 = self._model.match_xfeat(rgb_a, rgb_b, top_k=4096)
        mk0 = np.asarray(mk0)
        mk1 = np.asarray(mk1)
        return mk0, mk1, np.ones(len(mk0))  # XFeat returns mutual matches, no per-match conf

    # -- main --------------------------------------------------------------------------
    def estimate(self, img_a_bgr: np.ndarray, img_b_bgr: np.ndarray, mask: np.ndarray | None = None) -> WarpOutput:
        """Estimate the A->B warp via learned correspondences; identity on failure."""
        t0 = time.perf_counter()
        self.n_calls += 1
        try:
            rgb_a, sx, sy = self._resize(img_a_bgr)
            rgb_b, _, _ = self._resize(img_b_bgr)
            pa, pb, conf = self._match(rgb_a, rgb_b)

            if len(pa) > 0 and self.min_conf > 0 and conf is not None:
                keep = conf >= self.min_conf
                pa, pb = pa[keep], pb[keep]

            # Scale correspondences back to full resolution.
            scale = np.array([sx, sy], dtype=np.float64)
            pa = pa.astype(np.float64) * scale
            pb = pb.astype(np.float64) * scale

            # Uniform masking: drop matches whose source point falls on an object.
            if mask is not None and len(pa) > 0:
                h, w = mask.shape
                xi = np.clip(pa[:, 0].astype(int), 0, w - 1)
                yi = np.clip(pa[:, 1].astype(int), 0, h - 1)
                keep = mask[yi, xi] > 0
                pa, pb = pa[keep], pb[keep]

            if len(pa) < max(4, self.min_inliers):
                self.fail_count += 1
                return self._done(IDENTITY.copy(), False, len(pa), t0)

            H, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh)
            n_inl = int(inl.sum()) if inl is not None else 0
            if H is None or n_inl < self.min_inliers:
                self.fail_count += 1
                return self._done(IDENTITY.copy(), False, n_inl, t0)
            return self._done(np.asarray(H, dtype=np.float64), True, n_inl, t0)
        except Exception as e:  # matcher blew up on this pair -> identity, keep going
            from ultralytics.utils import LOGGER

            LOGGER.warning(f"[{self.kind}] pair failed ({e}); using identity")
            self.fail_count += 1
            return self._done(IDENTITY.copy(), False, 0, t0)

    def _done(self, H, ok, n_inliers, t0):
        self.total_time += time.perf_counter() - t0
        return WarpOutput(H=H, ok=ok, n_inliers=n_inliers)


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------
def make_estimator(method: str, downscale: int = 2, **kwargs):
    """Return an estimator exposing `.estimate(imgA, imgB, mask) -> WarpOutput`.

    Classic: sparseOptFlow[_nomask], orb[_nomask], sift[_nomask], ecc[_nomask].
    Learned: loftr, disk_lightglue, xfeat.
    """
    base = method[: -len("_nomask")] if method.endswith("_nomask") else method
    if base in ClassicEstimator.CLASSIC:
        return ClassicEstimator(method, downscale=downscale)
    if method in LearnedEstimator.KINDS:
        return LearnedEstimator(method, downscale=downscale, **kwargs)
    raise ValueError(f"Unknown warp method: {method}")
