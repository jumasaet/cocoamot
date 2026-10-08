"""Occlusion-bucket aggregation and paired significance tests (EMGI / EMTS vs the base tracker).

1. Combine the per-bucket evaluations into a 0-100% table: CLR_TP / CLR_FN / IDSW are additive and summed;
   CLR_FP is reconstructed per bucket (from CLR_Pr, or from MOTA when Pr = 0) and summed; CLR_Re, CLR_Pr and
   MOTA are recomputed from the sums. HOTA and IDF1 are NOT additive: they are approximated by a GT-weighted
   mean over buckets (GT = CLR_TP + CLR_FN) — an approximation, not what TrackEval would give on the union.
2. Per tracker (and pooled over trackers), paired deltas (variant - base) per sequence: mean ± sd,
   win/loss/tie and an exact two-sided Wilcoxon signed-rank test (ties dropped), Holm-Bonferroni corrected
   within each (level, metric) family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ["Tracker", "Variant", "Sequence"]
DEFAULT_LABELS = {"emgi": "EMGI", "emts": "EMTS", "emts_emgi": "EMGI+EMTS"}
LEGACY_VARIANTS = {"Base": "base", "sof": "emgi", "stit": "emts", "stit_sof": "emts_emgi"}
COMBINED_METRICS = [("HOTA", True), ("IDF1", True), ("IDSW", False)]
BUCKET_METRICS = [("HOTA", True), ("IDF1", True)]
POOLED = "ALL (pooled)"


def load_bucket_csv(path: str, legacy: bool = False) -> pd.DataFrame:
    """Load one bucket's per-sequence metrics (output of scripts/evaluate.py, COMBINED rows dropped).

    legacy=True reads the historical format (';' separator, ',' decimals, Spanish headers and variants).
    """
    if legacy:
        df = pd.read_csv(path, sep=";", decimal=",")
        df = df.rename(columns={"Variante": "Variant", "Secuencia": "Sequence"})
        df["Variant"] = df["Variant"].map(lambda v: LEGACY_VARIANTS.get(v, v))
    else:
        df = pd.read_csv(path)
    return df[df["Sequence"].astype(str).str.upper() != "COMBINED"].reset_index(drop=True)


def combine(df_all: pd.DataFrame) -> pd.DataFrame:
    """Sum the additive metrics over buckets and approximate HOTA/IDF1 by a GT-weighted mean."""
    df = df_all.copy()
    df["GT"] = df["CLR_TP"] + df["CLR_FN"]
    df["HOTA_w"] = df["HOTA"] * df["GT"]
    df["IDF1_w"] = df["IDF1"] * df["GT"]

    g = df.groupby(KEYS, as_index=False).agg(
        CLR_TP=("CLR_TP", "sum"),
        CLR_FN=("CLR_FN", "sum"),
        IDSW=("IDSW", "sum"),
        GT=("GT", "sum"),
        HOTA_w=("HOTA_w", "sum"),
        IDF1_w=("IDF1_w", "sum"),
        n_buckets=("bucket", "count"),
    )

    # FP per bucket: FP = TP*(1-Pr)/Pr; where TP = Pr = 0 that is 0/0, so solve it from
    # MOTA = 1 - (FN + FP + IDSW)/GT instead (exact whenever GT > 0).
    fp = df.copy()
    gt_b = fp["CLR_TP"] + fp["CLR_FN"]
    fp_from_pr = np.where(fp["CLR_Pr"] > 0, fp["CLR_TP"] * (1.0 - fp["CLR_Pr"]) / fp["CLR_Pr"], np.nan)
    fp_from_mota = np.where(gt_b > 0, (1.0 - fp["MOTA"]) * gt_b - fp["CLR_FN"] - fp["IDSW"], np.nan)
    fp["CLR_FP"] = np.where(np.isnan(fp_from_pr), np.nan_to_num(fp_from_mota), fp_from_pr)
    fp["CLR_FP"] = fp["CLR_FP"].clip(lower=0.0)
    fp = fp.groupby(KEYS, as_index=False)["CLR_FP"].sum()
    g = g.merge(fp, on=KEYS)

    g["CLR_Re"] = g["CLR_TP"] / g["GT"].where(g["GT"] > 0)
    g["CLR_Pr"] = g["CLR_TP"] / (g["CLR_TP"] + g["CLR_FP"]).where((g["CLR_TP"] + g["CLR_FP"]) > 0)
    g["HOTA"] = g["HOTA_w"] / g["GT"].where(g["GT"] > 0)
    g["IDF1"] = g["IDF1_w"] / g["GT"].where(g["GT"] > 0)
    g["MOTA"] = 1.0 - (g["CLR_FN"] + g["CLR_FP"] + g["IDSW"]) / g["GT"].where(g["GT"] > 0)

    cols = KEYS + ["CLR_Re", "HOTA", "IDF1", "IDSW", "CLR_Pr", "CLR_TP", "CLR_FN", "CLR_FP", "MOTA", "GT"]
    return g[cols].sort_values(KEYS, ignore_index=True)


def p_min(n: int) -> float:
    """Smallest two-sided p reachable by an exact Wilcoxon test with n non-zero pairs."""
    return 2.0 / (2**n) if n > 0 else np.nan


def test_pair(base, treat, higher_is_better: bool = True) -> dict:
    """Paired test of treat vs base: mean ± sd of the delta, win/loss/tie, two-sided Wilcoxon p."""
    d = np.asarray(treat, float) - np.asarray(base, float)
    d = d[~np.isnan(d)]
    n = len(d)
    sign = 1.0 if higher_is_better else -1.0
    # Ties are dropped by hand: scipy falls back to the normal approximation when zeros are present.
    dnz = d[d != 0]
    nz = len(dnz)
    p = np.nan if nz == 0 else wilcoxon(dnz, alternative="two-sided", method="exact" if nz <= 25 else "approx").pvalue
    return {
        "n": n,
        "n_nonzero": nz,
        "mean": float(np.mean(d)) if n else np.nan,
        "sd": float(np.std(d, ddof=1)) if n > 1 else np.nan,
        "win": int(np.sum(sign * d > 0)),
        "loss": int(np.sum(sign * d < 0)),
        "tie": int(np.sum(d == 0)),
        "p": p,
        "p_min": p_min(nz),
    }


def rows_for(df, level, tracker_specs, variants, metrics, baseline="base", labels=None):
    """Paired tests (baseline vs each variant) per tracker spec and metric."""
    labels = labels or DEFAULT_LABELS
    out = []
    piv = df.set_index(KEYS)
    for tracker_label, tr_list in tracker_specs:
        for var in variants:
            for metric, hib in metrics:
                b, t = [], []
                for tr in tr_list:
                    seqs = sorted({s for (T, V, s) in piv.index if T == tr and V == baseline})
                    for s in seqs:
                        try:
                            bv = piv.loc[(tr, baseline, s), metric]
                            tv = piv.loc[(tr, var, s), metric]
                        except KeyError:
                            continue
                        b.append(bv)
                        t.append(tv)
                r = test_pair(b, t, hib)
                r.update(level=level, tracker=tracker_label, comparison=f"{baseline} vs {var} ({labels.get(var, var)})", metric=metric)
                out.append(r)
    return out


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN-aware)."""
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    idx = np.where(~np.isnan(p))[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[j]))
        out[j] = running
    return out


def significance(
    buckets: dict[str, pd.DataFrame],
    test_buckets: list[str],
    variants: list[str],
    baseline: str = "base",
    labels: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (combined 0-100 table, significance summary).

    `buckets` maps a bucket label (e.g. "00_20") to its per-sequence metrics. Every bucket enters the combined
    table; only `test_buckets` get per-bucket tests (the original analysis excludes 80-100: recall too low).
    """
    df_all = pd.concat([d.assign(bucket=b) for b, d in buckets.items()], ignore_index=True)
    comb = combine(df_all)

    trackers = sorted(comb["Tracker"].unique())
    specs = [(tr, [tr]) for tr in trackers] + [(POOLED, trackers)]

    rows = rows_for(comb, "Combined 0-100", specs, variants, COMBINED_METRICS, baseline, labels)
    for b in test_buckets:
        sub = df_all[df_all["bucket"] == b]
        rows += rows_for(sub, f"Bucket {b.replace('_', '-')}", specs, variants, BUCKET_METRICS, baseline, labels)

    summary = pd.DataFrame(rows)
    summary["p_holm"] = np.nan
    for _, grp in summary.groupby(["level", "metric", summary["tracker"].eq(POOLED)]):
        summary.loc[grp.index, "p_holm"] = holm(grp["p"].values)

    cols = ["level", "tracker", "comparison", "metric", "n", "n_nonzero", "mean", "sd", "win", "loss", "tie", "p", "p_holm", "p_min"]
    return comb, summary[cols]
