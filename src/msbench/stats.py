#!/usr/bin/env python3
"""Confidence intervals and paired comparison, resampled by speaker.

Every number this benchmark publishes today is a point estimate (**D-20**). That
would be a minor omission if the observations were independent. They are not:

  * one prompt clip serves up to 7 examples (`speaker_n_in_subset`, cap 7 in
    pt-BR), so the rows sharing a prompt are strongly correlated;
  * the SIM anchor is *constant* within a speaker — `msbench.build.assemble` gives
    every example of a speaker the same prompt and the same second clip, so
    en-US's `n = 909` is 909 rows over a few hundred distinct measurements;
  * GT-speaker concentration is uncapped, and per-cell `n` in the breakdown
    tables drops to about 100.

Resampling rows would therefore understate the interval — it treats seven
readings of one voice as seven independent facts. Everything here resamples
**speakers** with replacement instead.

**Implementation note.** The obvious implementation — rebuild a DataFrame per
replicate with `pd.concat` — is unusable: 2,000 replicates over ~1,450 speakers
is three million small concatenations, and the en-US anchor did not finish in
ten minutes. Every statistic reported by this benchmark is a ratio of two sums
(corpus WER is Σerrors/Σwords; a mean is Σx/Σ1; a rate is Σindicator/Σ1), so a
statistic is expressed as that pair, per-cluster sums are computed once, and the
whole bootstrap becomes one vectorised gather. The result is identical and the
runtime is milliseconds. `cluster_bootstrap` still accepts a plain callable and
falls back to the slow path for anything that is not a ratio of sums.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 20260725
N_BOOT = 2000


@dataclass
class RatioStat:
    """A statistic expressed as (sum of numerator) / (sum of denominator).

    Writing statistics this way is what makes the cluster bootstrap tractable,
    and it also makes each definition explicit: `wer_corpus` is visibly
    Σ(S+D+I)/Σ N_ref rather than a mean of something.
    """

    name: str
    num: Callable[[pd.DataFrame], np.ndarray]
    den: Callable[[pd.DataFrame], np.ndarray]

    def __call__(self, df: pd.DataFrame) -> float:
        d = float(np.asarray(self.den(df), dtype=float).sum())
        if d == 0:
            return float("nan")
        return float(np.asarray(self.num(df), dtype=float).sum() / d)


def _ones(df):
    return np.ones(len(df), dtype=float)


wer_corpus = RatioStat("wer_corpus",
                       lambda d: d["subs"] + d["dels"] + d["ins"],
                       lambda d: d["n_ref_words"])
cer_corpus = RatioStat("cer_corpus",
                       lambda d: d["cer_err"],
                       lambda d: d["n_ref_chars"])


def macro(col: str) -> RatioStat:
    """Mean of a per-utterance column — v1's aggregation, kept for continuity."""
    return RatioStat(f"macro({col})", lambda d: d[col].astype(float), _ones)


def rate_above(col: str, thr: float) -> RatioStat:
    return RatioStat(f"rate({col}>{thr})",
                     lambda d: (d[col] > thr).astype(float), _ones)


def _cluster_sums(df: pd.DataFrame, stat: RatioStat, cluster_col: str):
    """Per-cluster (numerator, denominator) sums, in a fixed cluster order."""
    codes, _ = pd.factorize(df[cluster_col], use_na_sentinel=False)
    k = codes.max() + 1 if len(codes) else 0
    num = np.bincount(codes, weights=np.asarray(stat.num(df), dtype=float),
                      minlength=k)
    den = np.bincount(codes, weights=np.asarray(stat.den(df), dtype=float),
                      minlength=k)
    return num, den


def cluster_bootstrap(df: pd.DataFrame, statistic, cluster_col: str = "speaker_id",
                      n_boot: int = N_BOOT, seed: int = SEED,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (point estimate, lo, hi) at 1−alpha, resampling clusters."""
    if cluster_col not in df.columns:
        raise KeyError(f"{cluster_col!r} missing — cannot cluster; a row "
                       "bootstrap would understate the interval (D-20)")
    point = float(statistic(df))
    rng = np.random.default_rng(seed)

    if isinstance(statistic, RatioStat):
        num, den = _cluster_sums(df, statistic, cluster_col)
        k = len(num)
        if k < 2:
            return point, float("nan"), float("nan")
        idx = rng.integers(0, k, size=(n_boot, k))
        d = den[idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = np.where(d > 0, num[idx].sum(axis=1) / d, np.nan)
    else:
        groups = [g for _, g in df.groupby(cluster_col, sort=True, observed=True)]
        k = len(groups)
        if k < 2:
            return point, float("nan"), float("nan")
        reps = np.array([
            statistic(pd.concat([groups[i] for i in rng.integers(0, k, size=k)],
                                ignore_index=True))
            for _ in range(n_boot)], dtype=float)

    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame, statistic,
                     key: str = "utt", cluster_col: str = "speaker_id",
                     n_boot: int = N_BOOT, seed: int = SEED,
                     alpha: float = 0.05) -> dict:
    """Compare two systems on the utterances they share.

    Comparing two independent confidence intervals throws away the fact that
    both systems were measured on the same sentences: utterance difficulty is
    common to both and cancels in the difference. Each replicate resamples the
    shared speakers once and applies that same resample to both systems.

    Returns the delta (a − b), its CI, and P(delta < 0) — the probability that A
    is better when the statistic is an error rate.
    """
    shared = set(df_a[key]) & set(df_b[key])
    a = df_a[df_a[key].isin(shared)]
    b = df_b[df_b[key].isin(shared)]
    if cluster_col not in a.columns:
        raise KeyError(f"{cluster_col!r} missing from the left frame")

    spk = a[[key, cluster_col]].drop_duplicates(key).set_index(key)[cluster_col]
    b = b.assign(**{cluster_col: b[key].map(spk)})
    point = float(statistic(a)) - float(statistic(b))

    if not isinstance(statistic, RatioStat):
        raise TypeError("paired_bootstrap needs a RatioStat")

    # Both sides must be indexed by the SAME cluster ordering, or a replicate
    # would pair speaker i of A with speaker j of B and the pairing is lost.
    cats = pd.CategoricalDtype(sorted(set(spk.values)))
    a = a.assign(**{cluster_col: a[cluster_col].astype(cats)})
    b = b.assign(**{cluster_col: b[cluster_col].astype(cats)})

    def sums(df):
        codes = df[cluster_col].cat.codes.to_numpy()
        keep = codes >= 0
        k = len(cats.categories)
        return (np.bincount(codes[keep],
                            weights=np.asarray(statistic.num(df),
                                               dtype=float)[keep], minlength=k),
                np.bincount(codes[keep],
                            weights=np.asarray(statistic.den(df),
                                               dtype=float)[keep], minlength=k))

    na, da = sums(a)
    nb, db = sums(b)
    k = len(cats.categories)
    if k < 2:
        return {"delta": point, "lo": float("nan"), "hi": float("nan"),
                "p_a_better": float("nan"), "n_shared": len(shared),
                "n_clusters": k}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    with np.errstate(invalid="ignore", divide="ignore"):
        ra = na[idx].sum(1) / np.where(da[idx].sum(1) > 0, da[idx].sum(1), np.nan)
        rb = nb[idx].sum(1) / np.where(db[idx].sum(1) > 0, db[idx].sum(1), np.nan)
    deltas = (ra - rb)
    deltas = deltas[np.isfinite(deltas)]
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {"delta": point, "lo": float(lo), "hi": float(hi),
            "p_a_better": float(np.mean(deltas < 0)),
            "n_shared": len(shared), "n_clusters": k}


def fmt_ci(point: float, lo: float, hi: float, digits: int = 4) -> str:
    if not np.isfinite(lo):
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


if __name__ == "__main__":
    import time

    # 1. Why the cluster matters. 100 speakers, 7 rows each; the error rate is a
    # property of the SPEAKER, so rows within a speaker carry no new information.
    # A row bootstrap sees 700 independent observations and reports an interval
    # roughly sqrt(7) too narrow.
    rng = np.random.default_rng(0)
    spk = np.repeat(np.arange(100), 7)
    df = pd.DataFrame({"speaker_id": spk, "wer": rng.beta(2, 20, size=100)[spk]})
    p, lo, hi = cluster_bootstrap(df, macro("wer"))
    print(f"cluster bootstrap by speaker : {fmt_ci(p, lo, hi)}  width {hi - lo:.4f}")
    p2, lo2, hi2 = cluster_bootstrap(
        df.assign(speaker_id=np.arange(len(df))), macro("wer"))
    print(f"row bootstrap (wrong)        : {fmt_ci(p2, lo2, hi2)}  "
          f"width {hi2 - lo2:.4f}")
    print(f"understated by a factor of {(hi - lo) / (hi2 - lo2):.1f}\n")

    # 2. The fast path agrees with the generic one, and is not slow.
    big = pd.DataFrame({
        "speaker_id": np.repeat(np.arange(1400), 1),
        "subs": rng.integers(0, 3, 1400), "dels": rng.integers(0, 2, 1400),
        "ins": rng.integers(0, 2, 1400), "n_ref_words": rng.integers(4, 17, 1400)})
    t = time.time()
    fast = cluster_bootstrap(big, wer_corpus, n_boot=2000)
    t_fast = time.time() - t
    slow_stat = lambda d: wer_corpus(d)   # noqa: E731  -> forces the generic path
    t = time.time()
    slow = cluster_bootstrap(big, slow_stat, n_boot=200)
    t_slow = time.time() - t
    print(f"vectorised  {fmt_ci(*fast)}   2000 replicates in {t_fast:.2f}s")
    print(f"generic     {fmt_ci(*slow)}    200 replicates in {t_slow:.2f}s")
    print(f"speed-up per replicate: {(t_slow / 200) / (t_fast / 2000):.0f}x")
