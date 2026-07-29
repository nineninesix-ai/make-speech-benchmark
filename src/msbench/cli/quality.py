#!/usr/bin/env python3
"""DNSMOS P.835 and the audio-QC battery, over any of the three audio streams.

Originally this scored `prompt_audio` only, to answer one question: is reference
quality a confounder for SIM? Measured on all 8,200 prompts, Pearson r =
0.049-0.124 and Spearman 0.031-0.123 across six languages — reference quality
explains under 1.6 % of SIM variance and is **not** a confounder. That result
stands and the code that produced it is unchanged.

v2 adds `--audio {prompt,gt,sim_ref}` for a different question (**D-09**).

**Audio QC ran on prompt candidates only.** `msbench.build.prompts` rejected clips
for clipping, narrowband upsampling, low speech ratio, recording breaks and
speaker-outlier distance — and it applied all of that exclusively to prompts.
`gt_audio` and `sim_ref_audio` received the same *preprocessing* (16 kHz, VAD
trim, peak-normalise) but never faced the *rejection* filters. The human WER
anchor is therefore measured on ground-truth recordings that would not have been
accepted as prompts, and nobody has ever looked at how many.

This measures them. It does **not** filter them: dropping rows would change the
row composition of the published subsets and break `utt` stability with v1
(§2.1). The numbers ship as columns so that a user who wants a clean subset can
apply their own threshold and say so.

Nothing here is a filter. It is a reporting metric, as it was in v1.

    msbench-quality --audio prompt --workers 12
    msbench-quality --audio gt --langs ky --limit 100
"""

import argparse
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# The ONNX session inside a worker must be single-threaded. OMP_NUM_THREADS does
# NOT control onnxruntime — it has its own pool (intra_op_num_threads) which
# defaults to every core on the machine. With 24 processes x 96 threads the
# contention was such that the parallel run went at exactly the speed of the
# sequential one; see _init_worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import stats  # noqa: E402

from msbench.paths import workspace  # noqa: E402

# `speechmos` and `onnxruntime` live in the `quality` extra. Importing them at
# module scope would make `--help` fail on an install that never intends to run
# DNSMOS, so the import happens on first use instead.
dnsmos = None


def _dnsmos():
    """The DNSMOS module. Raises if the `quality` extra is not installed.

    Deliberately not swallowed by the per-clip `except` in `score`: a clip that
    fails to score is a NaN, but a missing dependency scoring *every* clip as
    NaN is a broken run pretending to be a finished one.
    """
    global dnsmos
    if dnsmos is None:
        try:
            from speechmos import dnsmos as dm
        except ImportError as e:
            raise SystemExit(
                "DNSMOS needs the `quality` extra: "
                'uv pip install -e ".[quality]"') from e
        dnsmos = dm
    return dnsmos

SR = 16000
LANGS = ["en-US", "es-ES", "es-MX", "nl-NL", "pt-BR", "ky"]
AUDIO_COL = {"prompt": "prompt_audio", "gt": "gt_audio",
             "sim_ref": "sim_ref_audio"}

_VAD = None


def _init_worker():
    """Force onnxruntime — and torch — to a single thread inside this worker."""
    import onnxruntime as ort
    orig = ort.InferenceSession

    def single_threaded(path, *a, **kw):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        return orig(path, sess_options=so, providers=["CPUExecutionProvider"])

    ort.InferenceSession = single_threaded
    import speechmos.dnsmos as dm
    dm.ort.InferenceSession = single_threaded

    global dnsmos
    dnsmos = dm

    import torch
    torch.set_num_threads(1)


def _vad():
    global _VAD
    if _VAD is None:
        from silero_vad import get_speech_timestamps, load_silero_vad
        _VAD = (load_silero_vad(), get_speech_timestamps)
    return _VAD


def hf_energy_db(wav, sr=SR, fmin=5000.0):
    """Share of energy above `fmin`, in dB.

    Copied in behaviour from `msbench.corpus.qc:hf_energy_db`, the metric the
    prompt filter used. Note the name it is stored under in the v1 schema —
    `qc_bandwidth_hz` — is wrong: the value is decibels, not hertz (**D-07**).
    v2 writes it as `qc_hf_energy_db`.
    """
    n = min(len(wav), sr * 4)
    if n < 1024:
        return float("nan")
    spec = np.abs(np.fft.rfft(wav[:n] * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    total = spec.sum()
    if total <= 0:
        return float("nan")
    return float(10 * np.log10(spec[freqs >= fmin].sum() / total + 1e-12))


def score(item):
    """DNSMOS plus the QC battery for one clip. Runs in a worker process."""
    utt, raw = item
    try:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:  # noqa: BLE001
        return None
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        # Packaged audio is already 16 kHz; this is a defensive path only.
        import soxr
        wav = soxr.resample(wav, sr, SR, quality="HQ").astype("float32")
    if len(wav) < SR // 2:
        return None

    out = {"utt": utt, "dur": len(wav) / SR,
           "clip_rate": float((np.abs(wav) > 0.99).mean()),
           "hf_energy_db": hf_energy_db(wav),
           "peak_dbfs": float(20 * np.log10(max(np.abs(wav).max(), 1e-12))),
           "rms_dbfs": float(20 * np.log10(max(np.sqrt((wav ** 2).mean()), 1e-12)))}

    try:
        import torch
        model, get_ts = _vad()
        ts = get_ts(torch.from_numpy(wav), model, sampling_rate=SR)
        if ts:
            speech = sum(t["end"] - t["start"] for t in ts) / SR
            out.update(speech_ratio=speech / out["dur"], n_segments=len(ts),
                       lead_sil=ts[0]["start"] / SR,
                       trail_sil=(len(wav) - ts[-1]["end"]) / SR)
        else:
            out.update(speech_ratio=0.0, n_segments=0,
                       lead_sil=out["dur"], trail_sil=0.0)
    except Exception:  # noqa: BLE001
        out.update(speech_ratio=np.nan, n_segments=-1,
                   lead_sil=np.nan, trail_sil=np.nan)

    dm = _dnsmos()
    try:
        m = dm.run(wav.astype("float32"), sr=SR)
        out.update(dnsmos_ovrl=float(m["ovrl_mos"]), dnsmos_sig=float(m["sig_mos"]),
                   dnsmos_bak=float(m["bak_mos"]))
    except Exception:  # noqa: BLE001
        out.update(dnsmos_ovrl=np.nan, dnsmos_sig=np.nan, dnsmos_bak=np.nan)
    return out


def iter_items(f: Path, column: str, limit=None):
    """Stream (utt, bytes) and the metadata, one row group at a time.

    Reading a whole subset with `pd.read_parquet` costs several GB of RAM per
    language because of the audio columns; the worker pool then holds copies.
    """
    pf = pq.ParquetFile(f)
    have = set(pf.schema_arrow.names)
    meta_cols = [c for c in ("utt", "speaker_id", "gt_speaker_id",
                             "speaker_gender", "prompt_dur", "prompt_dur_bin",
                             "prompt_split_origin", "gt_dur", "sim_ref_dur")
                 if c in have]
    items, meta, n = [], [], 0
    for i in range(pf.num_row_groups):
        tbl = pf.read_row_group(i, columns=meta_cols + [column])
        for r in tbl.to_pylist():
            if limit and n >= limit:
                return items, pd.DataFrame(meta)
            n += 1
            cell = r.get(column)
            meta.append({k: r[k] for k in meta_cols})
            if cell is not None:
                items.append((r["utt"], cell["bytes"]))
    return items, pd.DataFrame(meta)


# `verdict`-equivalent thresholds from msbench.corpus.qc, applied here as
# REPORTING flags only. Nothing is rejected.
MIN_SPEECH_RATIO = 0.75
MAX_CLIP_RATE = 1e-3
MIN_HF_DB = -45.0


def qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["qc_fail_clipping"] = df.clip_rate > MAX_CLIP_RATE
    df["qc_fail_narrowband"] = df.hf_energy_db < MIN_HF_DB
    df["qc_fail_speech_ratio"] = df.speech_ratio < MIN_SPEECH_RATIO
    df["qc_fail_any"] = (df.qc_fail_clipping | df.qc_fail_narrowband
                         | df.qc_fail_speech_ratio)
    return df


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--pack", default=str(here / "tts-bench-v1"))
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--audio", choices=list(AUDIO_COL), default="prompt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    pack = Path(args.pack)
    rep = Path(args.out_dir) if args.out_dir else pack / "reports" / "v2"
    rep.mkdir(parents=True, exist_ok=True)
    column = AUDIO_COL[args.audio]
    summary, corr_rows = [], []

    for lang in args.langs:
        f = pack / "data" / lang / "main.parquet"
        if not f.exists():
            continue
        t0 = time.time()
        items, meta = iter_items(f, column, args.limit)
        if not items:
            print(f"{lang}: no {args.audio} audio", flush=True)
            continue

        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker) as pool:
            got = [x for x in pool.map(score, items, chunksize=8) if x]
        res = qc_flags(pd.DataFrame(got).merge(meta, on="utt", how="left"))
        res["lang"] = lang
        res["audio"] = args.audio
        res.to_parquet(rep / f"quality_{lang}_{args.audio}.parquet", index=False)

        row = {"lang": lang, "audio": args.audio, "n": len(res),
               "OVRL": round(res.dnsmos_ovrl.mean(), 3),
               "SIG": round(res.dnsmos_sig.mean(), 3),
               "BAK": round(res.dnsmos_bak.mean(), 3),
               "OVRL p05": round(res.dnsmos_ovrl.quantile(.05), 3),
               "OVRL p95": round(res.dnsmos_ovrl.quantile(.95), 3),
               "clipping %": round(100 * res.qc_fail_clipping.mean(), 2),
               "narrowband %": round(100 * res.qc_fail_narrowband.mean(), 2),
               "low speech %": round(100 * res.qc_fail_speech_ratio.mean(), 2),
               "would fail prompt QC %": round(100 * res.qc_fail_any.mean(), 2)}

        # The v1 confounder question, kept: only meaningful against the prompt,
        # which is the clip SIM is measured against.
        if args.audio == "prompt":
            sim_f = rep / f"sim_{lang}_anchor_wavlm_sv.parquet"
            if sim_f.exists():
                sim = pd.read_parquet(sim_f)[["utt", "sim"]]
                j = res.merge(sim, on="utt").dropna(subset=["sim", "dnsmos_ovrl"])
                if len(j) > 20:
                    row["r(OVRL,SIM)"] = round(stats.pearsonr(j.dnsmos_ovrl,
                                                              j.sim)[0], 3)
                    row["rho"] = round(stats.spearmanr(j.dnsmos_ovrl, j.sim)[0], 3)
                    row["n matched"] = len(j)
                    j["tier"] = pd.qcut(j.dnsmos_ovrl, 3,
                                        labels=["low", "mid", "high"])
                    for t, g in j.groupby("tier", observed=True):
                        corr_rows.append({"lang": lang, "quality tier": t,
                                          "DNSMOS OVRL": round(g.dnsmos_ovrl.mean(), 3),
                                          "SIM": round(g.sim.mean(), 4), "n": len(g)})
        summary.append(row)
        print(f"{lang}/{args.audio}: n={len(res)}  OVRL {row['OVRL']}  "
              f"would fail prompt QC {row['would fail prompt QC %']}%  "
              f"({time.time() - t0:.0f}s)", flush=True)

    s = pd.DataFrame(summary)
    print(f"\n## DNSMOS + QC over {args.audio} audio\n")
    print(s.to_string(index=False))
    md = [f"# Audio quality — `{args.audio}`\n",
          "A **reporting** metric, not a filter: nothing is rejected here, and "
          "no row is removed from any subset.\n",
          "`would fail prompt QC %` applies the thresholds from "
          "`msbench.corpus.qc` — the ones prompts had to pass — to this audio "
          "stream. For `gt` and `sim_ref` that check was never run during the "
          "build (D-09), so the column states how much of the human anchor rests "
          "on audio that would not have been accepted as a prompt.\n",
          s.to_markdown(index=False), ""]
    if corr_rows:
        c = pd.DataFrame(corr_rows).pivot(index="lang", columns="quality tier",
                                          values="SIM")
        md += ["\n## SIM anchor by reference-quality tier\n",
               "Terciles of DNSMOS OVRL within each language.\n", c.to_markdown(), ""]
    (rep / f"quality_{args.audio}.md").write_text("\n".join(md) + "\n")
    print(f"\n-> {rep / f'quality_{args.audio}.md'}")


if __name__ == "__main__":
    main()
