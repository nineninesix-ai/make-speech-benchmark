#!/usr/bin/env python3
"""Audio QC — decode sanity, VAD trimming, speaker consistency.

Also importable: `msbench.build.prompts` reuses `read_clip`, `analyse` and
`verdict` to score prompt candidates.

**Scope is deliberately narrow.** References in this benchmark are meant to be
passed through a speech-restoration model downstream, so noise metrics (DNSMOS,
WADA-SNR) and ASR verification of prompts are not used for *filtering*. What
remains is what restoration cannot fix:

  * decode sanity — true sample rate, channels, clipping, DC offset, and the
    energy above 5 kHz (a clip upsampled from 8 kHz would get a hallucinated,
    not its own, timbre after restoration, so SIM on it would measure the
    restoration model rather than cloning);
  * VAD — speech ratio, recording breaks, edge silence;
  * speaker consistency — `client_id` is not guaranteed to be one person.

The headline number this produces is **speaker survival**, which determines
whether `cap_lang` is high enough to reach N (see IMPLEMENTATION_PLAN, S3).

    python -m msbench.corpus.qc --langs ky nl-NL --speakers 40

Three thresholds here were wrong in the first draft and rejected 100% of the
sample. All three were metric bugs, not bad data; see the comments on
`hf_energy_db`, `merge_segments` and the trim-then-measure order in `analyse`.
"""

import argparse
import io
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from msbench.corpus.inventory import load, select_target
from msbench.languages import CODES, LANGUAGES

SR_TARGET = 16000

MIN_SPEECH_RATIO = 0.75
MAX_LEAD_SIL = 0.30
MAX_CLIP_RATE = 1e-3
MIN_HF_DB = -45.0    # below this it is almost certainly upsampled from 8 kHz


def read_clip(audio_root, row):
    """Pull one mp3 out of a tar by byte offset, without unpacking the archive."""
    with open(audio_root / row.tar, "rb") as fh:
        fh.seek(int(row.offset))
        data = fh.read(int(row.nbytes))
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return wav, sr


def hf_energy_db(wav_mono, sr, fmin=5000.0):
    """Share of energy above `fmin`, in dB — detects upsampling from 8 kHz.

    Two rejected alternatives, both of which fired on perfectly normal speech:
    an energy percentile (~99% of speech energy sits below 4 kHz, so it flagged
    good clips as narrowband), and "frequency at -40 dB from peak" (measures
    where speech rolls off, not where the codec wall is).

    Measured on 200 clips x 3 languages: the distribution is unimodal with a
    median near -24 dB; genuine upsampling is the tail below -45 dB, 1-2% of clips.
    """
    n = min(len(wav_mono), sr * 4)
    if n < 1024:
        return float("nan")
    spec = np.abs(np.fft.rfft(wav_mono[:n] * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    total = spec.sum()
    if total <= 0:
        return float("nan")
    return float(10 * np.log10(spec[freqs >= fmin].sum() / total + 1e-12))


def merge_segments(ts, sr, max_gap=0.25):
    """Join VAD segments separated by less than `max_gap`.

    A pause between words is not a recording break. Only a long pause — the
    "started, stumbled, re-recorded" pattern — counts as one.
    """
    if not ts:
        return []
    out = [dict(ts[0])]
    for seg in ts[1:]:
        if (seg["start"] - out[-1]["end"]) / sr <= max_gap:
            out[-1]["end"] = seg["end"]
        else:
            out.append(dict(seg))
    return out


def analyse(wav, sr, vad_model, vad_utils):
    mono = wav.mean(axis=1)
    out = {
        "sr": sr,
        "n_ch": wav.shape[1],
        "dur_raw": len(mono) / sr,
        "clip_rate": float((np.abs(mono) > 0.99).mean()),
        "dc": float(mono.mean()),
        "hf_db": hf_energy_db(mono, sr),
    }
    if sr != SR_TARGET:   # the VAD runs at 16 kHz
        idx = np.linspace(0, len(mono) - 1, int(len(mono) * SR_TARGET / sr))
        mono16 = np.interp(idx, np.arange(len(mono)), mono).astype("float32")
    else:
        mono16 = mono

    get_ts = vad_utils[0]
    ts = get_ts(torch.from_numpy(mono16), vad_model, sampling_rate=SR_TARGET)
    if not ts:
        out.update(has_speech=False, dur=0.0, speech_ratio=0.0, n_segments=0,
                   lead_sil=out["dur_raw"], trail_sil=0.0, max_gap=0.0)
        out["_mono16"] = mono16
        return out

    merged = merge_segments(ts, SR_TARGET)
    lead = ts[0]["start"] / SR_TARGET
    trail = len(mono16) / SR_TARGET - ts[-1]["end"] / SR_TARGET

    # TRIM FIRST, measure second. Edge silence is removed by the VAD and is not
    # a property of the clip; measuring before trimming rejected 148 of 150
    # clips in the first run (median leading silence 0.8s against a 0.3s
    # threshold).
    pad = int(0.05 * SR_TARGET)
    a = max(0, ts[0]["start"] - pad)
    b = min(len(mono16), ts[-1]["end"] + pad)
    trimmed = mono16[a:b]

    dur_t = len(trimmed) / SR_TARGET
    speech = sum(t["end"] - t["start"] for t in ts) / SR_TARGET
    gaps = [(s2["start"] - s1["end"]) / SR_TARGET
            for s1, s2 in zip(ts, ts[1:], strict=False)] or [0.0]
    out.update(
        has_speech=True,
        dur=dur_t,
        speech_ratio=speech / dur_t if dur_t else 0.0,
        n_segments=len(merged),
        lead_sil=lead,
        trail_sil=trail,
        max_gap=max(gaps),
    )
    out["_mono16"] = trimmed
    return out


def verdict(m, min_speech, min_hf_db):
    """Only criteria that speech restoration cannot fix downstream."""
    reasons = []
    if m["n_ch"] != 1:
        reasons.append("not mono")
    if not m.get("has_speech", False):
        reasons.append("no speech found")
        return reasons
    if m["clip_rate"] > MAX_CLIP_RATE:
        reasons.append("clipping")
    if m["hf_db"] < min_hf_db:
        reasons.append("narrowband (upsampled)")
    if m["speech_ratio"] < min_speech:
        reasons.append("little speech after trimming")
    if m["n_segments"] > 1:
        reasons.append("recording break")
    if not (2.0 <= m["dur"] <= 12.0):
        reasons.append("duration out of window")
    return reasons


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent / "cv"
    ap.add_argument("--data-dir", default=str(root))
    ap.add_argument("--langs", nargs="+", default=CODES)
    ap.add_argument("--speakers", type=int, default=40)
    ap.add_argument("--per-speaker", type=int, default=8)
    ap.add_argument("--min-speech", type=float, default=MIN_SPEECH_RATIO)
    ap.add_argument("--min-hf-db", type=float, default=MIN_HF_DB)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.data_dir)
    audio_root = root / "audio"
    # pt has 9,464 clips physically present in two shards (train and dev), so
    # the join would otherwise multiply rows.
    index = (pd.read_parquet(root / "index.parquet")
             .drop_duplicates("path", keep="first").set_index("path"))

    vad_model, vad_utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                          trust_repo=True)
    from transformers import AutoFeatureExtractor, WavLMForXVector
    sv_name = "microsoft/wavlm-base-plus-sv"
    fe = AutoFeatureExtractor.from_pretrained(sv_name)
    sv = WavLMForXVector.from_pretrained(sv_name).eval().cuda()

    all_rows = []
    for lang in args.langs:
        df = load(str(root), LANGUAGES[lang].locale)
        sub = select_target(df, lang)
        sub = sub[(sub.dur >= 3) & (sub.dur <= 8)]

        vc = sub.client_id.value_counts()
        cand = vc[vc >= 3].index[: args.speakers]
        sample = (sub[sub.client_id.isin(cand)]
                  .groupby("client_id", group_keys=False)
                  .head(args.per_speaker))
        sample = sample.join(index, on="path", rsuffix="_idx")
        sample = sample[sample.tar.notna()]

        print(f"\n{'=' * 70}\n{lang}: {len(sample)} clips, "
              f"{sample.client_id.nunique()} speakers", flush=True)

        t0 = time.time()
        embs, metrics = defaultdict(list), []
        for row in sample.itertuples():
            try:
                wav, sr = read_clip(audio_root, row)
                m = analyse(wav, sr, vad_model, vad_utils)
            except Exception as exc:  # noqa: BLE001
                metrics.append({"path": row.path, "client_id": row.client_id,
                                "ok": False, "reasons": [f"decode: {exc}"[:40]]})
                continue
            mono16 = m.pop("_mono16")
            reasons = verdict(m, args.min_speech, args.min_hf_db)
            m.update(path=row.path, client_id=row.client_id, lang=lang,
                     ok=not reasons, reasons=reasons)
            metrics.append(m)
            with torch.no_grad():
                inp = fe(mono16, sampling_rate=SR_TARGET, return_tensors="pt")
                e = sv(inp.input_values.cuda()).embeddings[0]
                embs[row.client_id].append(
                    torch.nn.functional.normalize(e, dim=-1).cpu())

        md = pd.DataFrame(metrics)
        md["lang"] = lang
        dist = {}
        for cid, vecs in embs.items():
            if len(vecs) < 2:
                continue
            M = torch.stack(vecs)
            c = torch.nn.functional.normalize(M.mean(0), dim=-1)
            dist[cid] = (M @ c).numpy()
        all_rows.append(md)

        ok = md.ok.sum()
        print(f"  clips passed: {ok}/{len(md)} ({100 * ok / len(md):.0f}%)"
              f"   in {time.time() - t0:.0f}s")
        rc = pd.Series([r for rs in md.reasons for r in rs]).value_counts()
        for reason, n in rc.items():
            print(f"    {reason}: {n}")

        # Thresholds must be read off these distributions, not guessed — a
        # guessed set rejected the entire sample once already.
        def q(col, fmt="{:.2f}"):
            if col not in md or md[col].isna().all():
                return "n/a"
            s = md[col].dropna()
            return " / ".join(fmt.format(s.quantile(p)) for p in (.05, .5, .95))

        print("  distributions (p05/median/p95):")
        print(f"    speech_ratio: {q('speech_ratio')}   "
              f"trimmed off the edges: {q('lead_sil')} + {q('trail_sil')} s")
        print(f"    hf_db (energy >5kHz): {q('hf_db', '{:.0f}')} dB")
        print(f"    duration after trimming: {q('dur')} s (before {q('dur_raw')})")
        print(f"    internal max_gap: {q('max_gap')} s   "
              f"SR: {sorted(md.sr.dropna().unique().tolist())}")

        surv = md.groupby("client_id").ok.any()
        print(f"  SPEAKERS surviving: {surv.sum()}/{len(surv)} "
              f"({100 * surv.mean():.0f}%)")
        if dist:
            cos = np.concatenate(list(dist.values()))
            spread = {cid: float(d.min()) for cid, d in dist.items()}
            worst = sorted(spread.items(), key=lambda kv: kv[1])[:3]
            print(f"  speaker-centroid cosine: median {np.median(cos):.3f}, "
                  f"p05 {np.percentile(cos, 5):.3f}")
            print("  worst speakers: "
                  + ", ".join(f"{c[:8]}...={v:.2f}" for c, v in worst))

    out = Path(args.out) if args.out else root.parent / "reports" / "pilot_qc.parquet"
    out.parent.mkdir(exist_ok=True)
    pd.concat(all_rows, ignore_index=True).drop(columns=["reasons"]).to_parquet(out)
    print(f"\nmetrics -> {out}")


if __name__ == "__main__":
    main()
