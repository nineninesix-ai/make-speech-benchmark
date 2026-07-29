#!/usr/bin/env python3
"""S4 — assemble the package: per-language parquet plus a flat seed-tts-eval layout.

Three audio streams per example:
  prompt_audio    the reference, 16 kHz, VAD-trimmed (+ a copy at the original SR)
  gt_audio        a real recording of the target text; the speaker is arbitrary
  sim_ref_audio   a second clip of the *prompt* speaker, any text -> SIM topline

    python -m msbench.build.assemble --langs ky
"""

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from msbench.corpus.qc import SR_TARGET, analyse, read_clip
from msbench.paths import workspace
from msbench.schema import meta_lst_line

PEAK_DBFS = -1.0


class Loc:
    """An index row as an attribute holder.

    Not a pandas Series: `index.loc[p].to_frame().T.iloc[0]` coerces the columns
    to object dtype and the byte offsets stop being usable.
    """

    __slots__ = ("tar", "offset", "nbytes", "path")

    def __init__(self, tar, offset, nbytes, path):
        self.tar = tar
        self.offset = int(offset)
        self.nbytes = int(nbytes)
        self.path = path


def to_wav_bytes(x, sr):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def peak_norm(x, dbfs=PEAK_DBFS):
    peak = np.abs(x).max()
    if peak <= 0:
        return x
    return (x / peak * (10 ** (dbfs / 20))).astype("float32")


def load_trimmed(audio_root, row, vad_model, vad_utils):
    """Return (trimmed 16 kHz, trimmed original SR, sr, duration, metrics)."""
    wav, sr = read_clip(audio_root, row)
    m = analyse(wav, sr, vad_model, vad_utils)
    trimmed16 = m.pop("_mono16")
    if not m.get("has_speech"):
        return None
    mono = wav.mean(axis=1)
    # Same trim boundaries expressed as fractions, so they transfer to the
    # original sample rate without depending on rounding.
    frac_a = 0.0 if m["lead_sil"] <= 0.05 else (m["lead_sil"] - 0.05) / m["dur_raw"]
    frac_b = 1.0 - max(0.0, (m["trail_sil"] - 0.05)) / m["dur_raw"]
    a = max(0, int(frac_a * len(mono)))
    b = min(len(mono), int(frac_b * len(mono)))
    orig = mono[a:b] if b > a else mono
    return peak_norm(trimmed16), peak_norm(orig), sr, len(trimmed16) / SR_TARGET, m


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--root", default=str(here / "cv"))
    ap.add_argument("--work", default=str(here / "work"))
    ap.add_argument("--out", default=str(here / "tts-bench-v1"))
    ap.add_argument("--langs", nargs="+", default=None)
    ap.add_argument("--no-gt", action="store_true", help="skip ground-truth audio")
    args = ap.parse_args()

    root, work, out = Path(args.root), Path(args.work), Path(args.out)
    audio_root = root / "audio"
    index = (pd.read_parquet(root / "index.parquet")
             .drop_duplicates("path", keep="first").set_index("path"))
    idx_map = {p: (t, o, n) for p, t, o, n in
               zip(index.index, index.tar, index.offset, index.nbytes, strict=True)}

    def loc(path):
        t, o, n = idx_map[path]
        return Loc(t, o, n, path)

    vad_model, vad_utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                          trust_repo=True)

    langs = args.langs or [p.stem.replace("_selected", "")
                           for p in sorted(work.glob("*_selected.parquet"))]

    for lang in langs:
        sel = pd.read_parquet(work / f"{lang}_selected.parquet")
        qc_ok = pd.read_parquet(work / f"{lang}_qc.parquet").query("ok")
        by_spk = {c: g.path.tolist() for c, g in qc_ok.groupby("client_id")}

        t0, rows, misses = time.time(), [], {"gt": 0, "sim": 0, "prompt": 0}
        for r in sel.itertuples():
            rec = {"utt": r.utt, "lang": lang, "subset": "main",
                   "source": "cv17", "cv_version": "17.0"}

            got = load_trimmed(audio_root, loc(r.prompt_path), vad_model, vad_utils)
            if got is None:
                misses["prompt"] += 1
                continue
            p16, porig, psr, pdur, pm = got
            rec.update(
                prompt_audio={"bytes": to_wav_bytes(p16, SR_TARGET),
                              "path": f"{r.utt}.wav"},
                prompt_audio_orig={"bytes": to_wav_bytes(porig, psr),
                                   "path": f"{r.utt}.wav"},
                prompt_text=r.prompt_text, prompt_dur=float(pdur),
                prompt_sr_orig=int(psr), prompt_dur_bin=r.prompt_dur_bin,
                prompt_cv_path=r.prompt_path, prompt_split_origin=r.prompt_split,
                speaker_id=str(r.prompt_speaker)[:12],
                speaker_gender={"male_masculine": "male",
                                "female_feminine": "female"}.get(
                                    r.prompt_gender, "unknown"),
                speaker_gender_source="cv" if pd.notna(r.prompt_gender) else "unknown",
                speaker_age=r.prompt_age if pd.notna(r.prompt_age) else "unknown",
                speaker_accent_label="",
                speaker_n_in_subset=int(r.speaker_n_in_subset),
                qc_vad_speech_ratio=float(pm["speech_ratio"]),
                qc_lead_sil=float(pm["lead_sil"]), qc_trail_sil=float(pm["trail_sil"]),
                qc_clip_rate=float(pm["clip_rate"]), qc_bandwidth_hz=float(pm["hf_db"]),
                qc_spk_centroid_dist=float("nan"),
                qc_dnsmos_ovrl=float("nan"), qc_dnsmos_sig=float("nan"),
                qc_dnsmos_bak=float("nan"), qc_snr_db=float("nan"),
                qc_asr_cer=float("nan"),
            )

            rec.update(text=r.text, text_norm=r.text.lower(),
                       n_words=int(r.n_words), n_chars=len(r.text),
                       len_bin=r.len_bin, phones=r.phones,
                       n_phones=int(r.n_phones),
                       punct_type=r.punct_type, text_cv_path=r.text_path,
                       text_split_origin=r.text_split)

            gt = None if args.no_gt else load_trimmed(
                audio_root, loc(r.text_path), vad_model, vad_utils)
            if gt is None:
                misses["gt"] += 1
                rec.update(gt_audio=None, has_gt=False, gt_dur=0.0,
                           gt_speaker_id="", gt_same_speaker=False,
                           gt_gender="unknown")
            else:
                g16, _, _, gdur, _ = gt
                rec.update(gt_audio={"bytes": to_wav_bytes(g16, SR_TARGET),
                                     "path": f"{r.utt}.wav"},
                           has_gt=True, gt_dur=float(gdur),
                           gt_speaker_id=str(r.text_speaker)[:12],
                           gt_same_speaker=bool(r.gt_same_speaker),
                           gt_gender="unknown")

            others = [p for p in by_spk.get(r.prompt_speaker, [])
                      if p != r.prompt_path]
            s = load_trimmed(audio_root, loc(others[0]), vad_model,
                             vad_utils) if others else None
            if s is None:
                misses["sim"] += 1
                rec.update(sim_ref_audio=None, has_sim_ref=False,
                           sim_ref_text="", sim_ref_dur=0.0)
            else:
                s16, _, _, sdur, _ = s
                rec.update(sim_ref_audio={"bytes": to_wav_bytes(s16, SR_TARGET),
                                          "path": f"{r.utt}.wav"},
                           has_sim_ref=True, sim_ref_dur=float(sdur),
                           sim_ref_text=qc_ok.set_index("path").loc[
                               others[0], "sentence"])

            rec.update(category="general", subcategory="", tags=[], difficulty=0,
                       has_digit=False, has_abbrev=False, has_foreign=False,
                       notes="")
            rows.append(rec)

        df = pd.DataFrame(rows)
        d = out / "data" / lang
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / "main.parquet", index=False)

        # Flat layout for the stock seed-tts-eval scripts.
        flat = out / "seed-tts-eval" / lang
        for sub in ("prompt-wavs", "gt-wavs", "sim-refs"):
            (flat / sub).mkdir(parents=True, exist_ok=True)
        lines = []
        for rec in rows:
            (flat / "prompt-wavs" / f"{rec['utt']}.wav").write_bytes(
                rec["prompt_audio"]["bytes"])
            gtp = ""
            if rec["has_gt"]:
                (flat / "gt-wavs" / f"{rec['utt']}.wav").write_bytes(
                    rec["gt_audio"]["bytes"])
                gtp = f"gt-wavs/{rec['utt']}.wav"
            if rec["has_sim_ref"]:
                (flat / "sim-refs" / f"{rec['utt']}.wav").write_bytes(
                    rec["sim_ref_audio"]["bytes"])
            lines.append(meta_lst_line(rec, f"prompt-wavs/{rec['utt']}.wav", gtp))
        (flat / "meta.lst").write_text("\n".join(lines) + "\n")

        size = sum(f.stat().st_size for f in d.glob("*.parquet")) / 1e6
        print(f"{lang}: {len(df)} examples, {size:.0f} MB, "
              f"GT {int(df.has_gt.sum())}, sim_ref {int(df.has_sim_ref.sum())}, "
              f"misses {misses}, in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
