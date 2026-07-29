#!/usr/bin/env python3
"""Repackage the six subsets under schema v2.0.

Everything here is applied to the **existing** audio. Stages S1-S3 (pool
filtering, prompt QC, constrained selection) are not re-run, so `utt`, `text`,
`prompt_audio`, `gt_audio`, `sim_ref_audio` and the row composition of every
subset are byte-identical to v1. That stability is deliberate: `utt` is the join
key for anything already synthesised against v1, and a re-selection would
invalidate all of it for no methodological gain (V2_IMPLEMENTATION_PLAN §2.1).

What changes:

**Renamed.** `qc_bandwidth_hz` holds decibels, not hertz — it is the share of
energy above 5 kHz in dB, computed by `msbench.corpus.qc:hf_energy_db`, and
its observed values sit near −27. It becomes `qc_hf_energy_db` (**D-07**). The
old name is kept as a deprecated copy for one release.

**Retyped.** `tags` was `list<null>`, a degenerate type that cannot hold a value
at all — writing a string into it fails (**D-23**). Now `list<string>`.

**Corrected.** Kyrgyz `phones` contained the espeak dental markers `t[` and `d[`
(1,584 and 1,132 occurrences) while the card claims IPA. They become `t̪` and
`d̪` (**D-22**). This changes the phoneme inventory string; `n_phones` is
unaffected because the count was over segments, not characters.

**Added — quality of the other two audio streams.** Prompt QC rejected clips for
clipping, narrowband upsampling and low speech ratio, and it ran on prompt
candidates only. `gt_audio` and `sim_ref_audio` got the same *preprocessing* but
never faced the *rejection* filters, so the human WER anchor includes recordings
that would not have been accepted as prompts (**D-09**). The `gt_qc_*` and
`simref_qc_*` columns state this per row. They are **reporting** columns: nothing
is filtered, because dropping rows would break `utt` stability. A user who wants
a clean subset applies their own threshold and says so.

**Added.** `sim_ref_dur_bin`, so SIM breakdowns can control for the duration of
the *second* clip. v1 bins by prompt duration only, leaving the other clip in the
pair — down to ~2 s — uncontrolled (**D-21**).

**Added — the human anchors, per row.** v1 published its anchors only as
aggregate markdown, and the first v2 draft shipped them only as separate parquet
files under `reports/`. Both leave the benchmark's central numbers invisible from
the dataset itself: a reader browsing the rows sees the audio and the text but no
WER and no SIM anywhere. The `anchor_*` columns put them where the data is —
one recogniser's error rate, a second recogniser's for cross-checking, the
recogniser's actual transcript so a row's WER can be understood rather than
merely read, and the SIM anchor from all three encoders. They are `NaN` where the
underlying audio is absent (`has_gt` / `has_sim_ref` false).

Note that the SIM anchor is **constant within a speaker** by construction: every
example of a speaker shares one prompt clip and one second clip. The per-row
value is a convenience, not an independent observation, which is why every
interval in the reports clusters by speaker.

**Removed.** `qc_snr_db` and `qc_asr_cer` were 100 % null in every subset: they
were declared in the schema and never computed. An always-null column in a
dataset that advertises itself as documented is worse than an absent one.

    python -m msbench.build.repack_v2 --out tts-bench-v2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from msbench.languages import CODES
from msbench.paths import workspace

# espeak-ng emits a bracket for the dental diacritic; IPA uses U+032A.
ESPEAK_DENTAL = {"t[": "t̪", "d[": "d̪"}

DROP = ["qc_snr_db", "qc_asr_cer"]

# quality_{lang}_{stream}.parquet column -> destination prefix suffix
QC_MAP = {
    "dnsmos_ovrl": "dnsmos_ovrl", "dnsmos_sig": "dnsmos_sig",
    "dnsmos_bak": "dnsmos_bak", "clip_rate": "clip_rate",
    "hf_energy_db": "hf_energy_db", "speech_ratio": "vad_speech_ratio",
    "lead_sil": "lead_sil", "trail_sil": "trail_sil",
    "qc_fail_any": "fails_prompt_qc",
}


def sim_ref_dur_bin(x) -> str:
    """Same boundaries as msbench.cli.sim, so tables and data agree."""
    if x is None or not np.isfinite(x) or x <= 0:
        return "n/a"
    if x < 3.5:
        return "<3.5"
    return "3.5-4.5" if x < 4.5 else ">=4.5"


# Which recogniser provides the primary anchor for each subset. Whisper is
# unusable for Kyrgyz, which is why GigaAM is there; MMS is the second opinion
# everywhere because it decodes without a language model.
PRIMARY_ASR = {"en-US": "whisper", "es-ES": "whisper", "es-MX": "whisper",
               "nl-NL": "whisper", "pt-BR": "whisper", "ky": "gigaam"}
ENCODERS = ["wavlm_sv", "wavlm_ft", "ecapa"]

# Anchor columns that are counts rather than rates.
COUNT_COLS = {"anchor_subs", "anchor_dels", "anchor_ins",
              "anchor_n_ref_words", "anchor_cer_err", "anchor_n_ref_chars"}
COUNT_COLS |= {f"{c}_{s}" for c in ("anchor_subs", "anchor_dels", "anchor_ins",
                                    "anchor_cer_err") for s in ("mms", "scribe")}


def load_anchors(rep: Path, lang: str) -> pd.DataFrame:
    """utt -> the human anchor values, one column per measurement."""
    out = pd.DataFrame()

    def take(f: Path, cols: dict):
        nonlocal out
        if not f.exists():
            return
        d = pd.read_parquet(f).set_index("utt")
        d = d[[c for c in cols if c in d.columns]].rename(columns=cols)
        out = d if out.empty else out.join(d, how="outer")

    primary = PRIMARY_ASR[lang]
    # The raw edit counts, not just the per-utterance rates. Without them the
    # headline aggregation — Σ(S+D+I) / Σ N_ref — is not computable from the
    # dataset at all: averaging per-row rates gives the macro form, which v2
    # keeps only for continuity with v1. Six extra columns buy the ability to
    # derive the actual headline, and any slice of it, from the data alone.
    take(rep / f"wer_{lang}_gt_{primary}.parquet",
         {"wer": "anchor_wer", "cer": "anchor_cer", "hyp_norm": "anchor_hyp",
          "subs": "anchor_subs", "dels": "anchor_dels", "ins": "anchor_ins",
          "n_ref_words": "anchor_n_ref_words", "cer_err": "anchor_cer_err",
          "n_ref_chars": "anchor_n_ref_chars"})
    # The other two recognisers, with their counts as well, so a corpus-level
    # figure is derivable for each rather than only for the primary. The
    # reference length is not repeated: `n_ref_words` and `n_ref_chars` are
    # properties of the reference text and are identical across all three
    # (asserted in msbench.build.validate).
    for asr, suffix in (("mms", "mms"), ("elevenlabs", "scribe")):
        if asr == primary:
            continue
        cols = {"wer": f"anchor_wer_{suffix}", "cer": f"anchor_cer_{suffix}",
                "subs": f"anchor_subs_{suffix}", "dels": f"anchor_dels_{suffix}",
                "ins": f"anchor_ins_{suffix}",
                "cer_err": f"anchor_cer_err_{suffix}"}
        if asr == "elevenlabs":
            # Scribe's transcript is worth carrying: it is the most accurate
            # read of these recordings on five of the six subsets, so where it
            # disagrees with the reference the reference is worth a look.
            cols["hyp_norm"] = "anchor_hyp_scribe"
        take(rep / f"wer_{lang}_gt_{asr}.parquet", cols)
    for enc in ENCODERS:
        take(rep / f"sim_{lang}_anchor_{enc}.parquet", {"sim": f"anchor_sim_{enc}"})
    return out


def load_stream_qc(rep: Path, lang: str, stream: str, prefix: str):
    """utt -> {column: value} for one audio stream, or {} if not measured."""
    f = rep / f"quality_{lang}_{stream}.parquet"
    if not f.exists():
        return {}
    df = pd.read_parquet(f)
    keep = ["utt"] + [c for c in QC_MAP if c in df.columns]
    df = df[keep].set_index("utt")
    df.columns = [f"{prefix}_{QC_MAP[c]}" for c in df.columns]
    return df


def fix_phones(s: str) -> str:
    for a, b in ESPEAK_DENTAL.items():
        s = s.replace(a, b)
    return s


def build_columns(utts: list[str], lang: str, sim_ref_dur, phones, qc_gt, qc_sr,
                  anchors):
    """The per-row-group block of new columns, aligned to `utts`."""
    idx = pd.Index(utts)
    cols = {}

    cols["sim_ref_dur_bin"] = pa.array([sim_ref_dur_bin(x) for x in sim_ref_dur],
                                       type=pa.string())
    if lang == "ky":
        cols["phones"] = pa.array([fix_phones(p) for p in phones], type=pa.string())

    cols["anchor_asr"] = pa.array([PRIMARY_ASR[lang]] * len(utts), type=pa.string())
    if isinstance(anchors, pd.DataFrame) and len(anchors):
        a = anchors.reindex(idx)
        for c in a.columns:
            if c.startswith("anchor_hyp"):
                cols[c] = pa.array([None if pd.isna(x) else str(x) for x in a[c]],
                                   type=pa.string())
            elif c in COUNT_COLS:
                # Counts stay integers, nullable: a float32 edit count would be
                # an invitation to average it, and averaging counts is exactly
                # the mistake the corpus aggregation exists to avoid.
                cols[c] = pa.array([None if pd.isna(x) else int(x) for x in a[c]],
                                   type=pa.int32())
            else:
                cols[c] = pa.array(a[c].to_numpy(dtype="float64"), type=pa.float32())

    for block in (qc_gt, qc_sr):
        if isinstance(block, pd.DataFrame) and len(block):
            aligned = block.reindex(idx)
            for c in aligned.columns:
                v = aligned[c]
                if v.dtype == bool or c.endswith("fails_prompt_qc"):
                    # reindex introduces NaN for rows with no audio at all;
                    # "would fail QC" is undefined there, not False.
                    cols[c] = pa.array([None if pd.isna(x) else bool(x) for x in v],
                                       type=pa.bool_())
                else:
                    cols[c] = pa.array(v.to_numpy(dtype="float64"),
                                       type=pa.float32())
    return cols


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--pack", default=str(here / "tts-bench-v1"))
    ap.add_argument("--out", default=str(here / "tts-bench-v2"))
    ap.add_argument("--reports", default=None,
                    help="where the v2 quality parquets live")
    ap.add_argument("--langs", nargs="+", default=CODES)
    args = ap.parse_args()

    pack = Path(args.pack)
    out = Path(args.out)
    rep = Path(args.reports) if args.reports else pack / "reports" / "v2"

    for lang in args.langs:
        src = pack / "data" / lang / "main.parquet"
        if not src.exists():
            print(f"{lang}: no source parquet, skipping")
            continue
        qc_gt = load_stream_qc(rep, lang, "gt", "gt_qc")
        qc_sr = load_stream_qc(rep, lang, "sim_ref", "simref_qc")
        anchors = load_anchors(rep, lang)

        pf = pq.ParquetFile(src)
        dst = out / "data" / lang / "main.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)

        writer, n = None, 0
        for i in range(pf.num_row_groups):
            tbl = pf.read_row_group(i)
            utts = tbl.column("utt").to_pylist()
            new = build_columns(
                utts, lang,
                tbl.column("sim_ref_dur").to_pylist(),
                tbl.column("phones").to_pylist() if lang == "ky" else None,
                qc_gt, qc_sr, anchors)

            # `qc_bandwidth_hz` is decibels; ship it under the right name and
            # keep the old one for one release so nothing breaks silently. The
            # dtype is carried over rather than narrowed: the two columns must
            # be provably the same numbers, and a float32 cast would make them
            # differ in the last places for no benefit.
            new["qc_hf_energy_db"] = tbl.column("qc_bandwidth_hz")

            # A typed empty list, so a downstream writer can actually put a
            # string in it (D-23).
            new["tags"] = pa.array([[] for _ in utts],
                                   type=pa.list_(pa.string()))

            for name, arr in new.items():
                if name in tbl.column_names:
                    tbl = tbl.set_column(tbl.column_names.index(name),
                                         name, arr)
                else:
                    tbl = tbl.append_column(name, arr)
            for c in DROP:
                if c in tbl.column_names:
                    tbl = tbl.drop_columns([c])

            if writer is None:
                writer = pq.ParquetWriter(dst, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            n += tbl.num_rows
        writer.close()

        got = pq.ParquetFile(dst)
        print(f"{lang:<6} {n:5d} rows  {len(got.schema_arrow.names):3d} cols  "
              f"{dst.stat().st_size / 1e6:7.1f} MB  -> {dst}")


if __name__ == "__main__":
    main()
