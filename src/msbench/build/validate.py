#!/usr/bin/env python3
"""Verify the v2 pack before it is published.

The promise v2 makes to its users is that **nothing about the data changed** —
only the description of it. That promise is worth exactly as much as the check
behind it, so this asserts it rather than assuming it:

  * the same rows, in the same order, with the same `utt`;
  * byte-identical audio in all three streams;
  * unchanged `text`, `prompt_text`, and every column v2 does not claim to touch;
  * and, positively, that the columns v2 *does* claim to change actually changed
    in the stated direction.

Run before any upload.

    msbench-validate
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from msbench.languages import CODES
from msbench.paths import pack

AUDIO = ["prompt_audio", "prompt_audio_orig", "gt_audio", "sim_ref_audio"]
DROPPED = ["qc_snr_db", "qc_asr_cer"]
RENAMED = ("qc_bandwidth_hz", "qc_hf_energy_db")

fails: list[str] = []


def check(cond: bool, msg: str):
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
    if not cond:
        fails.append(msg)


def digest(col) -> str:
    h = hashlib.sha256()
    for v in col:
        if v is None:
            h.update(b"\x00")
        elif isinstance(v, dict):
            h.update(v["bytes"] or b"")
        else:
            h.update(str(v).encode())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(
        description="Check that the v2 pack differs from v1 only where it claims to.")
    ap.add_argument("--v1", default=None, help="v1 pack (default: $MSBENCH_HOME/tts-bench-v1)")
    ap.add_argument("--v2", default=None, help="v2 pack (default: $MSBENCH_HOME/tts-bench-v2)")
    ap.add_argument("--langs", nargs="+", default=CODES)
    args = ap.parse_args()

    v1 = Path(args.v1) if args.v1 else pack("v1")
    v2 = Path(args.v2) if args.v2 else pack("v2")

    for lang in args.langs:
        f1 = v1 / "data" / lang / "main.parquet"
        # The build writes data/<lang>/main.parquet; the publish layout is
        # <lang>/main-*.parquet. Accept either, so validation can run both
        # before and after the pack is arranged for upload.
        cands = ([v2 / "data" / lang / "main.parquet"]
                 + sorted((v2 / lang).glob("main-*.parquet")))
        have = [p for p in cands if p.exists()]
        if not have:
            fails.append(f"{lang}: v2 parquet missing")
            continue
        print(f"\n=== {lang}")
        t1 = pq.read_table(f1)
        t2 = pa.concat_tables([pq.read_table(p) for p in have])

        check(t1.num_rows == t2.num_rows,
              f"row count unchanged ({t2.num_rows})")
        check(t1.column("utt").to_pylist() == t2.column("utt").to_pylist(),
              "utt identical and in the same order")

        for c in AUDIO:
            check(digest(t1.column(c).to_pylist()) ==
                  digest(t2.column(c).to_pylist()), f"{c} byte-identical")

        names1, names2 = set(t1.column_names), set(t2.column_names)
        untouched = names1 & names2 - {"phones", "tags"}
        changed = [c for c in sorted(untouched)
                   if digest(t1.column(c).to_pylist()) !=
                   digest(t2.column(c).to_pylist())]
        check(not changed, f"carried-over columns unchanged "
                           f"({len(untouched)} checked){'' if not changed else f': {changed}'}")

        for c in DROPPED:
            check(c not in names2, f"{c} removed (was 100% null)")
        old, new = RENAMED
        check(new in names2, f"{new} present")
        check(old in names2, f"{old} kept as a deprecated alias")
        check(t2.column(old).to_pylist() == t2.column(new).to_pylist(),
              "the renamed column carries the same values")

        # Compare the value type, not the printed form: pyarrow names the inner
        # field "item" or "element" depending on where the array came from, and
        # both are list<string>.
        tags = t2.schema.field("tags").type
        check(pa.types.is_list(tags) and pa.types.is_string(tags.value_type),
              f"tags is list<string>, not the degenerate list<null> (got {tags})")

        check("sim_ref_dur_bin" in names2, "sim_ref_dur_bin added")
        bins = set(t2.column("sim_ref_dur_bin").to_pylist())
        check(bins <= {"<3.5", "3.5-4.5", ">=4.5", "n/a"},
              f"sim_ref_dur_bin vocabulary is closed ({sorted(bins)})")

        # The anchors carried per row must reproduce the aggregates that the
        # reports publish, or the dataset and the card would disagree about the
        # benchmark's central numbers.
        agg = pd.read_csv(v2 / "reports" / "csv" / "anchors.csv")
        sm = pd.read_csv(v2 / "reports" / "csv" / "sim.csv")
        primary = t2.column("anchor_asr").to_pylist()[0]
        check(len(set(t2.column("anchor_asr").to_pylist())) == 1,
              f"anchor_asr constant within the subset ({primary})")
        row = agg[(agg.lang == lang) & (agg.ASR == primary)]
        wer = np.array(t2.column("anchor_wer").to_pylist(), dtype=float)
        check(len(row) == 1 and abs(np.nanmean(wer)
                                    - row.iloc[0]["WER macro (v1)"]) < 5e-5,
              "mean(anchor_wer) reproduces the published macro WER")
        has_gt = np.array(t2.column("has_gt").to_pylist())
        check(int(np.isfinite(wer).sum()) == int(has_gt.sum()),
              f"anchor_wer present for exactly the rows with gt audio "
              f"({int(has_gt.sum())})")

        # The headline aggregation must be derivable from the shipped columns
        # alone. Averaging anchor_wer gives the macro form; the corpus form
        # needs the counts, which is why they are here.
        def col(name):
            return np.array([np.nan if v is None else v
                             for v in t2.column(name).to_pylist()], dtype=float)

        num = np.nansum(col("anchor_subs") + col("anchor_dels")
                        + col("anchor_ins"))
        den = np.nansum(col("anchor_n_ref_words"))
        check(abs(num / den - row.iloc[0]["WER corpus"]) < 5e-5,
              f"corpus WER derivable from the dataset alone "
              f"({num / den:.4f} == published {row.iloc[0]['WER corpus']:.4f})")
        cnum, cden = np.nansum(col("anchor_cer_err")), np.nansum(
            col("anchor_n_ref_chars"))
        check(abs(cnum / cden - row.iloc[0]["CER corpus"]) < 5e-5,
              "corpus CER derivable from the dataset alone")

        # The same must hold for the other two recognisers, which reuse the
        # primary's reference lengths rather than shipping their own.
        for asr, suffix in (("mms", "mms"), ("elevenlabs", "scribe")):
            if asr == primary or f"anchor_wer_{suffix}" not in names2:
                continue
            r2 = agg[(agg.lang == lang) & (agg.ASR == asr)]
            if len(r2) != 1:
                # A column with no published aggregate to check it against is
                # worse than a missing one: it would ship unverified.
                check(False, f"{asr} column present but absent from "
                             f"reports/csv/anchors.csv — regenerate the reports")
                continue
            n2 = np.nansum(col(f"anchor_subs_{suffix}")
                           + col(f"anchor_dels_{suffix}")
                           + col(f"anchor_ins_{suffix}"))
            want = r2.iloc[0]["WER corpus"]
            check(abs(n2 / den - want) < 5e-5,
                  f"corpus WER for {asr} derivable too "
                  f"({n2 / den:.4f} == published {want:.4f})")
        for enc in ("wavlm_sv", "wavlm_ft", "ecapa"):
            s = np.array(t2.column(f"anchor_sim_{enc}").to_pylist(), dtype=float)
            want = sm[(sm.lang == lang) & (sm.encoder == enc)].iloc[0]["anchor"]
            check(abs(np.nanmean(s) - want) < 5e-5,
                  f"mean(anchor_sim_{enc}) reproduces the published anchor")
        has_sr = np.array(t2.column("has_sim_ref").to_pylist())
        sv = np.array(t2.column("anchor_sim_wavlm_sv").to_pylist(), dtype=float)
        check(int(np.isfinite(sv).sum()) == int(has_sr.sum()),
              f"anchor_sim present for exactly the rows with a second clip "
              f"({int(has_sr.sum())})")

        gt_cols = [c for c in names2 if c.startswith("gt_qc_")]
        sr_cols = [c for c in names2 if c.startswith("simref_qc_")]
        check(len(gt_cols) == 9, f"gt_qc_* block present ({len(gt_cols)} columns)")
        check(len(sr_cols) == 9,
              f"simref_qc_* block present ({len(sr_cols)} columns)")

        # The QC blocks must cover exactly the rows that have that audio.
        has_gt = t2.column("has_gt").to_pylist()
        ovrl = t2.column("gt_qc_dnsmos_ovrl").to_pylist()
        covered = sum(1 for h, v in zip(has_gt, ovrl, strict=True) if h and v is not None)
        check(covered / max(1, sum(has_gt)) > 0.98,
              f"gt QC covers {100 * covered / max(1, sum(has_gt)):.1f}% of rows with gt audio")

        if lang == "ky":
            ph = t2.column("phones").to_pylist()
            check(not any("[" in p for p in ph),
                  "espeak dental brackets gone from ky phones")
            check(any("t̪" in p or "d̪" in p for p in ph),
                  "IPA dental diacritics present instead")
            n1 = t1.column("n_phones").to_pylist()
            n2 = t2.column("n_phones").to_pylist()
            check(n1 == n2, "n_phones unaffected by the phone relabelling")

    print("\n" + "=" * 60)
    if fails:
        print(f"VALIDATION FAILED — {len(fails)} problem(s):")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("VALIDATION PASSED — v2 differs from v1 only where it claims to.")


if __name__ == "__main__":
    main()
