#!/usr/bin/env python3
"""Attribute the v1 → v2 change in the human anchor to one cause per step.

The plan's stack-validation gate (§11.1) cannot be run as written: it re-aggregates
the stored `ref_norm`/`hyp_norm` from `reports/wer_{lang}_gt.parquet`, and those
files exist nowhere. They are written inside `tts-bench-v1/`, which is gitignored,
and the published dataset ships three markdown reports and no parquet. The
artifact the gate depends on is itself a v2 deliverable.

The gate is therefore split in two, and this is the *inexact* half:

  * `eval/tests/test_metrics.py` — exact, no GPU. Fixes the input strings and
    asserts to machine precision that the new aggregation reproduces v1's macro
    formula and the seed-tts-eval corpus formula.
  * this script — inexact, GPU. Re-decodes the audio under v1's exact settings
    and compares against the published numbers.

The second cannot be an assertion. Reproducing a decode from a different
`transformers`, a different GPU and a different cuDNN will not land on the fourth
decimal, and if it misses, the mismatch has five candidate causes at once. So the
comparison is **measured and reported**, never enforced: a difference beyond
`--tolerance` prints a warning to investigate, and the number goes into
`CHANGELOG.md` on its own line as environment drift rather than being absorbed
into a methodological delta.

What the ladder does buy is clean attribution, because each rung changes exactly
one thing on identical audio:

    published v1        the numbers in the dataset card
      ↓  environment drift — different library and hardware versions
    legacy              repetition_penalty=1.1, chunk_length_s=30, macro
      ↓  D-02           the repetition penalty removed
    nopen               chunk_length_s=30, macro
      ↓  D-10           long-form chunking off
    v2 (macro)          short-form path, macro
      ↓  D-01           corpus aggregation instead of the macro mean
    v2 (corpus)         the v2 headline

Run the three passes first (see `--help` for the exact commands), then this.

    msbench-replay
"""

import argparse
from pathlib import Path

import pandas as pd

import msbench.stats as stats
from msbench.paths import workspace

# Appendix B — the published v1 anchors, macro-averaged, with
# repetition_penalty=1.1. Source: reports/results.md and reports/summary.md of
# the released dataset.
PUBLISHED = {
    "en-US": {"wer": 0.0806, "cer": 0.0311, "n": 1500, "asr": "whisper"},
    "es-ES": {"wer": 0.0485, "cer": 0.0173, "n": 1499, "asr": "whisper"},
    "es-MX": {"wer": 0.0654, "cer": 0.0235, "n": 1500, "asr": "whisper"},
    "nl-NL": {"wer": 0.0407, "cer": 0.0118, "n": 1500, "asr": "whisper"},
    "pt-BR": {"wer": 0.0823, "cer": 0.0243, "n": 1500, "asr": "whisper"},
    "ky":    {"wer": 0.1104, "cer": 0.0336, "n": 694,  "asr": "gigaam"},
}

# Appendix B again, SIM side: the anchor from `microsoft/wavlm-base-plus-sv`
# over the rows where `has_sim_ref`. This one *is* exactly reproducible — no
# sampling, no decoding, a deterministic forward pass on stored 16 kHz audio —
# so it is checked to four decimals (plan §11.2).
PUBLISHED_SIM = {
    "en-US": {"sim": 0.9317, "n": 909},
    "es-ES": {"sim": 0.9435, "n": 1222},
    "es-MX": {"sim": 0.9456, "n": 1201},
    "nl-NL": {"sim": 0.9285, "n": 1319},
    "pt-BR": {"sim": 0.9339, "n": 983},
    "ky":    {"sim": 0.9214, "n": 599},
}

# rung -> filename suffix produced by run_wer2.py
RUNGS = [
    ("legacy", "_legacy", "repetition_penalty=1.1, chunk_length_s=30"),
    ("nopen", "_nopen", "penalty removed, chunk_length_s=30"),
    ("v2", "", "penalty removed, short-form path"),
]


def load(rep: Path, lang: str, asr: str, suffix: str):
    f = rep / f"wer_{lang}_gt_{asr}{suffix}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Share of utterances where the two passes produced the identical string.

    A far sharper diagnostic than the aggregate: two passes can agree on WER to
    three decimals while disagreeing on hundreds of clips, and can disagree on
    the aggregate while differing on a handful. If this is high, whatever moved
    the aggregate moved it through a small number of rows.
    """
    j = a[["utt", "hyp_norm"]].merge(b[["utt", "hyp_norm"]], on="utt",
                                     suffixes=("_a", "_b"))
    return float((j.hyp_norm_a == j.hyp_norm_b).mean()) if len(j) else float("nan")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Produce the three passes first:

  for L in en-US es-ES es-MX nl-NL pt-BR; do
    msbench-wer --lang $L --audio gt --asr whisper --batch 8 \\
        --repetition-penalty 1.1 --chunk-length-s 30 --tag legacy
    msbench-wer --lang $L --audio gt --asr whisper --batch 8 \\
        --chunk-length-s 30 --tag nopen
    msbench-wer --lang $L --audio gt --asr whisper --batch 8
  done
  msbench-wer --lang ky --audio gt --asr gigaam
""")
    here = workspace()
    ap.add_argument("--pack", default=str(here / "tts-bench-v1"))
    ap.add_argument("--tolerance", type=float, default=0.003,
                    help="absolute WER difference treated as environment drift")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rep = Path(args.pack) / "reports" / "v2"
    rows, notes = [], []

    for lang, pub in PUBLISHED.items():
        asr = pub["asr"]
        frames = {k: load(rep, lang, asr, sfx) for k, sfx, _ in RUNGS}
        # GigaAM has no decoding parameters to replay: the v1 configuration and
        # the v2 configuration are the same call, so one pass serves as all three.
        if asr == "gigaam" and frames["v2"] is not None:
            frames = {k: frames["v2"] for k in frames}
        if frames["legacy"] is None and frames["v2"] is None:
            print(f"{lang}: no runs found, skipping")
            continue

        r = {"lang": lang, "ASR": asr, "published v1 (macro)": pub["wer"]}
        prev = None
        for key, _, _ in RUNGS:
            df = frames[key]
            if df is None:
                continue
            m = stats.macro("wer")(df)
            r[f"{key} (macro)"] = round(m, 4)
            r[f"n {key}"] = len(df)
            prev = df
        if prev is not None:
            r["v2 (corpus)"] = round(stats.wer_corpus(prev), 4)
            p, lo, hi = stats.cluster_bootstrap(
                prev, stats.wer_corpus,
                cluster_col="gt_speaker_id" if "gt_speaker_id" in prev else
                "speaker_id")
            r["v2 corpus 95% CI"] = f"[{lo:.4f}, {hi:.4f}]"

        # One delta per rung, each isolating a single cause.
        if frames["legacy"] is not None:
            drift = r["legacy (macro)"] - pub["wer"]
            r["drift (env)"] = round(drift, 4)
            if abs(drift) > args.tolerance:
                notes.append(
                    f"{lang}: replaying the v1 settings gives "
                    f"{r['legacy (macro)']:.4f} against a published "
                    f"{pub['wer']:.4f} — {drift:+.4f}, beyond the "
                    f"{args.tolerance} drift tolerance. Investigate before "
                    "quoting the v1→v2 deltas for this subset.")
        if frames["legacy"] is not None and frames["nopen"] is not None:
            r["D-02 penalty"] = round(r["nopen (macro)"] - r["legacy (macro)"], 4)
            r["agree legacy/nopen"] = round(
                agreement(frames["legacy"], frames["nopen"]), 4)
        if frames["nopen"] is not None and frames["v2"] is not None:
            r["D-10 chunking"] = round(r["v2 (macro)"] - r["nopen (macro)"], 4)
            r["agree nopen/v2"] = round(agreement(frames["nopen"],
                                                  frames["v2"]), 4)
        if "v2 (corpus)" in r and "v2 (macro)" in r:
            r["D-01 aggregation"] = round(r["v2 (corpus)"] - r["v2 (macro)"], 4)
        rows.append(r)

    # SIM parity is a separate matter: the anchor is a deterministic forward pass
    # over stored audio, with no decoding and no sampling, so it either
    # reproduces exactly or the encoder pipeline has changed.
    sim_rows = []
    for lang, pub in PUBLISHED_SIM.items():
        f = rep / f"sim_{lang}_anchor_wavlm_sv.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        got = float(df.sim.mean())
        sim_rows.append({
            "lang": lang, "published v1": pub["sim"], "v2 replay": round(got, 4),
            "delta": round(got - pub["sim"], 4),
            "published n": pub["n"], "n rows": len(df),
            "distinct values": int(df.sim.round(6).nunique()),
            "speakers": int(df.speaker_id.nunique()),
            "matches to 4 dp": abs(round(got, 4) - pub["sim"]) < 5e-5})
    sim_t = pd.DataFrame(sim_rows)

    t = pd.DataFrame(rows)
    print("\n## v1 → v2 attribution ladder (human anchor, WER)\n")
    print(t.to_string(index=False))
    if not sim_t.empty:
        print("\n## SIM anchor parity with v1 (same encoder, wavlm_sv)\n")
        print(sim_t.to_string(index=False))
        bad = sim_t[~sim_t["matches to 4 dp"]]
        for r in bad.itertuples():
            notes.append(
                f"{r.lang}: SIM anchor {r._3} against a published {r._2} "
                f"({r.delta:+.4f}). This computation is deterministic, so a "
                "difference means the encoder or the audio path changed — "
                "investigate before publishing any SIM number.")
    for n in notes:
        print(f"\n!! {n}")
    if not notes:
        print(f"\nAll subsets reproduce the published v1 numbers within "
              f"±{args.tolerance} under v1 settings.")

    md = ["# v1 → v2 attribution ladder\n",
          "Each column is one deliberate change, measured on identical audio, so "
          "the migration table has a cause per delta rather than one lump.\n",
          "* **drift (env)** — replaying v1's exact decoding parameters on current "
          "library and hardware versions. Not a methodological change; reported so "
          "it is not mistaken for one.\n",
          "* **D-02 penalty** — `repetition_penalty=1.1` removed. Positive means "
          "the reported anchor was flattered by the penalty.\n",
          "* **D-10 chunking** — Whisper's long-form path switched off for clips of "
          "2-6 s.\n",
          "* **D-01 aggregation** — corpus-level instead of the macro mean. This is "
          "the largest term and it is a redefinition, not a correction.\n",
          t.to_markdown(index=False), ""]
    if not sim_t.empty:
        md += ["\n## SIM anchor parity with v1\n",
               "Same encoder, same stored audio, no decoding — this one is "
               "expected to reproduce exactly, and is the check that the audio "
               "path was not disturbed.\n",
               "`distinct values` is the number the anchor is really measured "
               "on: every example of a speaker shares one prompt clip and one "
               "second clip, so the published `n` counts rows, not "
               "measurements (D-20).\n",
               sim_t.to_markdown(index=False), ""]
    if notes:
        md += ["\n## Warnings\n"] + [f"* {n}\n" for n in notes]
    out = Path(args.out) if args.out else rep / "attribution.md"
    out.write_text("\n".join(md) + "\n")
    t.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
