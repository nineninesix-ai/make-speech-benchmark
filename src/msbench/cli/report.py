#!/usr/bin/env python3
"""Consolidated v2 reports: every headline number with a CI, every cell with `n`.

Three things distinguish these reports from v1's `eval/report.py`.

**Confidence intervals everywhere.** v1 publishes point estimates only, on
observations that are not independent — one prompt serves up to seven examples
and the anchor is literally constant within a speaker (**D-20**). Every interval
here is a cluster bootstrap over speakers.

**`n` in every cell.** v1's breakdown tables give a mean and nothing else, and
per-cell counts fall to about 100 — small enough that a difference between bins
can be noise, and the reader has no way to tell.

**"Human anchor", never "ceiling".** A model can exceed it, and the Seed-TTS
paper shows models doing so (English SIM 0.762 against a human 0.730; Mandarin
WER 1.115 against a human 1.254). The asymmetry is structural: synthesis is
conditioned on the prompt and inherits its channel, while the anchor compares two
*different* recordings of the person (**D-11**).

    msbench-report --all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import msbench.stats as stats
from msbench.languages import CODES
from msbench.paths import workspace

ASRS = ["elevenlabs", "whisper", "mms", "gigaam"]
ENCODERS = ["wavlm_sv", "wavlm_ft", "ecapa"]
MIN_CELL = 30       # below this a breakdown cell is flagged as unreliable


def load(p: Path):
    return pd.read_parquet(p) if p.exists() else None


def side(p: Path):
    j = p.with_suffix(".json")
    return json.loads(j.read_text()) if j.exists() else {}


def ci(df, fn, cluster):
    if df is None or len(df) < 2 or cluster not in df:
        return float("nan"), float("nan"), float("nan")
    return stats.cluster_bootstrap(df, fn, cluster_col=cluster)


# ---------------------------------------------------------------- WER anchors

def wer_tables(rep: Path):
    rows, breakdown = [], []
    for lang in CODES:
        for asr in ASRS:
            f = rep / f"wer_{lang}_gt_{asr}.parquet"
            df = load(f)
            if df is None:
                continue
            meta = side(f)
            cluster = ("gt_speaker_id" if "gt_speaker_id" in df
                       and df.gt_speaker_id.notna().any() else "speaker_id")
            p, lo, hi = ci(df, stats.wer_corpus, cluster)
            cp, clo, chi = ci(df, stats.cer_corpus, cluster)
            rows.append({
                "lang": lang, "ASR": asr, "n": len(df),
                "speakers": int(df[cluster].nunique()),
                "WER corpus": round(p, 4),
                "95% CI": f"[{lo:.4f}, {hi:.4f}]",
                "WER macro (v1)": round(stats.macro("wer")(df), 4),
                "CER corpus": round(cp, 4),
                "exact": f"{100 * (df.wer == 0).mean():.1f}%",
                "catastrophic": f"{100 * (df.wer > 0.5).mean():.1f}%",
                "missing": meta.get("n_missing", 0),
                "empty ref": meta.get("n_empty_ref", 0),
            })
            for col, label in (("len_bin", "target length"),
                               ("speaker_gender", "gender")):
                if col not in df:
                    continue
                for v, g in df.groupby(col, observed=True):
                    if len(g) == 0:
                        continue
                    gp, glo, ghi = ci(g, stats.wer_corpus, cluster)
                    breakdown.append({
                        "lang": lang, "ASR": asr, "cut": label, "value": v,
                        "WER corpus": round(gp, 4),
                        "95% CI": (f"[{glo:.4f}, {ghi:.4f}]"
                                   if np.isfinite(glo) else "n/a"),
                        "n": len(g), "speakers": int(g[cluster].nunique()),
                        "small cell": len(g) < MIN_CELL})
    return pd.DataFrame(rows), pd.DataFrame(breakdown)


def asr_agreement(rep: Path):
    """Where the recognisers disagree, and by how much.

    A single ASR cannot separate what a system got wrong from what the ASR is
    bad at, so every available pair is reported.

    Two of the pairs carry distinct meanings. **Whisper−MMS** measures how much
    the intelligible reading depends on the listener's expectations: MMS decodes
    without a language model, so a large gap means the acoustics alone do not
    carry the sentence. **Scribe−Whisper** measures how much of the human anchor
    was never the speech at all but the recogniser's own error — on these
    subsets Scribe reads the same human recordings 31-59 % more accurately, which
    raises the ceiling a synthesis system is being compared against.
    """
    rows = []
    for lang in CODES:
        frames = {a: load(rep / f"wer_{lang}_gt_{a}.parquet") for a in ASRS}
        frames = {k: v for k, v in frames.items() if v is not None}
        names = list(frames)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pair = stats.paired_bootstrap(
                    frames[a], frames[b], stats.wer_corpus,
                    cluster_col=("gt_speaker_id" if "gt_speaker_id" in frames[a]
                                 else "speaker_id"))
                j = frames[a][["utt", "hyp_norm"]].merge(
                    frames[b][["utt", "hyp_norm"]], on="utt",
                    suffixes=("_a", "_b"))
                rows.append({
                    "lang": lang, "A": a, "B": b,
                    "WER A": round(stats.wer_corpus(frames[a]), 4),
                    "WER B": round(stats.wer_corpus(frames[b]), 4),
                    "delta (A−B)": round(pair["delta"], 4),
                    "95% CI": f"[{pair['lo']:.4f}, {pair['hi']:.4f}]",
                    "identical transcripts": (f"{100 * (j.hyp_norm_a == j.hyp_norm_b).mean():.1f}%"
                                              if len(j) else "n/a"),
                    "n shared": pair["n_shared"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- SIM anchors

def sim_tables(rep: Path):
    rows, breakdown = [], []
    for lang in CODES:
        for enc in ENCODERS:
            fa = rep / f"sim_{lang}_anchor_{enc}.parquet"
            ff = rep / f"sim_{lang}_floor_{enc}.parquet"
            a, f = load(fa), load(ff)
            if a is None:
                continue
            p, lo, hi = ci(a, stats.macro("sim"), "speaker_id")
            r = {"lang": lang, "encoder": enc, "n rows": len(a),
                 "speakers": int(a.speaker_id.nunique()),
                 "distinct values": int(a.sim.round(6).nunique()),
                 "anchor": round(p, 4), "95% CI": f"[{lo:.4f}, {hi:.4f}]"}
            if f is not None:
                fl = f.sim_floor
                r.update({"floor mean": round(fl.mean(), 4),
                          "floor p95": round(fl.quantile(.95), 4),
                          "floor pairs": len(f),
                          "usable range": round(p - fl.mean(), 4)})
                # A model's readable score. Reported here as the transform, not
                # as a value: with no synthesis in this run there is nothing to
                # normalise yet.
                r["sim_norm formula"] = (f"(x − {fl.mean():.4f}) / "
                                         f"{p - fl.mean():.4f}")
            rows.append(r)

            for col, label in (("prompt_dur_bin", "prompt duration"),
                               ("sim_ref_dur_bin", "second-clip duration"),
                               ("speaker_gender", "gender")):
                if col not in a:
                    continue
                for v, g in a.groupby(col, observed=True):
                    gp, glo, ghi = ci(g, stats.macro("sim"), "speaker_id")
                    breakdown.append({
                        "lang": lang, "encoder": enc, "cut": label, "value": v,
                        "SIM": round(gp, 4),
                        "95% CI": (f"[{glo:.4f}, {ghi:.4f}]"
                                   if np.isfinite(glo) else "n/a"),
                        "n": len(g), "speakers": int(g.speaker_id.nunique()),
                        "small cell": int(g.speaker_id.nunique()) < MIN_CELL})
    return pd.DataFrame(rows), pd.DataFrame(breakdown)


# ------------------------------------------------------------------- quality

def quality_table(rep: Path):
    rows = []
    for lang in CODES:
        for audio in ("prompt", "gt", "sim_ref"):
            df = load(rep / f"quality_{lang}_{audio}.parquet")
            if df is None:
                continue
            rows.append({
                "lang": lang, "audio": audio, "n": len(df),
                "DNSMOS OVRL": round(df.dnsmos_ovrl.mean(), 3),
                "OVRL p05": round(df.dnsmos_ovrl.quantile(.05), 3),
                "clipping %": round(100 * df.qc_fail_clipping.mean(), 2),
                "narrowband %": round(100 * df.qc_fail_narrowband.mean(), 2),
                "low speech %": round(100 * df.qc_fail_speech_ratio.mean(), 2),
                "would fail prompt QC %": round(100 * df.qc_fail_any.mean(), 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- main

def section(title: str, body: str, df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty:
        return []
    return [f"\n## {title}\n", body + "\n", df.to_markdown(index=False), ""]


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--pack", default=str(here / "tts-bench-v1"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rep = Path(args.pack) / "reports" / "v2"
    wer, wer_cut = wer_tables(rep)
    agree = asr_agreement(rep)
    sim, sim_cut = sim_tables(rep)
    qual = quality_table(rep)

    md = [
        "# Human anchors — v2\n",
        "Every figure is a **human anchor**, not a ceiling. A synthesis system "
        "can and does exceed it: synthesis is conditioned on the prompt and "
        "inherits its recording channel, whereas the anchor compares two "
        "*different* recordings of the same person. The comparison is asymmetric "
        "by construction (D-11).\n",
        "Intervals are 95 % cluster bootstraps resampling **speakers**, not rows. "
        "Rows are not independent: one prompt serves up to seven examples, and "
        "the SIM anchor is constant within a speaker by construction (D-20).\n",
        "Anchor-normalised comparison is valid **within a language only**. "
        "Cross-language differences in the anchor are a property of the "
        "recogniser, and for `ky` it is a different recogniser entirely (D-19).\n",
    ]

    md += section(
        "WER anchor by recogniser",
        "`WER corpus` is the headline: Σ(S+D+I) / Σ N_ref, the seed-tts-eval "
        "definition. `WER macro (v1)` is the mean of per-utterance rates, which "
        "is what v1 published; on texts with a median of 6-7 words the two differ "
        "substantially (D-01). `catastrophic` is the share of utterances above "
        "50 % WER — the indicator that actually catches looping and dropped "
        "clauses.", wer)

    md += section(
        "Where the recognisers disagree",
        "MMS decodes with CTC and no language model; Whisper's internal LM "
        "repairs mispronunciations. A large gap means the intelligible reading "
        "depends on the listener's expectations, not on the acoustics (G-03). "
        "Deltas are paired bootstraps over the utterances both recognisers "
        "scored.", agree)

    md += section(
        "SIM anchor, impostor floor, and the usable range",
        "`floor` is the mean cosine between prompts of **different** speakers — "
        "the score an impostor gets for free. `usable range` is anchor minus "
        "floor: the entire span in which a cloning system can distinguish itself. "
        "Where that span is narrow, an absolute SIM value carries almost no "
        "information, which is why v1's anchor-only presentation could not be "
        "read (G-02). `distinct values` is below `n rows` because every example "
        "of a speaker shares one prompt and one second clip.", sim)

    md += section(
        "SIM breakdowns",
        "Every cell carries `n` and the number of distinct speakers behind it. "
        "`small cell` marks fewer than "
        f"{MIN_CELL} speakers, where the mean should not be read as a trend. "
        "`second-clip duration` is the control v1 lacks: the card bins by prompt "
        "duration only, leaving the other clip in the pair — down to ~2 s — "
        "uncontrolled (D-21).", sim_cut)

    md += section(
        "WER breakdowns",
        "Same rules: `n`, speakers, and a flag on thin cells.", wer_cut)

    md += section(
        "Audio quality of the three streams",
        "`would fail prompt QC %` applies the prompt-acceptance thresholds to "
        "each stream. Prompts pass by construction. `gt` and `sim_ref` never "
        "faced those filters during the build (D-09), so this column measures "
        "how much of the human anchor rests on audio that would have been "
        "rejected as a prompt. Reported, not filtered: removing rows would break "
        "`utt` stability with v1.", qual)

    out = Path(args.out) if args.out else rep / "results_v2.md"
    out.write_text("\n".join(md) + "\n")
    for name, df in (("anchors", wer), ("agreement", agree), ("sim", sim),
                     ("sim_breakdown", sim_cut), ("wer_breakdown", wer_cut),
                     ("quality", qual)):
        if df is not None and not df.empty:
            df.to_csv(rep / f"{name}.csv", index=False)
            print(f"\n## {name}\n")
            print(df.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
