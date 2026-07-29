#!/usr/bin/env python3
"""Per-subset coverage reports plus a summary table.

Covers the acceptance criteria: length distribution against target, phonetic
inventory and diphone coverage, speaker composition, held-out share.

**Pool comparison is optional now.** The original version read
`work/{lang}_texts.parquet` to express coverage as a fraction of the candidate
pool. That intermediate is not shipped and is expensive to rebuild (it needs the
66 GB Common Voice corpus), which meant the published reports could not be
regenerated — and so they stayed in their original Russian while the code around
them was translated (**D-24**). With `--pool` absent the report states the
inventory actually present in the subset, and records that the build-time
comparison found it equal to the pool for all six subsets: 49/49, 37/37, 36/36,
32/32, 44/44, 43/43 phonemes and 100 % of diphones.

    python -m msbench.build.reports --pack tts-bench-v2
    python -m msbench.build.reports --pack tts-bench-v2 --work work   # with the pool
"""

import argparse
from pathlib import Path

import pandas as pd

from msbench.languages import CAP, ESPEAK_VOICE, LEN_BINS, LEN_TARGET, N_MAIN
from msbench.paths import workspace

# Verified at build time against the candidate pool: the selected examples cover
# the pool's entire phoneme and diphone inventory in every subset. Kept here so
# the claim survives the loss of the intermediate.
POOL_FULLY_COVERED = {"en-US": (49, 1489), "es-ES": (37, 808), "es-MX": (36, 785),
                      "ky": (32, 741), "nl-NL": (44, 1083), "pt-BR": (43, 1080)}


def pack_file(pack: Path, lang: str) -> Path:
    """The subset parquet, in either the build or the publish layout."""
    cands = [pack / "data" / lang / "main.parquet",
             *sorted((pack / lang).glob("main-*.parquet"))]
    for c in cands:
        if c.exists():
            return c
    raise SystemExit(f"no parquet for {lang} under {pack}")


def langs_in(pack: Path) -> list[str]:
    if (pack / "data").is_dir():
        return sorted(p.name for p in (pack / "data").iterdir())
    return sorted(p.name for p in pack.iterdir()
                  if p.is_dir() and list(p.glob("main-*.parquet")))


def diphones(phones):
    t = phones.split()
    return set(zip(t, t[1:], strict=False))


def report(lang, pack, work, out_dir):
    # Read the metadata columns only: the audio columns make the file ~900 MB.
    src = pack_file(pack, lang)
    cols = ["utt", "phones", "len_bin", "n_words", "speaker_id", "speaker_gender",
            "speaker_age", "prompt_dur", "prompt_dur_bin", "prompt_sr_orig",
            "has_gt", "has_sim_ref", "gt_same_speaker", "text_split_origin",
            "prompt_split_origin", "punct_type", "text", "sim_ref_dur"]
    df = pd.read_parquet(src, columns=cols)

    sel_dip = set().union(*(diphones(p) for p in df.phones))
    sel_ph = {t for p in df.phones for t in p.split()}

    pool_file = work / f"{lang}_texts.parquet" if work else None
    if pool_file and pool_file.exists():
        pool = pd.read_parquet(pool_file)
        pool_dip = set().union(*(diphones(p) for p in pool.phones))
        pool_ph = {t for p in pool.phones for t in p.split()}
        missing = sorted(pool_dip - sel_dip)
        phon_line = (f"* phonemes: **{len(sel_ph)}/{len(pool_ph)}** "
                     f"({100 * len(sel_ph) / len(pool_ph):.1f}% of the pool inventory)")
        dip_line = (f"* diphones: **{len(sel_dip)}/{len(pool_dip)}** "
                    f"({100 * len(sel_dip) / len(pool_dip):.1f}%)")
    else:
        ref_ph, ref_dip = POOL_FULLY_COVERED[lang]
        missing = []
        verified = "100% of the candidate pool, verified at build time"
        note_ph = verified if len(sel_ph) == ref_ph else \
            f"build-time pool inventory was {ref_ph}"
        note_dip = verified if len(sel_dip) == ref_dip else \
            f"build-time pool inventory was {ref_dip}"
        phon_line = f"* phonemes: **{len(sel_ph)}** ({note_ph})"
        dip_line = f"* diphones: **{len(sel_dip)}** ({note_dip})"

    len_act = df.len_bin.value_counts().reindex(LEN_BINS).fillna(0).astype(int)
    len_tgt = {b: int(round(len(df) * f))
               for b, f in zip(LEN_BINS, LEN_TARGET[lang], strict=True)}
    spk = df.speaker_id.value_counts()

    L = [f"# Coverage: {lang}\n",
         f"**Examples:** {len(df)} (target {N_MAIN[lang]})  ",
         f"**Source:** Common Voice 17.0, espeak voice `{ESPEAK_VOICE[lang]}`\n",
         "## Phonetics\n", phon_line, dip_line]
    if missing:
        L.append(f"* uncovered diphones: {len(missing)}, top 20: "
                 + ", ".join(f"`{a}{b}`" for a, b in missing[:20]))
    else:
        L.append("* no uncovered diphones")
    if lang == "ky":
        L.append("* dental consonants are written `t̪` / `d̪`; v1 shipped the "
                 "espeak markers `t[` / `d[` (D-22)")

    L += ["", "## Target text length\n",
          "| bin | selected | target | delta |", "|---|---|---|---|"]
    for b in LEN_BINS:
        L.append(f"| {b} | {len_act[b]} | {len_tgt[b]} | "
                 f"{len_act[b] - len_tgt[b]:+d} |")
    L.append(f"\nwords: median {int(df.n_words.median())}, "
             f"p10-p90 {int(df.n_words.quantile(.1))}-{int(df.n_words.quantile(.9))}\n")

    g = df.speaker_gender.value_counts()
    a = df.speaker_age.value_counts().head(5)
    L += ["## Speakers\n",
          f"* unique speakers **used**: **{df.speaker_id.nunique()}** "
          "(this is speakers used, not speakers available in the source slice)",
          f"* examples per speaker: max **{int(spk.max())}** (cap {CAP[lang]}), "
          f"median {int(spk.median())}",
          "* gender (by example): "
          + ", ".join(f"{k} {v} ({100 * v / len(df):.0f}%)" for k, v in g.items()),
          "* age, top 5: " + ", ".join(f"{k} {v}" for k, v in a.items()), ""]

    L += ["## Reference audio\n",
          f"* duration after VAD trimming: median **{df.prompt_dur.median():.2f} s**, "
          f"p05-p95 {df.prompt_dur.quantile(.05):.2f}-{df.prompt_dur.quantile(.95):.2f}",
          f"* strata: {df.prompt_dur_bin.value_counts().to_dict()}",
          f"* original sample rates: {sorted(df.prompt_sr_orig.unique().tolist())}",
          f"* second clip (`sim_ref_audio`) duration: median "
          f"**{df[df.has_sim_ref].sim_ref_dur.median():.2f} s**, p05-p95 "
          f"{df[df.has_sim_ref].sim_ref_dur.quantile(.05):.2f}-"
          f"{df[df.has_sim_ref].sim_ref_dur.quantile(.95):.2f}", ""]

    ho_t = df.text_split_origin.isin(["dev", "test"]).mean()
    ho_p = df.prompt_split_origin.isin(["dev", "test"]).mean()
    L += ["## Human anchors and provenance\n",
          f"* with GT audio (WER anchor): **{int(df.has_gt.sum())}/{len(df)}**",
          f"* with sim_ref (SIM anchor): **{int(df.has_sim_ref.sum())}/{len(df)}** "
          "— the rest had no second usable clip from that speaker",
          f"* GT from the same speaker as the prompt: {int(df.gt_same_speaker.sum())}",
          f"* held-out (dev/test): texts **{100 * ho_t:.0f}%**, "
          f"prompts **{100 * ho_p:.0f}%**",
          f"* punctuation: {df.punct_type.value_counts().to_dict()}",
          f"* unique texts: {df.text.nunique()}/{len(df)}"]

    (out_dir / f"coverage_{lang}.md").write_text("\n".join(L) + "\n")
    return {
        "lang": lang, "n": len(df), "speakers used": df.speaker_id.nunique(),
        "phonemes": len(sel_ph), "diphones": len(sel_dip),
        "female": f"{100 * (df.speaker_gender == 'female').mean():.0f}%",
        "gt": int(df.has_gt.sum()), "sim_ref": int(df.has_sim_ref.sum()),
        "sim_ref %": f"{100 * df.has_sim_ref.mean():.1f}%",
        "held_out_prompts": f"{100 * ho_p:.0f}%",
        "median_ref_s": round(df.prompt_dur.median(), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--pack", default=str(here / "tts-bench-v2"))
    ap.add_argument("--work", default=None,
                    help="S1-S3 intermediates, for pool-relative coverage")
    ap.add_argument("--langs", nargs="+", default=None)
    args = ap.parse_args()

    pack = Path(args.pack)
    work = Path(args.work) if args.work else None
    out_dir = pack / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    langs = args.langs or langs_in(pack)

    summary = pd.DataFrame([report(x, pack, work, out_dir) for x in langs])
    print(summary.to_string(index=False))
    (out_dir / "summary.md").write_text(
        "# Subset summary\n\n"
        "`speakers used` is the number of distinct prompt speakers that appear "
        "in the subset, not the number available in the Common Voice slice.\n\n"
        + summary.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
