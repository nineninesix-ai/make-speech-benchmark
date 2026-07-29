#!/usr/bin/env python3
"""S1 — hard filters over the TSVs, producing candidate pools. No audio is read.

Two pools per language:
  work/{lang}_texts.parquet    target texts with phonemes and metadata
  work/{lang}_prompts.parquet  prompt candidates, up to K per speaker

Targets and prompts are **decoupled** — only the language and regional variant
have to match. That is what lets the S3 selector choose texts freely for
coverage and length instead of being tied to whatever the prompt speaker
happened to record.

    python -m msbench.build.pools --langs ky
"""

import argparse
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from msbench.corpus.inventory import load, select_target
from msbench.corpus.phonemes import phonemize_checked
from msbench.languages import ALPHABET, CAP, CODES, LANGUAGES, N_MAIN, len_bin
from msbench.paths import workspace

PROMPT_DUR = (3.0, 8.0)   # candidate window for the reference clip
TARGET_DUR = (2.0, 12.0)  # window for the clip backing a target text (its GT)
MIN_WORDS = 3


def punct_type(text):
    q, e = "?" in text, "!" in text
    if q and e:
        return "mixed"
    return "question" if q else "exclamation" if e else "declarative"


def norm_text(s):
    s = unicodedata.normalize("NFC", str(s)).strip()
    return re.sub(r"\s+", " ", s)


def alien_ratio(text, pattern):
    """Share of letters outside the language's alphabet."""
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 1.0
    return sum(0 if re.match(pattern, c) else 1 for c in letters) / len(letters)


def build(lang, root, out_dir, per_speaker):
    df = load(str(root), LANGUAGES[lang].locale)
    sub = select_target(df, lang)

    # pt has 9,464 clips listed in both train and dev — the same file, twice.
    # Dedup must be by `path`, not only by sentence.
    n0 = len(sub)
    sub = sub.drop_duplicates("path", keep="first").copy()
    dropped_path = n0 - len(sub)

    sub["sentence"] = sub.sentence.fillna("").map(norm_text)
    sub["n_words"] = sub.sentence.str.split().str.len()
    sub["up"] = pd.to_numeric(sub.up_votes, errors="coerce").fillna(0)
    sub["down"] = pd.to_numeric(sub.down_votes, errors="coerce").fillna(0)

    votes_ok = (sub.up >= 2) & (sub.down == 0)
    stats = {"clips in slice": n0, "duplicate paths dropped": dropped_path,
             "after vote filter": int(votes_ok.sum())}
    sub = sub[votes_ok]

    # --- target text pool ----------------------------------------------------
    pat = ALPHABET[lang]
    t = sub[(sub.dur >= TARGET_DUR[0]) & (sub.dur <= TARGET_DUR[1])].copy()
    t = t[t.n_words >= MIN_WORDS]
    t = t[~t.sentence.str.contains(r"[0-9]", regex=True)]
    t = t[t.sentence.map(lambda s: alien_ratio(s, pat) < 0.05)]
    stats["texts after filters"] = len(t)
    t = t.drop_duplicates("sentence_id").drop_duplicates("sentence")
    stats["texts after dedup"] = len(t)

    # Phonemise and drop texts where espeak silently switched language — those
    # carry another language's phonemes and would pollute the inventory.
    t0 = time.time()
    res = phonemize_checked(t.sentence.tolist(), lang)
    t["phones"] = [" ".join(toks) for toks, _ in res]
    t["n_phones"] = [len(toks) for toks, _ in res]
    t["lang_switch"] = [sw for _, sw in res]
    stats["phonemisation, s"] = round(time.time() - t0)
    stats["language switches dropped"] = int(t.lang_switch.sum())
    t = t[~t.lang_switch].drop(columns=["lang_switch"])
    t["len_bin"] = t.n_words.map(len_bin)
    t["punct_type"] = t.sentence.map(punct_type)
    stats["TEXTS TOTAL"] = len(t)

    # --- prompt candidate pool ----------------------------------------------
    p = sub[(sub.dur >= PROMPT_DUR[0]) & (sub.dur <= PROMPT_DUR[1])].copy()
    p = p[p.sentence.map(lambda s: alien_ratio(s, pat) < 0.05)]
    # Speakers actually needed is N/cap; take 2.5x that for selection freedom.
    need = int(np.ceil(N_MAIN[lang] / CAP[lang]) * 2.5)
    vc = p.client_id.value_counts()
    # Prefer speakers with dev/test clips — a held-out subset then comes for free.
    heldout = (p[p.split.isin(["dev", "test"])].client_id.value_counts()
               .reindex(vc.index).fillna(0))
    order = pd.DataFrame({"n": vc, "ho": heldout}).sort_values(
        ["ho", "n"], ascending=[False, False])
    p = p[p.client_id.isin(order.index[:need])]

    def spread(g):
        # Thin out by duration so a speaker's candidates span the duration
        # strata instead of clustering at one end.
        return g.iloc[:: max(1, len(g) // per_speaker)].head(per_speaker)

    p = (p.sort_values(["client_id", "dur"])
         .groupby("client_id", group_keys=False)[p.columns.tolist()]
         .apply(spread))
    stats["speakers in prompt pool"] = p.client_id.nunique()
    stats["prompt candidate clips"] = len(p)

    cols_t = ["path", "sentence", "sentence_id", "client_id", "gender", "age",
              "accents", "variant", "split", "dur", "n_words", "n_phones",
              "phones", "len_bin", "punct_type"]
    cols_p = ["path", "sentence", "client_id", "gender", "age", "accents",
              "variant", "split", "dur"]
    out_dir.mkdir(parents=True, exist_ok=True)
    t[cols_t].to_parquet(out_dir / f"{lang}_texts.parquet", index=False)
    p[cols_p].to_parquet(out_dir / f"{lang}_prompts.parquet", index=False)

    print(f"\n=== {lang} ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--root", default=str(here / "cv"))
    ap.add_argument("--out", default=str(here / "work"))
    ap.add_argument("--langs", nargs="+", default=CODES)
    ap.add_argument("--per-speaker", type=int, default=4)
    args = ap.parse_args()

    for lang in args.langs:
        build(lang, Path(args.root), Path(args.out), args.per_speaker)


if __name__ == "__main__":
    main()
