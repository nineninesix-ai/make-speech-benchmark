#!/usr/bin/env python3
"""Inventory a Common Voice locale for one of the benchmark's language variants.

Produces the numbers that every downstream decision rests on: how many clips and
speakers a variant actually has, how they split by gender and duration, and
whether the train/dev/test splits are disjoint.

Language definitions (locale, label predicate, thresholds) live in
`msbench.languages` — adding a language means editing that file, not this one.

    python -m msbench.corpus.inventory --data-dir ./cv
    python -m msbench.corpus.inventory --langs en-US ky --labels
"""

import argparse
import collections
import sys

import numpy as np
import pandas as pd

from msbench.languages import CODES, LANGUAGES, select, split_labels

SPLITS = ["train", "dev", "test"]

# Kept as a module-level alias: several build stages import TARGETS to learn
# which CV locale a language variant comes from.
TARGETS = {code: (spec.locale, spec.column, spec.match)
           for code, spec in LANGUAGES.items()}


def read_split(data_dir, locale, split):
    # quoting=3 (QUOTE_NONE): CV sentences contain bare quote characters that
    # would otherwise swallow half the file.
    path = f"{data_dir}/{locale}/{split}.tsv"
    return pd.read_csv(path, sep="\t", quoting=3, dtype=str).assign(split=split)


def load(data_dir, locale, with_duration=True):
    df = pd.concat([read_split(data_dir, locale, s) for s in SPLITS],
                   ignore_index=True)
    if with_duration:
        dur = pd.read_csv(f"{data_dir}/{locale}/clip_durations.tsv",
                          sep="\t", quoting=3)
        dur.columns = ["path", "dur_ms"]
        df = df.merge(dur, on="path", how="left")
        df["dur"] = df.dur_ms.astype(float) / 1000
    return df


def select_target(df, lang):
    """Rows belonging to the language variant. Thin wrapper over languages.select."""
    return select(df, lang)


def label_inventory(df, col, top=20):
    """Every distinct label in a locale, so region naming never has to be guessed."""
    cnt, spk = collections.Counter(), collections.defaultdict(set)
    for s, cid in zip(df[col].fillna(""), df.client_id, strict=True):
        for x in split_labels(s):
            cnt[x] += 1
            spk[x].add(cid)
    return [(lbl, n, len(spk[lbl])) for lbl, n in cnt.most_common(top)]


def report(data_dir, lang, show_labels=False):
    spec = LANGUAGES[lang]
    df = load(data_dir, spec.locale)
    sub = select_target(df, lang)
    sub["nw"] = sub.sentence.fillna("").str.split().str.len()

    spk_gender = sub.groupby("client_id").gender.first().fillna("n.d.").value_counts()
    vc = sub.client_id.value_counts()

    print("=" * 78)
    print(f"{lang}  (locale {spec.locale}, column {spec.column})"
          f"   train+dev+test total: {len(df)} clips")
    if show_labels:
        print("  label inventory:")
        for lbl, n, ns in label_inventory(df, spec.column):
            print(f"    {n:>8} clips {ns:>6} spk | {lbl[:70]}")
    print(f"  slice: {len(sub)} clips, {sub.client_id.nunique()} speakers")
    print(
        f"  speakers M/F/n.d.: {spk_gender.get('male_masculine', 0)}"
        f" / {spk_gender.get('female_feminine', 0)} / {spk_gender.get('n.d.', 0)}"
    )
    print(
        "  speakers with >=2/>=3/>=5/>=8 clips: "
        + " / ".join(str(int((vc >= k).sum())) for k in (2, 3, 5, 8))
        + f"   max clips for one: {int(vc.max())}"
    )
    print(
        f"  duration, median (p10-p90): {sub.dur.median():.2f} "
        f"({sub.dur.quantile(0.10):.2f}-{sub.dur.quantile(0.90):.2f})"
        f"  | clips 3-8s: {int(((sub.dur >= 3) & (sub.dur <= 8)).sum())}"
    )
    print(
        f"  words, median (p10-p90, max): {int(sub.nw.median())} "
        f"({int(sub.nw.quantile(0.10))}-{int(sub.nw.quantile(0.90))}, {int(sub.nw.max())})"
    )
    hist = np.histogram(sub.nw.clip(0, 25), bins=[0, 4, 7, 10, 13, 16, 26])[0]
    print(f"  length histogram <4/4-6/7-9/10-12/13-15/16+: {hist.tolist()}")
    print(
        f"  unique sentences: {sub.sentence.nunique()} of {len(sub)}"
        f"  | containing digits: "
        f"{int(sub.sentence.fillna('').str.contains(r'[0-9]').sum())}"
    )

    print("  by split:")
    for split in SPLITS:
        s = sub[sub.split == split]
        if s.empty:
            continue
        v = s.client_id.value_counts()
        g = s.groupby("client_id").gender.first().fillna("n.d.").value_counts()
        print(
            f"    {split:<5} {len(s):>7} clips, {s.client_id.nunique():>5} spk,"
            f" max/spk {int(v.max()):>5}, >=2 clips: {int((v >= 2).sum()):>5}"
            f"  | M{g.get('male_masculine', 0)} F{g.get('female_feminine', 0)}"
            f" ?{g.get('n.d.', 0)}"
        )

    # Historical: an early design required prompt and target from the same
    # speaker. That requirement was dropped (prompts and targets are decoupled),
    # but the pair ceiling remains a useful measure of how deep a slice is.
    dt = sub[sub.split.isin(["dev", "test"])].client_id.value_counts()
    print(f"  pair ceiling (sum floor(n/2) per speaker): "
          f"whole pool {int((vc // 2).sum())}, dev+test {int((dt // 2).sum())}")

    # Split disjointness. Broken upstream for pt in CV17.
    spk = {s: set(df[df.split == s].client_id) for s in SPLITS}
    sent = {s: set(df[df.split == s].sentence) for s in SPLITS}
    print(
        "  speaker overlap train&test / train&dev / dev&test: "
        f"{len(spk['train'] & spk['test'])} / {len(spk['train'] & spk['dev'])}"
        f" / {len(spk['dev'] & spk['test'])}"
    )
    print(
        f"  test sentences also present in train: "
        f"{len(sent['train'] & sent['test'])} of {len(sent['test'])}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./cv",
                    help="directory holding {locale}/{split}.tsv")
    ap.add_argument("--langs", nargs="+", default=CODES)
    ap.add_argument("--labels", action="store_true",
                    help="print the full label inventory of the locale")
    args = ap.parse_args()

    for lang in args.langs:
        if lang not in LANGUAGES:
            sys.exit(f"unknown language: {lang}; available: {CODES}")
        report(args.data_dir, lang, show_labels=args.labels)


if __name__ == "__main__":
    main()
