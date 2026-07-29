#!/usr/bin/env python3
"""S3 — example selection as constrained optimisation, not `sample()`.

Because prompts and targets are decoupled, they are selected **independently**
and then zipped:

  A. Texts — lazy greedy diphone set-cover under length-bin and prosody quotas.
     The coverage function is submodular, so greedy gives a 1-1/e approximation.
  B. Prompts — one per speaker, under a female-share quota and a per-speaker cap.
  C. Zip.

    python -m msbench.build.select --langs ky
"""

import argparse
import heapq
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from msbench.languages import CAP, LEN_BINS, LEN_TARGET, N_MAIN, TARGET_F
from msbench.paths import workspace

# Questions are 0.2-9% and exclamations 0.3-1.8% of the Common Voice pool
# depending on language, so a ">=10% of each" quota is unreachable everywhere.
# Take everything available up to this cap instead.
PROSODY_CAP = 0.10
RNG = np.random.default_rng(20260725)


def diphones_of(phones_str):
    toks = phones_str.split()
    return set(zip(toks, toks[1:], strict=False))


def greedy_texts(texts, n_target, len_target):
    """Lazy greedy selection by diphone-coverage gain, under length-bin quotas."""
    quota = {b: int(round(n_target * f)) for b, f in zip(LEN_BINS, len_target, strict=True)}
    diff = n_target - sum(quota.values())          # fix rounding
    for b in sorted(quota, key=lambda x: -quota[x]):
        if diff == 0:
            break
        quota[b] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1

    dips = [diphones_of(p) for p in texts.phones]
    heldout = texts.split.isin(["dev", "test"]).to_numpy()
    bins = texts.len_bin.to_numpy()
    punct = texts.punct_type.to_numpy()

    covered, chosen = set(), []
    taken_bin = Counter()
    stale = np.zeros(len(dips), bool)

    # Take questions and exclamations *before* the greedy phase: they are rare
    # enough that coverage-driven selection would simply never pick them, and
    # prosody would go untested altogether.
    prosodic = np.where(np.isin(punct, ["question", "exclamation", "mixed"]))[0]
    cap_prosodic = int(PROSODY_CAP * n_target)
    for i in prosodic[np.argsort(-heldout[prosodic].astype(int))]:
        if len(chosen) >= cap_prosodic:
            break
        b = bins[i]
        if taken_bin[b] >= quota[b]:
            continue
        covered |= dips[i]
        chosen.append(i)
        taken_bin[b] += 1
        stale[i] = True

    # Lazy greedy: (-gain, tie, idx); the tie-breaker prefers held-out clips.
    heap = [(-len(d - covered), -int(h), i)
            for i, (d, h) in enumerate(zip(dips, heldout, strict=True)) if not stale[i]]
    heapq.heapify(heap)

    while heap and len(chosen) < n_target:
        neg_gain, tie, i = heapq.heappop(heap)
        if stale[i]:
            continue
        b = bins[i]
        if taken_bin[b] >= quota[b]:
            stale[i] = True
            continue
        real = len(dips[i] - covered)
        if real < -neg_gain:                # estimate stale, push back
            heapq.heappush(heap, (-real, tie, i))
            continue
        covered |= dips[i]
        chosen.append(i)
        taken_bin[b] += 1
        stale[i] = True

    # Coverage saturated before the quotas filled — top up, preferring held-out.
    if len(chosen) < n_target:
        rest = np.where(~stale)[0]
        score = heldout[rest].astype(int) * 2 + RNG.random(len(rest))
        for i in rest[np.argsort(-score)]:
            if len(chosen) >= n_target:
                break
            b = bins[i]
            if taken_bin[b] >= quota[b]:
                continue
            covered |= dips[i]
            chosen.append(i)
            taken_bin[b] += 1

    all_dips = set().union(*dips) if dips else set()
    return texts.iloc[chosen].copy(), covered, all_dips, quota


def pick_prompts(prompts, n_target, cap, target_f):
    """N prompt slots: <=cap per speaker, female quota, held-out preference.

    Female speakers are filled up to `cap` first, everyone else round-robin from
    one slot. A plain round-robin over all speakers gave 23% female voices for
    en-US against a pool ceiling of 45%; filling *both* groups to cap would have
    hit the quota but halved the number of distinct voices. This does both.
    """
    p = prompts.copy()
    p["is_f"] = p.gender.fillna("") == "female_feminine"
    p["ho"] = p.split.isin(["dev", "test"]).astype(int)
    n_spk = len(p)
    if n_spk * cap < n_target:
        raise SystemExit(f"not enough speakers: {n_spk}*{cap} < {n_target}")

    need_f = int(np.ceil(target_f * n_target)) if target_f else 0
    p["score"] = p.ho * 1.5 + RNG.random(n_spk)
    fem = p[p.is_f].sort_values("score", ascending=False)
    rest = p[~p.is_f].sort_values("score", ascending=False)

    slots, used = [], Counter()

    def take(df, rounds):
        for rnd in range(rounds):
            for row in df.itertuples():
                if len(slots) >= n_target:
                    return
                if used[row.client_id] > rnd:
                    continue
                slots.append(row.Index)
                used[row.client_id] += 1

    if need_f:
        take(fem, min(cap, int(np.ceil(need_f / max(1, len(fem))))))
    take(rest, 1)
    take(pd.concat([fem, rest]), cap)      # top up to N, still respecting cap

    sel = prompts.loc[slots].copy()
    sel["n_in_subset"] = sel.client_id.map(Counter(sel.client_id))
    return sel


def run(lang, work, out_dir):
    texts = pd.read_parquet(work / f"{lang}_texts.parquet")
    prompts = pd.read_parquet(work / f"{lang}_prompt_sel.parquet")
    n = N_MAIN[lang]

    sel_t, covered, all_dips, quota = greedy_texts(texts, n, LEN_TARGET[lang])
    sel_p = pick_prompts(prompts, n, CAP[lang], TARGET_F[lang])

    sel_t = sel_t.reset_index(drop=True)
    sel_p = (sel_p.reset_index(drop=True)
             .iloc[RNG.permutation(len(sel_p))].reset_index(drop=True))
    m = min(len(sel_t), len(sel_p))
    pair = pd.DataFrame({
        "utt": [f"{lang}_{i + 1:04d}" for i in range(m)],
        "lang": lang,
        "text": sel_t.sentence[:m].values,
        "text_path": sel_t.path[:m].values,
        "text_split": sel_t.split[:m].values,
        "text_speaker": sel_t.client_id[:m].values,
        "n_words": sel_t.n_words[:m].values,
        "len_bin": sel_t.len_bin[:m].values,
        "punct_type": sel_t.punct_type[:m].values,
        "phones": sel_t.phones[:m].values,
        "n_phones": sel_t.n_phones[:m].values,
        "prompt_path": sel_p.path[:m].values,
        "prompt_text": sel_p.sentence[:m].values,
        "prompt_dur": sel_p.dur[:m].values,
        "prompt_split": sel_p.split[:m].values,
        "prompt_speaker": sel_p.client_id[:m].values,
        "prompt_gender": sel_p.gender[:m].values,
        "prompt_age": sel_p.age[:m].values,
        "prompt_dur_bin": sel_p.dur_bin[:m].values,
        "speaker_n_in_subset": sel_p.n_in_subset[:m].values,
    })
    pair["gt_same_speaker"] = pair.text_speaker == pair.prompt_speaker
    out_dir.mkdir(parents=True, exist_ok=True)
    pair.to_parquet(out_dir / f"{lang}_selected.parquet", index=False)

    cov = 100 * len(covered) / len(all_dips) if all_dips else 0
    fem = (pair.prompt_gender == "female_feminine").mean()
    print(f"\n=== {lang}: selected {len(pair)} from a pool of {len(texts)} texts ===")
    print(f"  diphone coverage: {len(covered)}/{len(all_dips)} ({cov:.1f}%)")
    print("  length bins: "
          f"{pair.len_bin.value_counts().reindex(LEN_BINS).fillna(0).astype(int).to_dict()}")
    print(f"       target: {quota}")
    print(f"  speakers: {pair.prompt_speaker.nunique()}, "
          f"max/speaker: {pair.speaker_n_in_subset.max()} (cap {CAP[lang]})")
    print(f"  female voices: {100 * fem:.0f}%"
          f" (target {TARGET_F[lang] if TARGET_F[lang] else 'best-effort'})")
    print(f"  prompt strata: {pair.prompt_dur_bin.value_counts().to_dict()}")
    print(f"  punctuation: {pair.punct_type.value_counts().to_dict()}")
    print(f"  held-out: texts {100 * pair.text_split.isin(['dev', 'test']).mean():.0f}%, "
          f"prompts {100 * pair.prompt_split.isin(['dev', 'test']).mean():.0f}%")
    print(f"  unique texts: {pair.text.nunique()}/{len(pair)}")
    return pair


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--work", default=str(here / "work"))
    ap.add_argument("--out", default=str(here / "work"))
    ap.add_argument("--langs", nargs="+", default=None)
    args = ap.parse_args()

    work = Path(args.work)
    langs = args.langs or [p.stem.replace("_prompt_sel", "")
                           for p in sorted(work.glob("*_prompt_sel.parquet"))]
    for lang in langs:
        run(lang, work, Path(args.out))


if __name__ == "__main__":
    main()
