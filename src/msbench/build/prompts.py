#!/usr/bin/env python3
"""S2 — QC the prompt candidates and pick one prompt per speaker.

Each speaker contributes a single fixed reference for all of their examples.
That keeps SIM variance down (the same voice is always cloned from the same
audio) and means a speaker only needs **one** clip to pass QC, which is why
speaker survival, not clip survival, is the number that matters for capacity.

Selection order among a speaker's passing clips: held-out (dev/test) first,
then the highest speech ratio.

QC scope is narrowed on purpose — see the module docstring of
`msbench.corpus.qc`, whose `read_clip` / `analyse` / `verdict` are reused here.

    python -m msbench.build.prompts --langs ky
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from msbench.corpus.qc import (
    MIN_HF_DB,
    MIN_SPEECH_RATIO,
    SR_TARGET,
    analyse,
    read_clip,
    verdict,
)
from msbench.languages import prompt_dur_bin
from msbench.paths import workspace


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--root", default=str(here / "cv"))
    ap.add_argument("--work", default=str(here / "work"))
    ap.add_argument("--langs", nargs="+", default=None)
    args = ap.parse_args()

    root, work = Path(args.root), Path(args.work)
    audio_root = root / "audio"
    index = (pd.read_parquet(root / "index.parquet")
             .drop_duplicates("path", keep="first").set_index("path"))

    vad_model, vad_utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                          trust_repo=True)
    from transformers import AutoFeatureExtractor, WavLMForXVector
    sv_name = "microsoft/wavlm-base-plus-sv"
    fe = AutoFeatureExtractor.from_pretrained(sv_name)
    sv = WavLMForXVector.from_pretrained(sv_name).eval().cuda()

    langs = args.langs or [p.stem.replace("_prompts", "")
                           for p in sorted(work.glob("*_prompts.parquet"))]

    for lang in langs:
        cand = pd.read_parquet(work / f"{lang}_prompts.parquet")
        cand = cand.join(index, on="path", rsuffix="_ix")
        cand = cand[cand.tar.notna()]
        print(f"\n=== {lang}: {len(cand)} candidates, "
              f"{cand.client_id.nunique()} speakers ===", flush=True)

        t0, rows, embs = time.time(), [], defaultdict(list)
        for row in cand.itertuples():
            try:
                wav, sr = read_clip(audio_root, row)
                m = analyse(wav, sr, vad_model, vad_utils)
            except Exception as exc:  # noqa: BLE001
                rows.append({"path": row.path, "client_id": row.client_id,
                             "ok": False, "reason": f"decode: {exc}"[:50]})
                continue
            mono16 = m.pop("_mono16")
            reasons = verdict(m, MIN_SPEECH_RATIO, MIN_HF_DB)
            m.update(path=row.path, client_id=row.client_id, ok=not reasons,
                     reason="; ".join(reasons), split=row.split,
                     sentence=row.sentence, gender=row.gender, age=row.age)
            rows.append(m)
            if not reasons:
                with torch.no_grad():
                    inp = fe(mono16, sampling_rate=SR_TARGET, return_tensors="pt")
                    e = sv(inp.input_values.cuda()).embeddings[0]
                embs[row.client_id].append(
                    (row.path, torch.nn.functional.normalize(e, dim=-1).cpu()))

        qc = pd.DataFrame(rows)

        # Speaker consistency: a client_id is not guaranteed to be one person.
        # Clips further than mu-2sigma from their speaker's centroid are dropped.
        cdist = {}
        for _cid, items in embs.items():
            M = torch.stack([v for _, v in items])
            c = torch.nn.functional.normalize(M.mean(0), dim=-1)
            cos = (M @ c).numpy()
            thr = cos.mean() - 2 * cos.std() if len(cos) > 2 else -1.0
            for (p, _), v in zip(items, cos, strict=True):
                cdist[p] = (float(v), bool(v < thr))
        qc["spk_cos"] = qc.path.map(lambda p: cdist.get(p, (np.nan, False))[0])
        qc["spk_outlier"] = qc.path.map(lambda p: cdist.get(p, (np.nan, False))[1])
        qc.loc[qc.spk_outlier, "ok"] = False
        qc.loc[qc.spk_outlier, "reason"] = "speaker outlier"

        good = qc[qc.ok].copy()
        good["dur_bin"] = good.dur.map(prompt_dur_bin)
        good["ho"] = good.split.isin(["dev", "test"]).astype(int)
        best = (good.sort_values(["ho", "speech_ratio"], ascending=False)
                .groupby("client_id", as_index=False).head(1))

        qc.to_parquet(work / f"{lang}_qc.parquet", index=False)
        best.to_parquet(work / f"{lang}_prompt_sel.parquet", index=False)

        surv = 100 * best.client_id.nunique() / cand.client_id.nunique()
        print(f"  clips passed: {len(good)}/{len(qc)} "
              f"({100 * len(good) / len(qc):.0f}%)   in {time.time() - t0:.0f}s")
        print(f"  SPEAKERS with a prompt: {best.client_id.nunique()}"
              f"/{cand.client_id.nunique()} ({surv:.0f}%)")
        print(f"  held-out among prompts: {int(best.ho.sum())}"
              f" ({100 * best.ho.mean():.0f}%)")
        print(f"  duration strata: {best.dur_bin.value_counts().to_dict()}")
        for r, n in qc[~qc.ok].reason.value_counts().head(5).items():
            print(f"    {r}: {n}")


if __name__ == "__main__":
    main()
