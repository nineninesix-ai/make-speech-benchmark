#!/usr/bin/env python3
"""Backfill columns that are computed after packaging.

  * qc_dnsmos_*          from msbench.cli.quality
  * qc_spk_centroid_dist from the S2 speaker-consistency pass
  * speaker_accent_label the raw CV label, secondary accents included
  * gt_gender            gender of the ground-truth speaker
  * text_norm            packaging writes a lower() placeholder; the real
                         normaliser lives in msbench.normalize, and leaving the
                         placeholder would invite someone to treat it as ground
                         truth and compute a wrong WER
  * n_syllables          dropped, see below

    python -m msbench.build.backfill
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from msbench.normalize import normalize
from msbench.paths import workspace


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--pack", default=str(here / "tts-bench-v1"))
    ap.add_argument("--work", default=str(here / "work"))
    args = ap.parse_args()

    pack, work = Path(args.pack), Path(args.work)
    for d in sorted((pack / "data").iterdir()):
        lang = d.name
        f = d / "main.parquet"
        df = pd.read_parquet(f)

        dpath = pack / "reports" / f"dnsmos_{lang}.parquet"
        if dpath.exists():
            dn = pd.read_parquet(dpath)[
                ["utt", "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak"]]
            m = df[["utt"]].merge(dn, on="utt", how="left")
            df["qc_dnsmos_ovrl"] = m.dnsmos_ovrl.to_numpy(dtype="float32")
            df["qc_dnsmos_sig"] = m.dnsmos_sig.to_numpy(dtype="float32")
            df["qc_dnsmos_bak"] = m.dnsmos_bak.to_numpy(dtype="float32")

        qpath = work / f"{lang}_qc.parquet"
        if qpath.exists():
            qc = pd.read_parquet(qpath)[["path", "spk_cos"]].dropna()
            cos = dict(zip(qc.path, qc.spk_cos, strict=True))
            df["qc_spk_centroid_dist"] = df.prompt_cv_path.map(
                lambda p: cos.get(p, np.nan)).to_numpy(dtype="float32")

        # The raw label, not just the target one: a speaker may declare a
        # secondary accent ("Nederlands Nederlands,Amsterdams") and that is
        # informative.
        ppath = work / f"{lang}_prompts.parquet"
        if ppath.exists():
            pp = pd.read_parquet(ppath)[["path", "accents", "variant"]]
            lab = {r.path: (r.accents if isinstance(r.accents, str) and r.accents
                            else r.variant if isinstance(r.variant, str) else "")
                   for r in pp.itertuples()}
            df["speaker_accent_label"] = df.prompt_cv_path.map(
                lambda p: lab.get(p, ""))

        # The GT speaker is arbitrary by design, but knowing their gender lets
        # the WER topline be split by speaker gender.
        tpath = work / f"{lang}_texts.parquet"
        if tpath.exists():
            tt = pd.read_parquet(tpath)[["path", "gender"]]
            g = {"male_masculine": "male", "female_feminine": "female"}
            gmap = {r.path: g.get(r.gender, "unknown") for r in tt.itertuples()}
            df["gt_gender"] = df.text_cv_path.map(lambda p: gmap.get(p, "unknown"))

        # n_syllables is dropped rather than approximated. Merging adjacent
        # vowels breaks on Spanish hiatus ("¿Qué día de la semana es?" -> 7
        # instead of 9); counting every vowel breaks on diphthongs
        # ("...María Eugenia..." -> 26 instead of 21). Its correlation with
        # n_phones is 0.97-0.985, so it carried no independent information.
        # For speech rate use n_phones / gt_dur.
        df = df.drop(columns=["n_syllables"], errors="ignore")

        df["text_norm"] = [normalize(t, lang) for t in df.text]

        df.to_parquet(f, index=False)
        acc = int((df.speaker_accent_label.fillna("") != "").sum())
        gtg = int((df.gt_gender != "unknown").sum())
        print(f"{lang}: dnsmos {int(df.qc_dnsmos_ovrl.notna().sum())}, "
              f"centroid {int(df.qc_spk_centroid_dist.notna().sum())}, "
              f"accent {acc}, gt_gender {gtg}/{len(df)}, "
              f"columns {len(df.columns)}")


if __name__ == "__main__":
    main()
