#!/usr/bin/env python3
"""Speaker similarity with any registered encoder, plus the missing baseline.

Four quantities, three of which v1 does not have:

    sim_o        cos(synthesis, prompt_audio)             the model's score
    sim_anchor   cos(sim_ref_audio, prompt_audio)         the human anchor
    sim_floor    cos(prompt_i, prompt_j), i != j speaker  impostor baseline  (G-02)
    sim_gt       cos(synthesis, gt_audio)                 only where gt_same_speaker

**Why the floor is not optional (G-02).** v1 reports an anchor and no baseline, so
an absolute SIM cannot be read at all. "0.87 against a ceiling of 0.93" sounds
like 94 % of the way there; whether it is good depends entirely on what two
*different* speakers score, and v1 never measured that. The normalised score is
the readable one:

    sim_norm = (sim_o - sim_floor) / (sim_anchor - sim_floor)

0 means indistinguishable from an impostor, 1 means as close as two recordings of
the same person. The floor is reported as a mean and a p95, because p95 is the
practical false-accept level: a system that lands there is being confused with
strangers one time in twenty.

The floor is sampled over cross-speaker prompt pairs *within* a language, so it
carries that language's channel and recording conditions rather than a generic
notion of "different voice".

**Effective n.** `msbench.build.assemble:181` gives every example of a speaker the
same `sim_ref_audio` clip, and `msbench.build.prompts` gives every example of a
speaker the same prompt. So all rows of one speaker carry the *same* anchor value:
the anchor's `n=909` for en-US is 909 rows over far fewer distinct measurements.
This script reports both counts and clusters the bootstrap by speaker (**D-20**).

**Preprocessing.** Pack audio is already 16 kHz, VAD-trimmed and peak-normalised,
so it is fed as stored; synthesis goes through the full parity pipeline in
`audio.prepare` (**D-03**, **D-05**). Prompt embeddings are cached by speaker
(**D-06**) — v1 recomputed one prompt embedding up to seven times.

    msbench-sim --lang ky --mode anchor --encoder wavlm_ft
    msbench-sim --lang ky --mode floor  --encoder ecapa --pairs 5000
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import msbench.sim as sim_registry
import msbench.stats as stats
from msbench.audio import read_cell
from msbench.paths import workspace

META = ["utt", "speaker_id", "speaker_gender", "len_bin", "prompt_dur",
        "prompt_dur_bin", "sim_ref_dur", "has_sim_ref", "has_gt",
        "gt_same_speaker"]


def sim_ref_dur_bin(seconds: float) -> str:
    """Terciles of `sim_ref_dur`, so SIM breakdowns can control for it (D-21).

    The card bins SIM by `prompt_dur` only, leaving the *other* clip in the pair
    — which goes down to ~2 s — uncontrolled. That is a live confounder for the
    Kyrgyz non-monotonicity.
    """
    if not np.isfinite(seconds) or seconds <= 0:
        return "n/a"
    if seconds < 3.5:
        return "<3.5"
    return "3.5-4.5" if seconds < 4.5 else ">=4.5"


def read_pack(pack: Path, lang: str, subset: str, cols: list[str], limit=None):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(pack / "data" / lang / f"{subset}.parquet")
    have = set(pf.schema_arrow.names)
    take = [c for c in cols if c in have]
    out, n = [], 0
    for i in range(pf.num_row_groups):
        for row in pf.read_row_group(i, columns=take).to_pylist():
            if limit and n >= limit:
                return out
            n += 1
            out.append(row)
    return out


def embed_by_speaker(enc, rows, column, key="speaker_id"):
    """One embedding per speaker for a clip column that is constant per speaker.

    Fixes D-06: v1 re-embedded the same prompt for every example of the speaker,
    up to seven times in pt-BR.
    """
    first, order = {}, []
    for r in rows:
        k = r[key]
        if k in first or r.get(column) is None:
            continue
        wav = read_cell(r[column], stored=True)
        if wav is None or len(wav) < 400:
            continue
        first[k] = wav
        order.append(k)
    if not order:
        return {}
    embs = enc.embed([first[k] for k in order])
    return dict(zip(order, embs, strict=True))


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--pack", default=str(here / "tts-bench-v1"))
    ap.add_argument("--lang", required=True)
    ap.add_argument("--subset", default="main")
    ap.add_argument("--mode", choices=["anchor", "floor", "synth"], default="anchor")
    ap.add_argument("--encoder", default="wavlm_sv",
                    choices=sim_registry.ENCODERS)
    ap.add_argument("--synth-dir", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--pairs", type=int, default=5000, help="floor mode only")
    ap.add_argument("--seed", type=int, default=stats.SEED)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no-preprocess", action="store_true",
                    help="score synthesis as delivered, skipping parity (D-05)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pack = Path(args.pack)
    t0 = time.time()
    enc = sim_registry.build(args.encoder, batch=args.batch)

    cols = META + ["prompt_audio"]
    if args.mode == "anchor":
        cols.append("sim_ref_audio")
    if args.mode == "synth":
        cols.append("gt_audio")
    rows = read_pack(pack, args.lang, args.subset, cols, args.limit)

    summary = {"lang": args.lang, "mode": args.mode, "encoder": args.encoder,
               "encoder_scale": enc.scale, "n_rows_read": len(rows)}

    # ---- floor: cross-speaker prompt pairs ---------------------------------
    if args.mode == "floor":
        pe = embed_by_speaker(enc, rows, "prompt_audio")
        spk = sorted(pe)
        if len(spk) < 2:
            raise SystemExit(f"{args.lang}: need >=2 speakers for a floor")
        M = np.stack([pe[s] for s in spk])
        rng = np.random.default_rng(args.seed)
        n_pairs = min(args.pairs, len(spk) * (len(spk) - 1) // 2)
        seen, out = set(), []
        # Sample without replacement over unordered pairs: a duplicated pair
        # would be counted twice and narrow the distribution for free.
        while len(out) < n_pairs and len(seen) < len(spk) * (len(spk) - 1) // 2:
            i, j = rng.integers(0, len(spk), size=2)
            if i == j or (min(i, j), max(i, j)) in seen:
                continue
            seen.add((min(i, j), max(i, j)))
            out.append({"speaker_a": spk[i], "speaker_b": spk[j],
                        "sim_floor": float(M[i] @ M[j])})
        res = pd.DataFrame(out)
        res["lang"] = args.lang
        res["encoder"] = args.encoder
        res["mode"] = args.mode
        s = res.sim_floor
        summary.update(n_speakers=len(spk), n_pairs=len(res),
                       floor_mean=float(s.mean()), floor_median=float(s.median()),
                       floor_p95=float(s.quantile(.95)),
                       floor_p99=float(s.quantile(.99)),
                       floor_max=float(s.max()), floor_std=float(s.std()))
        name = f"sim_{args.lang}_floor_{args.encoder}"

    else:
        # ---- anchor / synth ------------------------------------------------
        pe = embed_by_speaker(enc, rows, "prompt_audio")
        recs = []
        if args.mode == "anchor":
            rows = [r for r in rows if r.get("has_sim_ref")]
            re_ = embed_by_speaker(enc, rows, "sim_ref_audio")
            for r in rows:
                a, b = pe.get(r["speaker_id"]), re_.get(r["speaker_id"])
                if a is None or b is None:
                    continue
                recs.append({**{k: r.get(k) for k in META},
                             "sim": float(a @ b)})
        else:
            if not args.synth_dir:
                raise SystemExit("--synth-dir is required in synth mode")
            for r in rows:
                a = pe.get(r["speaker_id"])
                wav = read_cell(Path(args.synth_dir) / f"{r['utt']}.wav",
                                stored=False, trim=not args.no_preprocess,
                                peak=not args.no_preprocess)
                if a is None or wav is None:
                    continue
                e = enc.embed([wav])[0]
                rec = {**{k: r.get(k) for k in META}, "sim": float(a @ e)}
                if r.get("gt_same_speaker") and r.get("has_gt"):
                    g = read_cell(r.get("gt_audio"), stored=True)
                    if g is not None:
                        rec["sim_gt"] = float(e @ enc.embed([g])[0])
                recs.append(rec)

        res = pd.DataFrame(recs)
        if res.empty:
            raise SystemExit("no rows scored")
        res["sim_ref_dur_bin"] = res.sim_ref_dur.map(sim_ref_dur_bin)
        # Carry the run identity in the data, not only in the filename: without
        # these two columns the per-utterance artifacts cannot be concatenated
        # into one browsable table, because a row gives no way to tell which
        # encoder produced it.
        res["lang"] = args.lang
        res["encoder"] = args.encoder
        res["mode"] = args.mode
        s = res.sim
        # The anchor is constant within a speaker by construction (one prompt and
        # one sim-ref clip per speaker), so the number of DISTINCT measurements is
        # the speaker count, not the row count.
        summary.update(n_rows=len(res), n_speakers=int(res.speaker_id.nunique()),
                       n_distinct_values=int(res.sim.round(6).nunique()),
                       mean=float(s.mean()), median=float(s.median()),
                       p05=float(s.quantile(.05)), p95=float(s.quantile(.95)),
                       std=float(s.std()))
        p, lo, hi = stats.cluster_bootstrap(res, stats.macro("sim"),
                                            cluster_col="speaker_id")
        summary["mean_ci"] = [round(lo, 6), round(hi, 6)]
        name = f"sim_{args.lang}_{args.mode}"
        if args.model_name:
            name += f"_{args.model_name}"
        name += f"_{args.encoder}"

    summary["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out) if args.out else pack / "reports" / "v2" / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{args.lang} / {args.mode} / {args.encoder}")
    for k, v in summary.items():
        if k in ("lang", "mode", "encoder"):
            continue
        print(f"  {k:<18} {v}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
