#!/usr/bin/env python3
"""WER/CER over a subset, with any registered ASR backend. Replaces run_wer.py.

What is different from v1, and why:

  * **the backend is a choice, not a constant** — Whisper's internal language
    model repairs mispronounced synthesis, so one recogniser cannot distinguish a
    model's errors from its own habits (**G-03**);
  * **no `repetition_penalty`** unless asked for — see `asr/whisper.py` (**D-02**);
  * **aggregation is computed by `metrics.py`**, which reports the corpus-level
    rate seed-tts-eval uses alongside v1's macro mean (**D-01**);
  * **nothing is dropped silently** — empty references are counted (**D-04**);
  * **the output carries the numerators and the speaker**, so any user can
    re-aggregate, slice, or bootstrap without a GPU and without re-running ASR.

The last point is the one that matters most for the published dataset: v1 shipped
aggregate markdown only, so no reader could check a number or compute their own.

    msbench-wer --lang en-US --audio gt --asr whisper
    msbench-wer --lang ky    --audio gt --asr mms
    msbench-wer --lang es-MX --audio synth --asr whisper \\
                            --synth-dir out/xtts/es-MX --model-name xtts-v2
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd  # noqa: E402

import msbench.asr as asr_registry
import msbench.stats as stats
from msbench.audio import read_cell
from msbench.metrics import score_pairs
from msbench.normalize import normalize
from msbench.paths import workspace

# Columns pulled from the pack. Reading the audio column of a 900 MB parquet is
# the expensive part, so the metadata read stays narrow.
META = ["utt", "text", "speaker_id", "gt_speaker_id", "gt_same_speaker",
        "speaker_gender", "len_bin", "n_words", "prompt_dur", "prompt_dur_bin",
        "has_gt", "has_sim_ref", "sim_ref_dur"]


def load_rows(pack: Path, lang: str, subset: str, audio: str, limit=None):
    """Metadata plus the audio column being scored, streamed one row group at a time."""
    import pyarrow.parquet as pq

    col = {"gt": "gt_audio", "prompt": "prompt_audio",
           "sim_ref": "sim_ref_audio"}.get(audio)
    f = pack / "data" / lang / f"{subset}.parquet"
    pf = pq.ParquetFile(f)
    have = set(pf.schema_arrow.names)
    cols = [c for c in META if c in have] + ([col] if col else [])

    n = 0
    for i in range(pf.num_row_groups):
        tbl = pf.read_row_group(i, columns=cols)
        for row in tbl.to_pylist():
            if limit and n >= limit:
                return
            n += 1
            yield row


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--pack", default=str(here / "tts-bench-v1"))
    ap.add_argument("--lang", required=True)
    ap.add_argument("--subset", default="main")
    ap.add_argument("--audio", choices=["gt", "synth"], default="gt")
    ap.add_argument("--asr", default="whisper")
    ap.add_argument("--synth-dir", default=None)
    ap.add_argument("--model-name", default=None,
                    help="name of the synthesis run, used in the output filename")
    ap.add_argument("--tag", default=None, help="extra suffix, e.g. 'legacy'")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    # v1 replay switches. Off by default; see asr/whisper.py.
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--chunk-length-s", type=float, default=None)
    # Hosted-backend switches. A network recogniser is bound by round-trip
    # latency rather than by GPU memory, so `--batch` does not govern its
    # throughput and its default timeout is far too short for long clips under
    # load; both need to be reachable from the command line.
    ap.add_argument("--workers", type=int, default=None,
                    help="concurrent requests, hosted backends only")
    ap.add_argument("--timeout", type=int, default=None,
                    help="per-request timeout in seconds, hosted backends only")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-preprocess", action="store_true",
                    help="score synthesis as delivered, skipping parity (D-05)")
    ap.add_argument("--no-ci", action="store_true", help="skip the bootstrap")
    args = ap.parse_args()

    pack = Path(args.pack)
    kw = {"batch": args.batch}
    if args.asr == "whisper":
        kw.update(repetition_penalty=args.repetition_penalty,
                  chunk_length_s=args.chunk_length_s)
    for name in ("workers", "timeout", "seed"):
        if getattr(args, name) is not None:
            kw[name] = getattr(args, name)
    backend = asr_registry.build(args.asr, **kw)
    if not backend.supports(args.lang):
        raise SystemExit(f"backend {args.asr!r} does not cover {args.lang!r}")

    rows, buf, t0 = [], [], time.time()
    stored = args.audio != "synth"

    def flush():
        if not buf:
            return
        hyps = backend.transcribe([w for _, w in buf], args.lang)
        for (meta, _), h in zip(buf, hyps, strict=True):
            rows.append({**meta, "hyp_raw": h, "missing": False})
        buf.clear()

    n_total = 0
    for r in load_rows(pack, args.lang, args.subset, args.audio, args.limit):
        n_total += 1
        meta = {k: r.get(k) for k in META if k in r}
        if args.audio == "synth":
            cell = Path(args.synth_dir) / f"{r['utt']}.wav"
        else:
            cell = r.get("gt_audio")
        wav = read_cell(cell, stored=stored,
                        trim=not args.no_preprocess, peak=not args.no_preprocess)
        if wav is None or len(wav) < 160:
            rows.append({**meta, "hyp_raw": None, "missing": True})
            continue
        buf.append((meta, wav))
        if len(buf) >= args.batch:
            flush()
            if len(rows) % 200 < args.batch:
                print(f"  {len(rows)}/{n_total}  ({time.time() - t0:.0f}s)",
                      flush=True)
    flush()

    res = pd.DataFrame(rows)
    ok = res[~res.missing].copy()
    ok["ref_norm"] = [normalize(t, args.lang) for t in ok.text]
    ok["hyp_norm"] = [normalize(t, args.lang) for t in ok.hyp_raw]

    agg = score_pairs(list(zip(ok.ref_norm, ok.hyp_norm, strict=True)),
                      n_missing=int(res.missing.sum()), utts=list(ok.utt))
    per = pd.DataFrame(agg.per_utt)
    # Re-attach the metadata: the cluster column is what makes a bootstrap
    # honest, and the breakdown columns are what let a reader slice the result.
    per = per.merge(ok.drop(columns=["ref_norm", "hyp_norm"]), on="utt",
                    how="left")
    per["asr"] = args.asr
    per["lang"] = args.lang
    per["audio"] = args.audio

    # For the human anchor the voice on the recording is the GT speaker, not the
    # prompt speaker; for synthesis it is the prompt speaker being cloned.
    cluster = "gt_speaker_id" if args.audio == "gt" else "speaker_id"
    if cluster not in per or per[cluster].isna().all() or (per[cluster] == "").all():
        cluster = "speaker_id"

    summary = agg.summary()
    summary.update(lang=args.lang, asr=args.asr, audio=args.audio,
                   n_rows_read=n_total, cluster_col=cluster,
                   n_clusters=int(per[cluster].nunique()),
                   config=getattr(backend, "config", lambda: {})(),
                   preprocess=not args.no_preprocess,
                   seconds=round(time.time() - t0, 1))

    if not args.no_ci and len(per) > 1:
        for label, fn in (("wer_corpus", stats.wer_corpus),
                          ("cer_corpus", stats.cer_corpus),
                          ("wer_macro", stats.macro("wer")),
                          ("catastrophic_rate", stats.rate_above("wer", 0.5))):
            p, lo, hi = stats.cluster_bootstrap(per, fn, cluster_col=cluster)
            summary[f"{label}_ci"] = [round(lo, 6), round(hi, 6)]

    name = f"wer_{args.lang}_{args.audio}"
    if args.model_name:
        name += f"_{args.model_name}"
    name += f"_{args.asr}"
    if args.tag:
        name += f"_{args.tag}"
    out = Path(args.out) if args.out else pack / "reports" / "v2" / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    per.to_parquet(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2,
                                                   ensure_ascii=False,
                                                   default=str))

    ci = summary.get("wer_corpus_ci")
    print(f"\n{args.lang} / {args.audio} / {args.asr}: "
          f"n={agg.n_scored}  missing={agg.n_missing}  empty_ref={agg.n_empty_ref}")
    print(f"  WER corpus  {agg.wer_corpus:.4f}"
          + (f"  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""))
    print(f"  WER macro   {agg.wer_macro:.4f}   (v1 aggregation)")
    print(f"  CER corpus  {agg.cer_corpus:.4f}   CER macro {agg.cer_macro:.4f}")
    print(f"  S/D/I       {agg.subs}/{agg.dels}/{agg.ins}  over "
          f"{agg.n_ref_words} reference words")
    print(f"  exact       {100 * agg.exact_match:.1f}%   "
          f"catastrophic (WER>0.5) {100 * agg.catastrophic_rate:.1f}%")
    print(f"  clusters    {summary['n_clusters']} by {cluster}")
    print(f"  {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
