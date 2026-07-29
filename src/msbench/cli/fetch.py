#!/usr/bin/env python3
"""Assemble the local evaluation pack from the published Hugging Face dataset.

The pipeline scripts read one file per subset (``<pack>/data/<lang>/main.parquet``)
while the Hub stores each subset as several shards. This script downloads the
shards and streams them into the single-file layout, one row group at a time —
the audio columns make a naive ``pd.concat`` of a whole subset cost several GB of
RAM, and this machine has 31 GB total.

    msbench-fetch                 # all six subsets
    msbench-fetch --langs ky      # one subset
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from msbench.languages import CODES
from msbench.paths import workspace

REPO = "nineninesix/multilingual-speech-benchmark"


def assemble(src_dir: Path, lang: str, out: Path) -> int:
    """Concatenate `<src_dir>/<lang>/main-*.parquet` into `out`. Returns row count."""
    shards = sorted(src_dir.joinpath(lang).glob("main-*.parquet"))
    if not shards:
        raise SystemExit(f"no shards for {lang} under {src_dir}")
    out.parent.mkdir(parents=True, exist_ok=True)

    writer, n = None, 0
    try:
        for s in shards:
            pf = pq.ParquetFile(s)
            if writer is None:
                writer = pq.ParquetWriter(out, pf.schema_arrow, compression="zstd")
            for i in range(pf.num_row_groups):
                batch = pf.read_row_group(i)
                writer.write_table(batch)
                n += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--pack", default=str(here / "tts-bench-v1"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--langs", nargs="+", default=CODES)
    ap.add_argument("--force", action="store_true",
                    help="reassemble even if main.parquet already exists")
    args = ap.parse_args()

    print(f"downloading {args.repo} ...", flush=True)
    src = Path(snapshot_download(args.repo, repo_type="dataset",
                                 allow_patterns=["*/main-*.parquet", "*.md"]))
    print(f"  -> {src}", flush=True)

    pack = Path(args.pack)
    for lang in args.langs:
        out = pack / "data" / lang / "main.parquet"
        if out.exists() and not args.force:
            print(f"{lang:<6} exists, skipping ({out})")
            continue
        n = assemble(src, lang, out)
        mb = out.stat().st_size / 1e6
        print(f"{lang:<6} {n:5d} rows  {mb:7.1f} MB  -> {out}", flush=True)

    # Ship the published markdown reports alongside, for the v1 -> v2 diff.
    rep = pack / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    for md in (src / "reports").glob("*.md") if (src / "reports").exists() else []:
        (rep / f"v1_{md.name}").write_bytes(md.read_bytes())
    print(f"\npack ready: {pack}")


if __name__ == "__main__":
    main()
