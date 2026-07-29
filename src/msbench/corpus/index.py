#!/usr/bin/env python3
"""Index the tar shards: path -> (shard, byte offset, size).

Common Voice ships mp3 inside tar archives with no random access, but tar stores
the offset of every member. One pass over the headers builds an index, after
which any clip is a seek+read away — the 64 GB corpus never has to be unpacked
to disk.

    python -m msbench.corpus.index            # all locales -> cv/index.parquet
    python -m msbench.corpus.index --langs ky
"""

import argparse
import tarfile
import time
from pathlib import Path

import pandas as pd

SPLITS = ["train", "dev", "test"]


def index_lang(audio_dir, lang):
    rows = []
    for split in SPLITS:
        d = audio_dir / lang / split
        if not d.is_dir():
            continue
        for tar_path in sorted(d.glob("*.tar")):
            with tarfile.open(tar_path) as tf:
                for m in tf:
                    if m.isfile():
                        rows.append((Path(m.name).name, lang, split,
                                     str(tar_path.relative_to(audio_dir)),
                                     m.offset_data, m.size))
    return rows


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent / "cv"
    ap.add_argument("--root", default=str(root))
    ap.add_argument("--langs", nargs="+", default=["es", "pt", "nl", "en", "ky"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "index.parquet"
    audio_dir = root / "audio"

    frames = []
    for lang in args.langs:
        t0 = time.time()
        rows = index_lang(audio_dir, lang)
        frames.append(pd.DataFrame(
            rows, columns=["path", "lang", "split", "tar", "offset", "nbytes"]))
        print(f"  {lang}: {len(rows)} clips in {time.time() - t0:.0f}s", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"\n{len(df)} entries -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
