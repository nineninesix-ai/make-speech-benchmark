#!/usr/bin/env python3
"""S0 — download Common Voice 17 audio shards through huggingface_hub (xet).

Not curl: anonymous HTTP downloads from the Hub are throttled per connection,
and the connection does not fail, it silently degrades. Measured on this corpus:
84 MB/s at the start, 7.8 MB/s on average, individual connections down to
0.2 MB/s with no error in the log. The xet protocol fetches a file in parallel
chunks and reached ~260 MB/s over the same link without a token.

    python scripts/fetch_audio.py                 # all locales
    python scripts/fetch_audio.py --langs nl ky
    python scripts/fetch_audio.py --workers 2

A token is optional but raises the rate limits:  export HF_TOKEN=hf_...
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "fsicoli/common_voice_17_0"

# From the mirror's n_shards.json, train/dev/test only (other/invalidated are
# unused: they are not vote-validated).
NSHARDS = {
    "es": {"train": 9, "dev": 1, "test": 1},
    "pt": {"train": 4, "dev": 1, "test": 1},
    "nl": {"train": 1, "dev": 1, "test": 1},
    "en": {"train": 28, "dev": 1, "test": 1},
    "ky": {"train": 1, "dev": 1, "test": 1},
}


def shard_list(langs):
    return [f"audio/{lang}/{split}/{lang}_{split}_{i}.tar"
            for lang in langs
            for split, n in NSHARDS[lang].items()
            for i in range(n)]


def fetch(rel_path, root):
    dest = root / rel_path
    if dest.exists() and dest.stat().st_size > 0:
        return rel_path, dest.stat().st_size, 0.0, True
    t0 = time.time()
    hf_hub_download(REPO, rel_path, repo_type="dataset", local_dir=str(root))
    return rel_path, dest.stat().st_size, time.time() - t0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent / "cv"))
    ap.add_argument("--langs", nargs="+", default=list(NSHARDS))
    ap.add_argument("--workers", type=int, default=3,
                    help="files in parallel; xet parallelises chunks within a file")
    args = ap.parse_args()

    root = Path(args.root)
    shards = shard_list(args.langs)
    todo = [s for s in shards if not (root / s).exists()]
    print(f"shards total: {len(shards)}, to download: {len(todo)}, "
          f"workers: {args.workers}", flush=True)

    t0, total = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, s, root): s for s in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                rel, size, dt, cached = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  ! FAILED {futures[fut]}: {exc}", file=sys.stderr, flush=True)
                continue
            total += size
            rate = size / dt / 1e6 if dt > 0 else 0
            avg = total / (time.time() - t0) / 1e6
            print(f"  [{k}/{len(todo)}] {rel}  {size / 1e9:.2f} GB  "
                  f"{'(cached)' if cached else f'{rate:.0f} MB/s'}  avg {avg:.0f} MB/s",
                  flush=True)

    dt = time.time() - t0
    print(f"\ndone: {total / 1e9:.1f} GB in {dt / 60:.1f} min "
          f"({total / dt / 1e6:.0f} MB/s)")


if __name__ == "__main__":
    main()
