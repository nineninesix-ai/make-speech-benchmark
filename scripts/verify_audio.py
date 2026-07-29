#!/usr/bin/env python3
"""S0 check — tar integrity and cross-check against the transcripts.

An interrupted download produces silent corruption: the archive still opens but
is truncated. So this walks every member header to the end rather than just
checking that a file exists, and verifies that the clips listed in the TSVs are
actually inside.

    python scripts/verify_audio.py
    python scripts/verify_audio.py --langs ky nl
"""

import argparse
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

SPLITS = ["train", "dev", "test"]


def scan_tars(audio_dir, lang):
    """Names of the mp3 files inside every shard, plus a list of broken archives."""
    names, broken = set(), []
    for split in SPLITS:
        d = audio_dir / lang / split
        if not d.is_dir():
            continue
        for tar_path in sorted(d.glob("*.tar")):
            try:
                with tarfile.open(tar_path) as tf:
                    n = 0
                    for member in tf:   # streaming walk catches truncation
                        if member.isfile():
                            names.add(Path(member.name).name)
                            n += 1
                    if n == 0:
                        broken.append((tar_path.name, "empty archive"))
            except Exception as exc:  # noqa: BLE001
                broken.append((tar_path.name, str(exc)[:60]))
    return names, broken


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent / "cv"
    ap.add_argument("--root", default=str(root))
    ap.add_argument("--langs", nargs="+", default=["es", "pt", "nl", "en", "ky"])
    args = ap.parse_args()

    root = Path(args.root)
    audio_dir = root / "audio"
    total_broken, total_missing = 0, 0

    for lang in args.langs:
        want = defaultdict(set)
        for split in SPLITS:
            tsv = root / lang / f"{split}.tsv"
            if not tsv.exists():
                continue
            df = pd.read_csv(tsv, sep="\t", quoting=3, dtype=str)
            want[split] = set(df.path.dropna())

        have, broken = scan_tars(audio_dir, lang)
        want_all = set().union(*want.values()) if want else set()
        missing = want_all - have
        extra = have - want_all

        print(f"[{lang}] {'OK' if not broken and not missing else 'PROBLEM'}")
        print(f"  mp3 in tars: {len(have)}   in TSVs (train+dev+test): {len(want_all)}")
        if extra:
            print(f"  extra in tars (not in train/dev/test): {len(extra)}")
        if missing:
            print(f"  MISSING from audio: {len(missing)}")
            for m in sorted(missing)[:5]:
                print(f"    {m}")
        for name, err in broken:
            print(f"  BROKEN {name}: {err}")
        total_broken += len(broken)
        total_missing += len(missing)

    print(f"\ntotal: {total_broken} broken archives, {total_missing} missing clips")
    return 1 if (total_broken or total_missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
