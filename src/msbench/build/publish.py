#!/usr/bin/env python3
"""Publish the v2 pack to the Hugging Face Hub.

Refuses to run unless `msbench.build.validate` passes, because the promise v2 makes
— that only the description changed, not the data — is the one thing a release
must not get wrong.

Stale files from v1 (the old multi-shard parquet layout, and the reports that
were superseded) are removed in the same commit, so the repository never holds
two versions of the same subset.

    msbench-publish --dry-run
    msbench-publish
"""

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

from msbench.paths import workspace

REPO = "nineninesix/multilingual-speech-benchmark"

# Remote paths that may be left over from v1. Anything matching these that is not
# part of this upload is deleted in the same commit.
DELETE = ["*/main-*.parquet", "reports/*.md"]

MESSAGE = """v2.0 — corpus-level metrics, impostor floor, confidence intervals, full card

Data unchanged: audio byte-identical in all three streams, same rows in the same
order, `utt` stable. Verified by msbench.build.validate before this upload.

Schema: qc_hf_energy_db replaces the misnamed qc_bandwidth_hz (it holds decibels,
not hertz); tags becomes list<string> from the degenerate list<null>;
sim_ref_dur_bin added; gt_qc_* and simref_qc_* report the quality of the two
audio streams that never faced the prompt-acceptance filters; Kyrgyz phones use
IPA dentals instead of espeak markers; qc_snr_db and qc_asr_cer removed as they
were 100% null in every subset.

Metrics: WER is now corpus-level (seed-tts-eval convention) with the macro form
retained for continuity; every headline number carries a 95% cluster bootstrap
over speakers; a second recogniser (MMS, CTC without a language model) reads all
six subsets; SIM ships for three encoders with an impostor floor, without which
an absolute SIM cannot be read at all.

Reports: per-utterance parquet for every run now ships in reports/per_utterance/,
so any number in the card can be recomputed, sliced or re-tested without a GPU.
Coverage reports are in English.

Pipeline: https://github.com/nineninesix-ai/make-speech-benchmark
"""


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--pack", default=str(here / "tts-bench-v2"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    pack = Path(args.pack)
    if not (pack / "README.md").exists():
        raise SystemExit(f"no README.md in {pack} — run msbench-card first")

    if not args.skip_validate:
        print("validating the pack against v1 ...", flush=True)
        r = subprocess.run([sys.executable, "-m", "msbench.build.validate",
                            "--v2", str(pack)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.stdout.write(r.stdout[-3000:])
            # A validator that could not start is not a pack that failed
            # validation, and the two must never print the same message: one
            # means do not publish, the other means fix the invocation.
            if "VALIDATION FAILED" not in r.stdout:
                sys.stderr.write(r.stderr[-2000:])
                raise SystemExit("the validator did not run — nothing was uploaded")
            raise SystemExit("validation failed — nothing was uploaded")
        print("  passed\n")

    files = sorted(p for p in pack.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB -> {args.repo}")
    if args.dry_run:
        print("dry run — nothing uploaded")
        return

    api = HfApi()
    url = api.upload_folder(
        folder_path=str(pack), repo_id=args.repo, repo_type="dataset",
        commit_message=MESSAGE.split("\n")[0],
        commit_description="\n".join(MESSAGE.split("\n")[1:]).strip(),
        delete_patterns=DELETE,
    )
    print(f"\npublished: {url}")


if __name__ == "__main__":
    main()
