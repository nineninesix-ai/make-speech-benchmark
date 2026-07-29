#!/usr/bin/env python3
"""Generate the dataset-card YAML front matter from the packaged parquet.

Written rather than hand-maintained for one reason: in v1 the card's declared
schema and the shipped data had drifted apart — `qc_bandwidth_hz` was documented
as hertz while holding decibels, `tags` was declared `Sequence(string)` while the
parquet carried the degenerate `list<null>`, and `qc_snr_db` / `qc_asr_cer` were
declared and never computed. A card that is derived from the files cannot drift.

Feature types come from the arrow schema; the four audio columns are the only
ones that need a hint, since a struct of `{bytes, path}` is indistinguishable
from any other struct at the arrow level.

    python -m msbench.build.card_yaml --pack tts-bench-v2 > /tmp/front_matter.yaml
"""

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from msbench.languages import CODES
from msbench.paths import workspace

AUDIO_16K = {"prompt_audio", "gt_audio", "sim_ref_audio"}
AUDIO_NATIVE = {"prompt_audio_orig"}

ARROW_TO_HF = {
    pa.string(): "string", pa.large_string(): "string",
    pa.float64(): "float64", pa.float32(): "float32",
    pa.int64(): "int64", pa.int32(): "int32", pa.int16(): "int16",
    pa.int8(): "int8", pa.bool_(): "bool",
}


def feature(name: str, typ) -> dict:
    if name in AUDIO_16K:
        return {"name": name, "dtype": {"audio": {"sampling_rate": 16000}}}
    if name in AUDIO_NATIVE:
        return {"name": name, "dtype": "audio"}
    if pa.types.is_list(typ) or pa.types.is_large_list(typ):
        return {"name": name, "list": ARROW_TO_HF.get(typ.value_type, "string")}
    for k, v in ARROW_TO_HF.items():
        if typ.equals(k):
            return {"name": name, "dtype": v}
    raise SystemExit(f"unmapped arrow type for {name}: {typ}")


def front_matter(pack: Path) -> str:
    """The YAML block, read from the parquet the pack actually contains."""
    info, configs = [], []
    for lang in CODES:
        files = sorted((pack / lang).glob("main-*.parquet"))
        if not files:
            continue
        schema = pq.ParquetFile(files[0]).schema_arrow
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        nbytes = sum(f.stat().st_size for f in files)
        info.append({
            "config_name": lang,
            "features": [feature(n, t)
                         for n, t in zip(schema.names, schema.types, strict=True)],
            "splits": [{"name": "main", "num_bytes": nbytes,
                        "num_examples": rows}],
            "download_size": nbytes,
            "dataset_size": nbytes,
        })
        configs.append({"config_name": lang,
                        "data_files": [{"split": "main",
                                        "path": f"{lang}/main-*"}]})

    # The per-utterance metric artifacts, declared as configs so the dataset
    # viewer can browse them. Without this they are downloadable but invisible:
    # a reader on the dataset page would see the audio and the text and no
    # measurement anywhere. Each group is schema-homogeneous, and every row
    # carries `lang` plus the recogniser or encoder that produced it.
    extra = [
        ("metrics-wer", "reports/per_utterance/wer_*.parquet",
         "per-utterance WER/CER with S/D/I counts and both normalised strings"),
        ("metrics-sim", "reports/per_utterance/sim_*_anchor_*.parquet",
         "per-utterance SIM anchor for every encoder"),
        ("metrics-sim-floor", "reports/per_utterance/sim_*_floor_*.parquet",
         "impostor floor: cross-speaker prompt pairs"),
        ("metrics-quality", "reports/per_utterance/quality_*.parquet",
         "DNSMOS and the QC battery per audio stream"),
    ]
    for name, path, _ in extra:
        if list(pack.glob(path)):
            configs.append({"config_name": name,
                            "data_files": [{"split": "train", "path": path}]})

    front = {
        "dataset_info": info,
        "configs": configs,
        "license": "cc0-1.0",
        "task_categories": ["text-to-speech", "automatic-speech-recognition"],
        "language": ["en", "es", "pt", "nl", "ky"],
        "multilinguality": "multilingual",
        "source_datasets": ["fsicoli/common_voice_17_0"],
        "annotations_creators": ["crowdsourced", "machine-generated"],
        "tags": ["tts", "voice-cloning", "zero-shot-tts", "speech-synthesis",
                 "benchmark", "evaluation", "common-voice", "seed-tts-eval",
                 "speaker-similarity", "intelligibility"],
        "pretty_name": "Multilingual Speech Benchmark for Zero-Shot TTS",
        "size_categories": ["1K<n<10K"],
    }
    if not info:
        raise SystemExit(f"no parquet found under {pack} — nothing to describe")
    return yaml.safe_dump(front, sort_keys=False, allow_unicode=True,
                          default_flow_style=False, width=100)


def main():
    ap = argparse.ArgumentParser()
    here = workspace()
    ap.add_argument("--pack", default=str(here / "tts-bench-v2"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    text = front_matter(Path(args.pack))
    if args.out:
        Path(args.out).write_text(text)
        print(f"-> {args.out}  ({len(text.splitlines())} lines)")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
